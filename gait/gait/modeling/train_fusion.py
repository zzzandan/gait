# -*- coding: utf-8 -*-
"""
train_fusion.py

=========================================================
一、脚本功能说明
=========================================================
这个脚本用于训练“多特征融合步态识别模型（GaitFusionModel）”。

它建立在你已经完成的这些模块之上：

1. silhouette 分支：
   - silhouette_dataset.py
   - silhouette_branch.py
   - train_silhouette.py

2. skeleton 分支：
   - skeleton_dataset.py
   - skeleton_branch.py
   - train_skeleton.py

3. 融合模型：
   - fusion_model.py

本脚本的任务是：
- 同时读取“剪影序列”和“骨架序列”
- 按 person_id / seq_name 把两个模态严格配对
- 将两路输入一起送入 GaitFusionModel
- 计算融合分类损失
- 可选地加入两个单分支的辅助分类损失
- 训练并保存 best / last checkpoint

=========================================================
二、为什么这里要专门写一个配对数据集
=========================================================
单独训练 silhouette 或 skeleton 时，各自的数据集是独立读取的。
但做融合时，必须保证：

- 同一个 batch 里的 silhouette 序列和 skeleton 序列
  代表的是“同一个人、同一个序列”
- 标签一致
- 时间顺序一致
- 序列采样方式一致

因此，这里单独写一个 PairedFusionDataset，
专门负责把两个模态按同一条样本对齐。

=========================================================
三、当前训练损失设计
=========================================================
当前总损失由三部分组成：

1. 融合分类损失：
   fusion_loss = CE(fusion_logits, y)

2. 剪影辅助损失（可选）：
   silhouette_loss = CE(silhouette_logits, y)

3. 骨架辅助损失（可选）：
   skeleton_loss = CE(skeleton_logits, y)

总损失：
   total_loss = fusion_loss
              + silhouette_aux_weight * silhouette_loss
              + skeleton_aux_weight   * skeleton_loss

默认给两个辅助损失一个较小权重，是因为：
- 融合分类结果仍然是训练的主目标
- 但保留单分支监督，通常有助于小数据场景下训练更稳定

=========================================================
四、后续可扩展方向
=========================================================
1. 加载单分支预训练权重
2. 冻结单分支，仅训练 fusion head
3. 端到端联合微调
4. 加入 triplet loss / center loss
5. 改为注意力融合、门控融合等更复杂策略

=========================================================
"""

from __future__ import annotations

import json
import time
import random
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from fusion_model import GaitFusionModel


# =========================================================
# 1. 基础工具函数
# =========================================================
def ensure_dir(path: str | Path) -> None:
    """
    如果目录不存在，则递归创建。
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    """
    固定随机种子，尽可能提高实验的可复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(data: dict, save_path: str | Path) -> None:
    """
    将字典保存为 JSON 文件，便于记录训练配置和日志。
    """
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def format_seconds(seconds: float) -> str:
    """
    把秒数格式化为易读字符串。
    """
    seconds = int(seconds)

    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# =========================================================
# 2. 配对融合数据集
# =========================================================
class PairedFusionDataset(Dataset):
    """
    融合训练专用的“配对数据集”。

    目录要求：
    silhouette_root/
        person_id/
            seq_name/
                frame_xxx.png
                ...

    skeleton_root/
        person_id/
            seq_name/
                frame_xxx.npy
                ...

    核心原则：
    - 只保留“两个模态都存在”的 person_id
    - 在每个 person_id 下，只保留“两个模态都存在”的 seq_name
    - 每个样本 = 一对配好的 silhouette_seq + skeleton_seq
    - 返回：
        x_silhouette: [T, 1, H, W]
        x_skeleton:   [T, 2, H, W]
        y: long
    """

    def __init__(
        self,
        silhouette_root: str,
        skeleton_root: str,
        seq_len: int = 30,
        img_h: int = 64,
        img_w: int = 44,
        train: bool = True
    ):
        self.silhouette_root = Path(silhouette_root)
        self.skeleton_root = Path(skeleton_root)
        self.seq_len = seq_len
        self.img_h = img_h
        self.img_w = img_w
        self.train = train

        self.samples: List[dict] = []
        self.label_map: Dict[str, int] = {}

        self._build_index()

    def _build_index(self) -> None:
        """
        扫描两个根目录，并建立严格对齐的样本索引。

        为什么要以“共同 person_id + 共同 seq_name”为准：
        - 这样可以最大程度保证融合训练时两路数据是一一对应的
        - 防止某一路多出一些序列，导致标签对不上
        """
        if not self.silhouette_root.exists():
            raise FileNotFoundError(f"找不到剪影数据目录: {self.silhouette_root}")

        if not self.skeleton_root.exists():
            raise FileNotFoundError(f"找不到骨架数据目录: {self.skeleton_root}")

        silhouette_person_ids = {
            d.name for d in self.silhouette_root.iterdir() if d.is_dir()
        }
        skeleton_person_ids = {
            d.name for d in self.skeleton_root.iterdir() if d.is_dir()
        }

        # 只保留两个模态共同拥有的类别
        common_person_ids = sorted(silhouette_person_ids & skeleton_person_ids)

        if len(common_person_ids) == 0:
            raise ValueError(
                "未找到共同的 person_id，请检查 silhouette_root 和 skeleton_root 的目录结构。"
            )

        self.label_map = {pid: idx for idx, pid in enumerate(common_person_ids)}

        for pid in common_person_ids:
            sil_person_dir = self.silhouette_root / pid
            ske_person_dir = self.skeleton_root / pid

            sil_seq_names = {
                d.name for d in sil_person_dir.iterdir() if d.is_dir()
            }
            ske_seq_names = {
                d.name for d in ske_person_dir.iterdir() if d.is_dir()
            }

            common_seq_names = sorted(sil_seq_names & ske_seq_names)

            for seq_name in common_seq_names:
                sil_seq_dir = sil_person_dir / seq_name
                ske_seq_dir = ske_person_dir / seq_name

                sil_frame_paths = sorted([
                    str(p) for p in sil_seq_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
                ])

                ske_frame_paths = sorted([
                    str(p) for p in ske_seq_dir.iterdir()
                    if p.is_file() and p.suffix.lower() == ".npy"
                ])

                # 两路都必须有帧
                if len(sil_frame_paths) == 0 or len(ske_frame_paths) == 0:
                    continue

                self.samples.append({
                    "person_id": pid,
                    "label": self.label_map[pid],
                    "seq_name": seq_name,
                    "sil_frame_paths": sil_frame_paths,
                    "ske_frame_paths": ske_frame_paths
                })

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_indices(self, common_num_frames: int) -> np.ndarray:
        """
        从“共同可对齐长度”里采样 seq_len 帧。

        这里和单分支训练的设计思路一致：
        - train=True  : 随机采样
        - train=False : 均匀采样

        为什么使用 common_num_frames：
        - 因为两个模态的序列长度可能略有差异
        - 为了稳妥起见，只按两者最短公共长度来采样
        """
        if common_num_frames >= self.seq_len:
            if self.train:
                indices = np.sort(
                    np.random.choice(common_num_frames, self.seq_len, replace=False)
                )
            else:
                indices = np.linspace(0, common_num_frames - 1, self.seq_len).astype(int)
        else:
            if self.train:
                indices = np.sort(
                    np.random.choice(common_num_frames, self.seq_len, replace=True)
                )
            else:
                indices = np.linspace(0, common_num_frames - 1, self.seq_len)
                indices = np.round(indices).astype(int)

        return indices

    def _load_one_silhouette_frame(self, img_path: str) -> np.ndarray:
        """
        读取一帧剪影图，输出 [1, H, W]。
        """
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取剪影图像: {img_path}")

        img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # [1, H, W]
        return img

    def _load_one_skeleton_frame(self, npy_path: str) -> np.ndarray:
        """
        读取一帧 Skeleton Map，输出 [2, H, W]。
        """
        arr = np.load(npy_path)

        if arr.ndim != 3:
            raise ValueError(
                f"Skeleton Map 文件维度错误，期望 [C,H,W]，实际为 {arr.shape}，文件: {npy_path}"
            )

        c, h, w = arr.shape
        if c != 2:
            raise ValueError(
                f"Skeleton Map 通道数应为 2，实际为 {c}，文件: {npy_path}"
            )

        if h != self.img_h or w != self.img_w:
            resized = np.zeros((2, self.img_h, self.img_w), dtype=np.float32)
            for i in range(2):
                resized[i] = cv2.resize(
                    arr[i], (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST
                ).astype(np.float32)
            arr = resized
        else:
            arr = arr.astype(np.float32)

        return arr

    def __getitem__(self, idx: int):
        """
        返回一个融合训练样本：
            x_silhouette: [T, 1, H, W]
            x_skeleton:   [T, 2, H, W]
            y: long
        """
        sample = self.samples[idx]

        sil_frame_paths = sample["sil_frame_paths"]
        ske_frame_paths = sample["ske_frame_paths"]
        label = sample["label"]

        # 只按公共最短长度做时间对齐
        common_num_frames = min(len(sil_frame_paths), len(ske_frame_paths))

        if common_num_frames <= 0:
            raise ValueError(
                f"样本无可用帧，person_id={sample['person_id']}, seq_name={sample['seq_name']}"
            )

        indices = self._sample_indices(common_num_frames)

        sil_frames = []
        ske_frames = []

        for i in indices:
            sil_frames.append(self._load_one_silhouette_frame(sil_frame_paths[i]))
            ske_frames.append(self._load_one_skeleton_frame(ske_frame_paths[i]))

        x_silhouette = np.stack(sil_frames, axis=0).astype(np.float32)  # [T,1,H,W]
        x_skeleton = np.stack(ske_frames, axis=0).astype(np.float32)    # [T,2,H,W]

        x_silhouette = torch.from_numpy(x_silhouette)
        x_skeleton = torch.from_numpy(x_skeleton)
        y = torch.tensor(label, dtype=torch.long)

        return x_silhouette, x_skeleton, y


# =========================================================
# 3. 数据集划分
# =========================================================
def build_stratified_split_indices(
    dataset: PairedFusionDataset,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """
    按类别分层切分 train / val。

    原因和前面单分支脚本完全一致：
    - 小数据集下，直接全局随机切分容易让验证集类别分布失衡
    - 分层切分更稳
    """
    rng = random.Random(seed)
    label_to_indices = defaultdict(list)

    for idx, sample in enumerate(dataset.samples):
        label = int(sample["label"])
        label_to_indices[label].append(idx)

    train_indices = []
    val_indices = []

    for _, indices in label_to_indices.items():
        indices = indices.copy()
        rng.shuffle(indices)

        n = len(indices)

        if n == 1:
            train_indices.extend(indices)
            continue

        n_val = int(round(n * val_ratio))
        n_val = max(1, n_val)
        n_val = min(n - 1, n_val)

        val_part = indices[:n_val]
        train_part = indices[n_val:]

        train_indices.extend(train_part)
        val_indices.extend(val_part)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices


def build_dataloaders(args) -> Tuple[DataLoader, DataLoader, PairedFusionDataset]:
    """
    创建训练集和验证集 DataLoader。

    这里仍然遵循：
    - train dataset: train=True -> 随机采样
    - val dataset  : train=False -> 均匀采样
    """
    train_dataset_full = PairedFusionDataset(
        silhouette_root=args.silhouette_root,
        skeleton_root=args.skeleton_root,
        seq_len=args.seq_len,
        img_h=args.img_h,
        img_w=args.img_w,
        train=True
    )

    val_dataset_full = PairedFusionDataset(
        silhouette_root=args.silhouette_root,
        skeleton_root=args.skeleton_root,
        seq_len=args.seq_len,
        img_h=args.img_h,
        img_w=args.img_w,
        train=False
    )

    if len(train_dataset_full) == 0:
        raise ValueError("配对融合数据集为空，请检查两个根目录是否正确对齐。")

    train_indices, val_indices = build_stratified_split_indices(
        dataset=train_dataset_full,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    if len(train_indices) == 0:
        raise ValueError("训练集为空，请检查 val_ratio 或数据集结构。")

    train_subset = Subset(train_dataset_full, train_indices)
    val_subset = Subset(val_dataset_full, val_indices)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, val_loader, train_dataset_full


# =========================================================
# 4. 权重加载函数
# =========================================================
def load_branch_checkpoint_if_needed(
    model: GaitFusionModel,
    silhouette_ckpt_path: str | None = None,
    skeleton_ckpt_path: str | None = None,
    device: torch.device | str = "cpu"
) -> None:
    """
    如果提供了单分支 checkpoint，就加载进去。

    用法：
    - 你已经分别训练好了 silhouette / skeleton 的 best.pth
    - 现在做融合训练时，可以先把两个 branch 初始化为预训练权重
    - 这样通常会更稳定，也更符合“先单分支、后融合”的训练逻辑
    """
    if silhouette_ckpt_path:
        ckpt = torch.load(silhouette_ckpt_path, map_location=device)
        model.silhouette_branch.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(f"[加载] silhouette 分支权重: {silhouette_ckpt_path}")

    if skeleton_ckpt_path:
        ckpt = torch.load(skeleton_ckpt_path, map_location=device)
        model.skeleton_branch.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(f"[加载] skeleton 分支权重: {skeleton_ckpt_path}")


# =========================================================
# 5. 单轮训练 / 验证
# =========================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    silhouette_aux_weight: float,
    skeleton_aux_weight: float
) -> Dict[str, float]:
    """
    训练一个 epoch。

    总损失：
        total_loss = fusion_loss
                   + silhouette_aux_weight * silhouette_loss
                   + skeleton_aux_weight   * skeleton_loss
    """
    model.train()

    running_total_loss = 0.0
    running_fusion_loss = 0.0
    running_sil_loss = 0.0
    running_ske_loss = 0.0
    running_correct = 0
    running_total = 0

    for x_silhouette, x_skeleton, y in loader:
        x_silhouette = x_silhouette.to(device, non_blocking=True)
        x_skeleton = x_skeleton.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(x_silhouette, x_skeleton)

        fusion_logits = out["logits"]
        silhouette_logits = out["silhouette_logits"]
        skeleton_logits = out["skeleton_logits"]

        fusion_loss = criterion(fusion_logits, y)
        sil_loss = criterion(silhouette_logits, y)
        ske_loss = criterion(skeleton_logits, y)

        total_loss = (
            fusion_loss
            + silhouette_aux_weight * sil_loss
            + skeleton_aux_weight * ske_loss
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        preds = torch.argmax(fusion_logits, dim=1)

        running_total_loss += total_loss.item() * batch_size
        running_fusion_loss += fusion_loss.item() * batch_size
        running_sil_loss += sil_loss.item() * batch_size
        running_ske_loss += ske_loss.item() * batch_size
        running_correct += (preds == y).sum().item()
        running_total += batch_size

    denom = max(running_total, 1)

    return {
        "total_loss": running_total_loss / denom,
        "fusion_loss": running_fusion_loss / denom,
        "silhouette_loss": running_sil_loss / denom,
        "skeleton_loss": running_ske_loss / denom,
        "acc": running_correct / denom
    }


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    silhouette_aux_weight: float,
    skeleton_aux_weight: float
) -> Dict[str, float]:
    """
    验证一个 epoch。
    """
    model.eval()

    running_total_loss = 0.0
    running_fusion_loss = 0.0
    running_sil_loss = 0.0
    running_ske_loss = 0.0
    running_correct = 0
    running_total = 0

    for x_silhouette, x_skeleton, y in loader:
        x_silhouette = x_silhouette.to(device, non_blocking=True)
        x_skeleton = x_skeleton.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(x_silhouette, x_skeleton)

        fusion_logits = out["logits"]
        silhouette_logits = out["silhouette_logits"]
        skeleton_logits = out["skeleton_logits"]

        fusion_loss = criterion(fusion_logits, y)
        sil_loss = criterion(silhouette_logits, y)
        ske_loss = criterion(skeleton_logits, y)

        total_loss = (
            fusion_loss
            + silhouette_aux_weight * sil_loss
            + skeleton_aux_weight * ske_loss
        )

        batch_size = y.size(0)
        preds = torch.argmax(fusion_logits, dim=1)

        running_total_loss += total_loss.item() * batch_size
        running_fusion_loss += fusion_loss.item() * batch_size
        running_sil_loss += sil_loss.item() * batch_size
        running_ske_loss += ske_loss.item() * batch_size
        running_correct += (preds == y).sum().item()
        running_total += batch_size

    denom = max(running_total, 1)

    return {
        "total_loss": running_total_loss / denom,
        "fusion_loss": running_fusion_loss / denom,
        "silhouette_loss": running_sil_loss / denom,
        "skeleton_loss": running_ske_loss / denom,
        "acc": running_correct / denom
    }


# =========================================================
# 6. checkpoint 保存
# =========================================================
def save_checkpoint(
    save_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_val_acc: float,
    args,
    label_map: dict
) -> None:
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    ckpt = {
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "args": vars(args),
        "label_map": label_map,
    }

    torch.save(ckpt, save_path)


# =========================================================
# 7. 主训练函数
# =========================================================
def main(args):
    set_seed(args.seed)

    save_dir = Path(args.save_dir)
    ckpt_dir = save_dir / "checkpoints"
    log_dir = save_dir / "logs"

    ensure_dir(save_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("开始训练：GaitFusionModel")
    print("=" * 72)
    print(f"device: {device}")

    train_loader, val_loader, full_dataset = build_dataloaders(args)
    num_classes = len(full_dataset.label_map)

    print(f"silhouette_root: {args.silhouette_root}")
    print(f"skeleton_root:   {args.skeleton_root}")
    print(f"类别数 num_classes: {num_classes}")
    print(f"总样本数: {len(full_dataset)}")
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")

    save_json(full_dataset.label_map, save_dir / "label_map.json")
    save_json(vars(args), save_dir / "train_config.json")

    model = GaitFusionModel(
        num_classes=num_classes,
        silhouette_feature_dim=args.silhouette_feature_dim,
        skeleton_feature_dim=args.skeleton_feature_dim,
        fusion_dim=args.fusion_dim,
        silhouette_num_bins=args.silhouette_num_bins,
        dropout=args.dropout
    ).to(device)

    # -----------------------------------------------------
    # 先加载单分支预训练权重（如果提供了）
    # -----------------------------------------------------
    load_branch_checkpoint_if_needed(
        model=model,
        silhouette_ckpt_path=args.silhouette_ckpt,
        skeleton_ckpt_path=args.skeleton_ckpt,
        device=device
    )

    # -----------------------------------------------------
    # 如有需要，冻结两个 branch，只训练 fusion head
    # -----------------------------------------------------
    if args.freeze_branches:
        model.set_branch_trainable(
            silhouette_trainable=False,
            skeleton_trainable=False
        )
        print("[设置] 已冻结 silhouette_branch 和 skeleton_branch，仅训练融合头。")

    criterion = nn.CrossEntropyLoss()

    # 只优化 requires_grad=True 的参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma
    )

    history = {
        "train_total_loss": [],
        "train_fusion_loss": [],
        "train_silhouette_loss": [],
        "train_skeleton_loss": [],
        "train_acc": [],
        "val_total_loss": [],
        "val_fusion_loss": [],
        "val_silhouette_loss": [],
        "val_skeleton_loss": [],
        "val_acc": [],
        "lr": []
    }

    best_val_acc = -1.0
    best_epoch = -1
    train_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            silhouette_aux_weight=args.silhouette_aux_weight,
            skeleton_aux_weight=args.skeleton_aux_weight
        )

        if len(val_loader) > 0:
            val_metrics = validate_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                silhouette_aux_weight=args.silhouette_aux_weight,
                skeleton_aux_weight=args.skeleton_aux_weight
            )
        else:
            val_metrics = {
                "total_loss": float("nan"),
                "fusion_loss": float("nan"),
                "silhouette_loss": float("nan"),
                "skeleton_loss": float("nan"),
                "acc": float("nan")
            }

        scheduler.step()

        history["train_total_loss"].append(train_metrics["total_loss"])
        history["train_fusion_loss"].append(train_metrics["fusion_loss"])
        history["train_silhouette_loss"].append(train_metrics["silhouette_loss"])
        history["train_skeleton_loss"].append(train_metrics["skeleton_loss"])
        history["train_acc"].append(train_metrics["acc"])

        history["val_total_loss"].append(val_metrics["total_loss"])
        history["val_fusion_loss"].append(val_metrics["fusion_loss"])
        history["val_silhouette_loss"].append(val_metrics["silhouette_loss"])
        history["val_skeleton_loss"].append(val_metrics["skeleton_loss"])
        history["val_acc"].append(val_metrics["acc"])

        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start_time

        print(
            f"[Epoch {epoch:03d}/{args.epochs:03d}] "
            f"lr={current_lr:.6f} | "
            f"train_total={train_metrics['total_loss']:.4f}, train_fusion={train_metrics['fusion_loss']:.4f}, "
            f"train_acc={train_metrics['acc']:.4f} | "
            f"val_total={val_metrics['total_loss']:.4f}, val_fusion={val_metrics['fusion_loss']:.4f}, "
            f"val_acc={val_metrics['acc']:.4f} | "
            f"time={format_seconds(epoch_time)}"
        )

        save_checkpoint(
            save_path=ckpt_dir / "last.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_acc=best_val_acc,
            args=args,
            label_map=full_dataset.label_map
        )

        if not np.isnan(val_metrics["acc"]) and val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch

            save_checkpoint(
                save_path=ckpt_dir / "best.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_acc=best_val_acc,
                args=args,
                label_map=full_dataset.label_map
            )

            print(f"  -> 保存新的 best checkpoint: epoch={epoch}, val_acc={best_val_acc:.4f}")

        save_json(history, log_dir / "history.json")

    total_time = time.time() - train_start_time

    summary = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "total_train_time_sec": total_time,
        "total_train_time_readable": format_seconds(total_time),
        "num_classes": num_classes,
        "num_total_samples": len(full_dataset),
    }

    save_json(summary, save_dir / "train_summary.json")

    print("=" * 72)
    print("训练完成")
    print(f"best_epoch: {best_epoch}")
    print(f"best_val_acc: {best_val_acc:.4f}")
    print(f"total_time: {format_seconds(total_time)}")
    print(f"checkpoint(best): {ckpt_dir / 'best.pth'}")
    print(f"checkpoint(last): {ckpt_dir / 'last.pth'}")
    print("=" * 72)


# =========================================================
# 8. 参数解析
# =========================================================
def build_parser():
    parser = argparse.ArgumentParser(description="训练多特征融合步态识别模型 GaitFusionModel")

    # 数据相关
    parser.add_argument(
        "--silhouette_root",
        type=str,
        default="/home/zzzandan/desk/gait/gait/gait/data/silhouettes",
        help="剪影数据根目录"
    )
    parser.add_argument(
        "--skeleton_root",
        type=str,
        default="/home/zzzandan/desk/gait/gait/gait/data/skeletons",
        help="骨架数据根目录"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/zzzandan/desk/gait/gait/gait/outputs/fusion_exp1",
        help="实验输出目录"
    )
    parser.add_argument("--seq_len", type=int, default=30, help="每个序列采样多少帧")
    parser.add_argument("--img_h", type=int, default=64, help="输入图像高度")
    parser.add_argument("--img_w", type=int, default=44, help="输入图像宽度")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例")

    # DataLoader
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers")

    # 模型结构
    parser.add_argument("--silhouette_feature_dim", type=int, default=256, help="剪影分支 embedding 维度")
    parser.add_argument("--skeleton_feature_dim", type=int, default=256, help="骨架分支 embedding 维度")
    parser.add_argument("--fusion_dim", type=int, default=256, help="融合 embedding 维度")
    parser.add_argument("--silhouette_num_bins", type=int, default=4, help="剪影分支水平分块数")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout 比例")

    # 预训练和冻结
    parser.add_argument(
        "--silhouette_ckpt",
        type=str,
        default="/home/zzzandan/desk/gait/gait/gait/outputs/silhouette_exp1/checkpoints/best.pth",
        help="剪影分支预训练 checkpoint 路径，可为空"
    )
    parser.add_argument(
        "--skeleton_ckpt",
        type=str,
        default="/home/zzzandan/desk/gait/gait/gait/outputs/skeleton_exp1/checkpoints/best.pth",
        help="骨架分支预训练 checkpoint 路径，可为空"
    )
    parser.add_argument(
        "--freeze_branches",
        action="store_true",
        help="是否冻结两个单分支，仅训练融合头"
    )

    # 辅助损失
    parser.add_argument("--silhouette_aux_weight", type=float, default=0.2, help="剪影辅助损失权重")
    parser.add_argument("--skeleton_aux_weight", type=float, default=0.2, help="骨架辅助损失权重")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="初始学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--lr_step_size", type=int, default=10, help="StepLR step_size")
    parser.add_argument("--lr_gamma", type=float, default=0.5, help="StepLR gamma")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    return parser


# =========================================================
# 9. 程序入口
# =========================================================
if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
