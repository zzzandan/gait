# -*- coding: utf-8 -*-
"""
train_fusion_condition_split.py

作用：
使用“按条件划分”的方式训练多特征融合步态识别模型 GaitFusionModel，
而不是简单的随机分层划分。

为什么要这样做：
- 你之前已经把剪影单分支和骨架单分支都切换到“按条件划分”评估
- 如果融合部分还继续使用随机划分，那么三组实验口径就不一致
- 因此，融合训练也必须使用同样的条件划分规则，才能公平比较：
    训练条件：nm-01, nm-02, nm-03, nm-04
    验证条件：nm-05, nm-06, bg-01, bg-02, cl-01, cl-02

当前脚本特点：
1. 同时读取 silhouettes 和 skeletons
2. 先做 person_id / seq_name 严格配对
3. 再按 seq_name 里的条件字段进行 train / val 划分
4. 支持加载“按条件划分训练得到的单分支 best.pth”
5. 支持冻结两个 branch，只训练 fusion head
6. 自动导出 train / val 的配对序列名单，便于核查划分是否合理
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
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(data: dict, save_path: str | Path) -> None:
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def parse_condition_from_seq_name(seq_name: str) -> str:
    """
    从 seq_name 中提取条件字段。

    例如：
        nm-01_090 -> nm-01
        bg-02_018 -> bg-02
        cl-01_144 -> cl-01
    """
    if "_" not in seq_name:
        raise ValueError(f"seq_name 不符合预期格式，无法解析条件字段：{seq_name}")
    return seq_name.split("_", 1)[0]


def parse_csv_list(s: str) -> List[str]:
    items = [x.strip() for x in s.split(",")]
    return [x for x in items if x]


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
    - 只保留两个模态都存在的 person_id
    - 每个 person_id 下，只保留两个模态都存在的 seq_name
    - 每个样本 = 一对配好的 silhouette_seq + skeleton_seq
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

        common_person_ids = sorted(silhouette_person_ids & skeleton_person_ids)
        if len(common_person_ids) == 0:
            raise ValueError("未找到共同的 person_id，请检查两个根目录。")

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

                if len(sil_frame_paths) == 0 or len(ske_frame_paths) == 0:
                    continue

                self.samples.append({
                    "person_id": pid,
                    "label": self.label_map[pid],
                    "seq_name": seq_name,
                    "sil_seq_dir": str(sil_seq_dir),
                    "ske_seq_dir": str(ske_seq_dir),
                    "sil_frame_paths": sil_frame_paths,
                    "ske_frame_paths": ske_frame_paths
                })

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_indices(self, common_num_frames: int) -> np.ndarray:
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
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取剪影图像: {img_path}")

        img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # [1, H, W]
        return img

    def _load_one_skeleton_frame(self, npy_path: str) -> np.ndarray:
        arr = np.load(npy_path)

        if arr.ndim != 3:
            raise ValueError(
                f"Skeleton Map 文件维度错误，期望 [C,H,W]，实际为 {arr.shape}，文件: {npy_path}"
            )

        c, h, w = arr.shape
        if c != 2:
            raise ValueError(f"Skeleton Map 通道数应为 2，实际为 {c}，文件: {npy_path}")

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
        sample = self.samples[idx]

        sil_frame_paths = sample["sil_frame_paths"]
        ske_frame_paths = sample["ske_frame_paths"]
        label = sample["label"]

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
# 3. 条件划分相关函数
# =========================================================
def build_condition_split_indices(
    dataset: PairedFusionDataset,
    train_conditions: List[str],
    val_conditions: List[str],
    require_each_person_has_train_and_val: bool = True
) -> Tuple[List[int], List[int], dict]:
    """
    按条件划分 train / val。
    """
    train_set = set(train_conditions)
    val_set = set(val_conditions)

    overlap = train_set & val_set
    if overlap:
        raise ValueError(f"训练条件和验证条件存在重叠：{sorted(overlap)}")

    train_indices = []
    val_indices = []
    person_stats = defaultdict(lambda: {"train": 0, "val": 0})
    ignored_conditions = defaultdict(int)

    for idx, sample in enumerate(dataset.samples):
        seq_name = sample["seq_name"]
        condition = parse_condition_from_seq_name(seq_name)
        pid = str(sample["person_id"])

        if condition in train_set:
            train_indices.append(idx)
            person_stats[pid]["train"] += 1
        elif condition in val_set:
            val_indices.append(idx)
            person_stats[pid]["val"] += 1
        else:
            ignored_conditions[condition] += 1

    if not train_indices:
        raise ValueError("训练集为空，请检查 train_conditions 是否与你的数据命名一致。")
    if not val_indices:
        raise ValueError("验证集为空，请检查 val_conditions 是否与你的数据命名一致。")

    if require_each_person_has_train_and_val:
        bad_persons = []
        for pid in sorted(person_stats.keys()):
            if person_stats[pid]["train"] == 0 or person_stats[pid]["val"] == 0:
                bad_persons.append({
                    "person_id": pid,
                    "train_count": person_stats[pid]["train"],
                    "val_count": person_stats[pid]["val"]
                })
        if bad_persons:
            raise ValueError(
                "存在某些 person_id 在 train 或 val 中没有样本，无法形成合理评估：\n"
                + json.dumps(bad_persons, ensure_ascii=False, indent=2)
            )

    split_info = {
        "train_conditions": sorted(train_set),
        "val_conditions": sorted(val_set),
        "ignored_conditions": dict(sorted(ignored_conditions.items())),
        "person_stats": dict(sorted(person_stats.items()))
    }
    return train_indices, val_indices, split_info


def export_split_records(
    dataset: PairedFusionDataset,
    train_indices: List[int],
    val_indices: List[int],
    save_dir: str | Path
) -> None:
    save_dir = Path(save_dir)
    ensure_dir(save_dir)

    def sample_to_record(sample):
        return {
            "person_id": sample["person_id"],
            "seq_name": sample["seq_name"],
            "condition": parse_condition_from_seq_name(sample["seq_name"]),
            "sil_seq_dir": sample["sil_seq_dir"],
            "ske_seq_dir": sample["ske_seq_dir"],
            "num_sil_frames": len(sample["sil_frame_paths"]),
            "num_ske_frames": len(sample["ske_frame_paths"]),
        }

    train_records = [sample_to_record(dataset.samples[i]) for i in train_indices]
    val_records = [sample_to_record(dataset.samples[i]) for i in val_indices]

    with open(save_dir / "train_split_records.json", "w", encoding="utf-8") as f:
        json.dump(train_records, f, indent=2, ensure_ascii=False)
    with open(save_dir / "val_split_records.json", "w", encoding="utf-8") as f:
        json.dump(val_records, f, indent=2, ensure_ascii=False)

    def write_grouped_txt(path: Path, records: List[dict]):
        grouped = defaultdict(list)
        for r in records:
            grouped[r["person_id"]].append(r["seq_name"])
        with open(path, "w", encoding="utf-8") as f:
            for pid in sorted(grouped.keys()):
                f.write(f"[person_id={pid}]\n")
                for seq_name in sorted(grouped[pid]):
                    f.write(f"  {seq_name}\n")
                f.write("\n")

    write_grouped_txt(save_dir / "train_split_grouped.txt", train_records)
    write_grouped_txt(save_dir / "val_split_grouped.txt", val_records)


def build_dataloaders(args) -> Tuple[DataLoader, DataLoader, PairedFusionDataset, dict, List[int], List[int]]:
    """
    创建按条件划分后的 train / val DataLoader。
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

    train_conditions = parse_csv_list(args.train_conditions)
    val_conditions = parse_csv_list(args.val_conditions)

    train_indices, val_indices, split_info = build_condition_split_indices(
        dataset=train_dataset_full,
        train_conditions=train_conditions,
        val_conditions=val_conditions,
        require_each_person_has_train_and_val=(not args.allow_incomplete_person_split)
    )

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

    return train_loader, val_loader, train_dataset_full, split_info, train_indices, val_indices


# =========================================================
# 4. 权重加载
# =========================================================
def load_branch_checkpoint_if_needed(
    model: GaitFusionModel,
    silhouette_ckpt_path: str | None = None,
    skeleton_ckpt_path: str | None = None,
    device: torch.device | str = "cpu"
) -> None:
    """
    如果提供了单分支 checkpoint，就加载进去。
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
    split_dir = save_dir / "split_info"

    ensure_dir(save_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(log_dir)
    ensure_dir(split_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("开始训练：GaitFusionModel（按条件划分）")
    print("=" * 72)
    print(f"device: {device}")

    train_loader, val_loader, full_dataset, split_info, train_indices, val_indices = build_dataloaders(args)
    num_classes = len(full_dataset.label_map)

    print(f"silhouette_root: {args.silhouette_root}")
    print(f"skeleton_root:   {args.skeleton_root}")
    print(f"类别数 num_classes: {num_classes}")
    print(f"总样本数: {len(full_dataset)}")
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    print(f"训练条件: {split_info['train_conditions']}")
    print(f"验证条件: {split_info['val_conditions']}")

    save_json(full_dataset.label_map, save_dir / "label_map.json")
    save_json(vars(args), save_dir / "train_config.json")
    save_json(split_info, save_dir / "condition_split_summary.json")

    export_split_records(
        dataset=full_dataset,
        train_indices=train_indices,
        val_indices=val_indices,
        save_dir=split_dir
    )

    model = GaitFusionModel(
        num_classes=num_classes,
        silhouette_feature_dim=args.silhouette_feature_dim,
        skeleton_feature_dim=args.skeleton_feature_dim,
        fusion_dim=args.fusion_dim,
        silhouette_num_bins=args.silhouette_num_bins,
        dropout=args.dropout
    ).to(device)

    load_branch_checkpoint_if_needed(
        model=model,
        silhouette_ckpt_path=args.silhouette_ckpt,
        skeleton_ckpt_path=args.skeleton_ckpt,
        device=device
    )

    if args.freeze_branches:
        model.set_branch_trainable(
            silhouette_trainable=False,
            skeleton_trainable=False
        )
        print("[设置] 已冻结 silhouette_branch 和 skeleton_branch，仅训练融合头。")

    criterion = nn.CrossEntropyLoss()

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

        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            silhouette_aux_weight=args.silhouette_aux_weight,
            skeleton_aux_weight=args.skeleton_aux_weight
        )

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

        if val_metrics["acc"] > best_val_acc:
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
        "train_conditions": split_info["train_conditions"],
        "val_conditions": split_info["val_conditions"]
    }
    save_json(summary, save_dir / "train_summary.json")

    print("=" * 72)
    print("训练完成")
    print(f"best_epoch: {best_epoch}")
    print(f"best_val_acc: {best_val_acc:.4f}")
    print(f"total_time: {format_seconds(total_time)}")
    print(f"checkpoint(best): {ckpt_dir / 'best.pth'}")
    print(f"checkpoint(last): {ckpt_dir / 'last.pth'}")
    print(f"划分名单目录: {split_dir}")
    print("=" * 72)


# =========================================================
# 8. 参数解析
# =========================================================
def build_parser():
    parser = argparse.ArgumentParser(description="训练多特征融合模型 GaitFusionModel（按条件划分 train / val）")

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
        default="/home/zzzandan/desk/gait/gait/gait/outputs/fusion_condition_split_exp1",
        help="实验输出目录"
    )

    parser.add_argument(
        "--train_conditions",
        type=str,
        default="nm-01,nm-02,nm-03,nm-04",
        help="训练条件，逗号分隔"
    )
    parser.add_argument(
        "--val_conditions",
        type=str,
        default="nm-05,nm-06,bg-01,bg-02,cl-01,cl-02",
        help="验证条件，逗号分隔"
    )
    parser.add_argument(
        "--allow_incomplete_person_split",
        action="store_true",
        help="允许某些 person_id 只出现在 train 或 val 中。默认不允许。"
    )

    parser.add_argument("--seq_len", type=int, default=30, help="每个序列采样多少帧")
    parser.add_argument("--img_h", type=int, default=64, help="输入图像高度")
    parser.add_argument("--img_w", type=int, default=44, help="输入图像宽度")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers")

    parser.add_argument("--silhouette_feature_dim", type=int, default=256, help="剪影分支 embedding 维度")
    parser.add_argument("--skeleton_feature_dim", type=int, default=256, help="骨架分支 embedding 维度")
    parser.add_argument("--fusion_dim", type=int, default=256, help="融合 embedding 维度")
    parser.add_argument("--silhouette_num_bins", type=int, default=4, help="剪影分支水平分块数")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout 比例")

    parser.add_argument(
        "--silhouette_ckpt",
        type=str,
        default="/home/zzzandan/desk/gait/gait/gait/outputs/silhouette_condition_split_exp1/checkpoints/best.pth",
        help="剪影分支预训练 checkpoint 路径"
    )
    parser.add_argument(
        "--skeleton_ckpt",
        type=str,
        default="/home/zzzandan/desk/gait/gait/gait/outputs/skeleton_condition_split_exp1/checkpoints/best.pth",
        help="骨架分支预训练 checkpoint 路径"
    )
    parser.add_argument(
        "--freeze_branches",
        action="store_true",
        help="是否冻结两个单分支，仅训练融合头"
    )

    parser.add_argument("--silhouette_aux_weight", type=float, default=0.2, help="剪影辅助损失权重")
    parser.add_argument("--skeleton_aux_weight", type=float, default=0.2, help="骨架辅助损失权重")

    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="初始学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--lr_step_size", type=int, default=10, help="StepLR step_size")
    parser.add_argument("--lr_gamma", type=float, default=0.5, help="StepLR gamma")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
