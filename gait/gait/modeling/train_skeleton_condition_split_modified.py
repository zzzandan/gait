# -*- coding: utf-8 -*-
"""
train_skeleton_condition_split.py

作用：
使用“按条件划分”的方式训练骨架单分支模型，而不是简单的随机分层划分。

为什么要这样做：
- 你之前的随机划分方式，会让同一个 person_id 在训练集和验证集里同时出现大量“相近条件”的样本
- 这种设置下，验证集往往偏容易，可能导致 val_acc 偏高
- 对 CASIA-B 这类数据，更合理的做法是按条件划分，例如：
    训练条件：nm-01, nm-02, nm-03, nm-04
    验证条件：nm-05, nm-06, bg-01, bg-02, cl-01, cl-02

当前脚本特点：
1. 复用你现有的 skeleton_dataset.py 与 skeleton_branch.py
2. 不再使用随机划分 train / val
3. 改为根据 seq_name 中的“条件字段”进行划分
4. 自动导出 train / val 的序列名单，方便继续核查划分是否合理

目录命名约定：
- seq_name 应形如：
    nm-01_090
    nm-06_054
    bg-01_018
    cl-02_144
- 其中 "_" 前面的部分会被视为条件名：
    nm-01
    nm-06
    bg-01
    cl-02
"""

from __future__ import annotations

import json
import time
import random
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from eval_utils import export_thesis_reports, extract_condition_group

from skeleton_dataset import SkeletonSequenceDataset
from skeleton_branch import SkeletonBranch


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
    从 seq_name 中提取“条件字段”。

    例如：
        nm-01_090 -> nm-01
        bg-02_018 -> bg-02
        cl-01_144 -> cl-01
    """
    if "_" not in seq_name:
        raise ValueError(f"seq_name 不符合预期格式，无法解析条件字段：{seq_name}")
    return seq_name.split("_", 1)[0]


def parse_csv_list(s: str) -> List[str]:
    """
    将命令行传入的逗号分隔字符串解析为列表。
    """
    items = [x.strip() for x in s.split(",")]
    return [x for x in items if x]


# =========================================================
# 2. 条件划分相关函数
# =========================================================
def build_condition_split_indices(
    dataset: SkeletonSequenceDataset,
    train_conditions: List[str],
    val_conditions: List[str],
    require_each_person_has_train_and_val: bool = True
) -> Tuple[List[int], List[int], dict]:
    """
    按条件划分 train / val。

    规则：
    - seq_name 形如 nm-01_090
    - 条件名取 "_" 前面的部分，如 nm-01
    - 如果该条件在 train_conditions 中，则该样本进入 train
    - 如果该条件在 val_conditions 中，则该样本进入 val

    注意：
    - train_conditions 与 val_conditions 不能有重叠
    - 若某个条件两边都不在，则该样本被忽略
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
        seq_name = Path(sample["seq_dir"]).name
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
    dataset: SkeletonSequenceDataset,
    train_indices: List[int],
    val_indices: List[int],
    save_dir: str | Path
) -> None:
    """
    导出 train / val 的样本名单，便于核查划分是否合理。
    """
    save_dir = Path(save_dir)
    ensure_dir(save_dir)

    def sample_to_record(sample):
        seq_name = Path(sample["seq_dir"]).name
        return {
            "person_id": sample["person_id"],
            "seq_name": seq_name,
            "condition": parse_condition_from_seq_name(seq_name),
            "seq_dir": sample["seq_dir"],
            "num_frames": len(sample["frame_paths"]),
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


def build_dataloaders(args) -> Tuple[DataLoader, DataLoader, SkeletonSequenceDataset, dict, List[int], List[int]]:
    """
    创建按条件划分后的 train / val DataLoader。
    """
    # 训练集使用 train=True，对序列做随机采样
    train_dataset_full = SkeletonSequenceDataset(
        root_dir=args.data_root,
        seq_len=args.seq_len,
        img_h=args.img_h,
        img_w=args.img_w,
        train=True
    )

    # 验证集使用 train=False，对序列做均匀采样
    val_dataset_full = SkeletonSequenceDataset(
        root_dir=args.data_root,
        seq_len=args.seq_len,
        img_h=args.img_h,
        img_w=args.img_w,
        train=False
    )

    if len(train_dataset_full) == 0:
        raise ValueError(f"数据集为空，请检查目录：{args.data_root}")

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
# 3. 单轮训练 / 验证
# =========================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for batch in loader:
        if len(batch) == 3:
            batch_x, batch_y, _ = batch
        else:
            batch_x, batch_y = batch
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        out = model(batch_x)
        logits = out["logits"]
        loss = criterion(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = batch_y.size(0)
        preds = torch.argmax(logits, dim=1)

        running_loss += loss.item() * batch_size
        running_correct += (preds == batch_y).sum().item()
        running_total += batch_size

    return {
        "loss": running_loss / max(running_total, 1),
        "acc": running_correct / max(running_total, 1)
    }


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for batch in loader:
        if len(batch) == 3:
            batch_x, batch_y, _ = batch
        else:
            batch_x, batch_y = batch
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        out = model(batch_x)
        logits = out["logits"]
        loss = criterion(logits, batch_y)

        batch_size = batch_y.size(0)
        preds = torch.argmax(logits, dim=1)

        running_loss += loss.item() * batch_size
        running_correct += (preds == batch_y).sum().item()
        running_total += batch_size

    return {
        "loss": running_loss / max(running_total, 1),
        "acc": running_correct / max(running_total, 1)
    }


# =========================================================
# 4. checkpoint 保存
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
# 5. 主训练函数
# =========================================================
def main(args):
    set_seed(args.seed)

    save_dir = Path(args.save_dir)
    ckpt_dir = save_dir / "checkpoints"
    log_dir = save_dir / "logs"
    split_dir = save_dir / "split_info"
    report_dir = save_dir / "thesis_reports"

    ensure_dir(save_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(log_dir)
    ensure_dir(split_dir)
    ensure_dir(report_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("开始训练：SkeletonBranch（按条件划分）")
    print("=" * 70)
    print(f"device: {device}")

    train_loader, val_loader, full_dataset, split_info, train_indices, val_indices = build_dataloaders(args)
    num_classes = len(full_dataset.label_map)

    print(f"数据根目录: {args.data_root}")
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

    model = SkeletonBranch(
        num_classes=num_classes,
        in_channels=2,
        feature_dim=args.feature_dim,
        dropout=args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    best_val_acc = -1.0
    best_epoch = -1
    train_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate_one_epoch(model, val_loader, criterion, device)

        scheduler.step()

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start_time

        print(
            f"[Epoch {epoch:03d}/{args.epochs:03d}] "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_metrics['loss']:.4f}, train_acc={train_metrics['acc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['acc']:.4f} | "
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
        "val_conditions": split_info["val_conditions"],
    }
    save_json(summary, save_dir / "train_summary.json")
    save_json({"best_epoch": best_epoch, "best_val_acc": best_val_acc}, save_dir / "best_metrics.json")

    # 使用 best checkpoint 在验证集上重新跑一遍，导出论文需要的统计文件
    best_ckpt = torch.load(ckpt_dir / "best.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"], strict=True)
    final_val_metrics = validate_one_epoch(model, val_loader, criterion, device)
    export_thesis_reports(
        records=final_val_metrics["records"],
        num_classes=num_classes,
        report_dir=report_dir
    )

    print("=" * 70)
    print("训练完成")
    print(f"best_epoch: {best_epoch}")
    print(f"best_val_acc: {best_val_acc:.4f}")
    print(f"total_time: {format_seconds(total_time)}")
    print(f"checkpoint(best): {ckpt_dir / 'best.pth'}")
    print(f"checkpoint(last): {ckpt_dir / 'last.pth'}")
    print(f"划分名单目录: {split_dir}")
    print("=" * 70)


# =========================================================
# 6. 参数解析
# =========================================================
def build_parser():
    parser = argparse.ArgumentParser(description="训练骨架单分支模型 SkeletonBranch（按条件划分 train / val）")

    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/zzzandan/desk/gait/gait/gait/data/skeletons",
        help="骨架数据根目录，目录结构应为 person_id/seq_name/frame.npy"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/zzzandan/desk/gait/gait/gait/outputs/skeleton_condition_split_exp1",
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
    parser.add_argument("--img_h", type=int, default=64, help="输入 Skeleton Map 高度")
    parser.add_argument("--img_w", type=int, default=44, help="输入 Skeleton Map 宽度")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers")
    parser.add_argument("--feature_dim", type=int, default=256, help="最终 embedding 维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="全连接层后的 dropout 比例")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="初始学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减系数")
    parser.add_argument("--lr_step_size", type=int, default=10, help="StepLR 的 step_size")
    parser.add_argument("--lr_gamma", type=float, default=0.5, help="StepLR 的 gamma")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
