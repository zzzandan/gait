# -*- coding: utf-8 -*-
"""
train_skeleton.py
骨架单分支训练脚本
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

from skeleton_dataset import SkeletonSequenceDataset
from skeleton_branch import SkeletonBranch


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


def build_stratified_split_indices(
    dataset: SkeletonSequenceDataset,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
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


def build_dataloaders(args) -> Tuple[DataLoader, DataLoader, SkeletonSequenceDataset]:
    train_dataset_full = SkeletonSequenceDataset(
        root_dir=args.data_root,
        seq_len=args.seq_len,
        img_h=args.img_h,
        img_w=args.img_w,
        train=True
    )

    val_dataset_full = SkeletonSequenceDataset(
        root_dir=args.data_root,
        seq_len=args.seq_len,
        img_h=args.img_h,
        img_w=args.img_w,
        train=False
    )

    if len(train_dataset_full) == 0:
        raise ValueError(f"数据集为空，请检查目录：{args.data_root}")

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

    for batch_x, batch_y in loader:
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

    epoch_loss = running_loss / max(running_total, 1)
    epoch_acc = running_correct / max(running_total, 1)

    return {"loss": epoch_loss, "acc": epoch_acc}


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

    for batch_x, batch_y in loader:
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

    epoch_loss = running_loss / max(running_total, 1)
    epoch_acc = running_correct / max(running_total, 1)

    return {"loss": epoch_loss, "acc": epoch_acc}


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


def main(args):
    set_seed(args.seed)

    save_dir = Path(args.save_dir)
    ckpt_dir = save_dir / "checkpoints"
    log_dir = save_dir / "logs"

    ensure_dir(save_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("开始训练：SkeletonBranch")
    print("=" * 70)
    print(f"device: {device}")

    train_loader, val_loader, full_dataset = build_dataloaders(args)
    num_classes = len(full_dataset.label_map)

    print(f"数据根目录: {args.data_root}")
    print(f"类别数 num_classes: {num_classes}")
    print(f"总样本数: {len(full_dataset)}")
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")

    save_json(full_dataset.label_map, save_dir / "label_map.json")
    save_json(vars(args), save_dir / "train_config.json")

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

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        if len(val_loader) > 0:
            val_metrics = validate_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device
            )
        else:
            val_metrics = {"loss": float("nan"), "acc": float("nan")}

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

    print("=" * 70)
    print("训练完成")
    print(f"best_epoch: {best_epoch}")
    print(f"best_val_acc: {best_val_acc:.4f}")
    print(f"total_time: {format_seconds(total_time)}")
    print(f"checkpoint(best): {ckpt_dir / 'best.pth'}")
    print(f"checkpoint(last): {ckpt_dir / 'last.pth'}")
    print("=" * 70)


def build_parser():
    parser = argparse.ArgumentParser(description="训练骨架单分支模型 SkeletonBranch")

    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/zzzandan/desk/gait/gait/gait/data/skeletons",
        help="骨架数据根目录，目录结构应为 person_id/seq_name/frame.npy"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/zzzandan/desk/gait/gait/gait/outputs/skeleton_exp1",
        help="实验输出目录，用于保存 checkpoint、日志、配置等"
    )
    parser.add_argument("--seq_len", type=int, default=30, help="每个序列采样多少帧")
    parser.add_argument("--img_h", type=int, default=64, help="输入 Skeleton Map 高度")
    parser.add_argument("--img_w", type=int, default=44, help="输入 Skeleton Map 宽度")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例，按类别分层切分")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader 的 num_workers")
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
