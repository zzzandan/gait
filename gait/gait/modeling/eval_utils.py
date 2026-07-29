
# -*- coding: utf-8 -*-
"""
论文结果导出工具：
1. 保存逐样本预测明细 predictions.csv
2. 按具体条件 / 条件大类（NM/BG/CL）统计准确率
3. 生成论文可直接使用的小表 condition_accuracy_table.csv
4. 生成混淆矩阵图 confusion_matrix.png
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def extract_condition_group(condition: str) -> str:
    cond = str(condition).lower()
    if cond.startswith("nm"):
        return "NM"
    if cond.startswith("bg"):
        return "BG"
    if cond.startswith("cl"):
        return "CL"
    return "OTHER"


def save_predictions_csv(records: List[dict], save_path: str | Path) -> None:
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fieldnames = [
        "person_id", "condition", "condition_group", "seq_name",
        "gt_label", "pred_label", "correct"
    ]
    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def compute_condition_accuracy(records: List[dict]):
    cond_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    group_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    total = 0
    total_correct = 0

    for r in records:
        cond = r["condition"]
        group = r["condition_group"]
        correct = int(r["correct"])

        cond_stats[cond]["total"] += 1
        cond_stats[cond]["correct"] += correct

        group_stats[group]["total"] += 1
        group_stats[group]["correct"] += correct

        total += 1
        total_correct += correct

    cond_rows = []
    for cond in sorted(cond_stats.keys()):
        s = cond_stats[cond]
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        cond_rows.append({
            "condition": cond,
            "total": s["total"],
            "correct": s["correct"],
            "acc": acc,
        })

    group_rows = []
    for group in ["NM", "BG", "CL", "OTHER"]:
        if group not in group_stats:
            continue
        s = group_stats[group]
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        group_rows.append({
            "condition_group": group,
            "total": s["total"],
            "correct": s["correct"],
            "acc": acc,
        })

    overall_acc = total_correct / total if total > 0 else 0.0
    return cond_rows, group_rows, overall_acc


def save_condition_accuracy_csv(cond_rows, group_rows, overall_acc, save_dir: str | Path) -> None:
    save_dir = Path(save_dir)
    ensure_dir(save_dir)

    cond_csv = save_dir / "condition_accuracy.csv"
    with open(cond_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "total", "correct", "acc"])
        writer.writeheader()
        for row in cond_rows:
            writer.writerow(row)

    group_csv = save_dir / "condition_group_accuracy.csv"
    with open(group_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["condition_group", "total", "correct", "acc"])
        writer.writeheader()
        for row in group_rows:
            writer.writerow(row)

    thesis_csv = save_dir / "condition_accuracy_table.csv"
    with open(thesis_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["NM正常步态", "BG背包", "CL大衣", "总体Acc"])
        nm = next((x["acc"] for x in group_rows if x["condition_group"] == "NM"), 0.0)
        bg = next((x["acc"] for x in group_rows if x["condition_group"] == "BG"), 0.0)
        cl = next((x["acc"] for x in group_rows if x["condition_group"] == "CL"), 0.0)
        writer.writerow([
            f"{nm * 100:.2f}%",
            f"{bg * 100:.2f}%",
            f"{cl * 100:.2f}%",
            f"{overall_acc * 100:.2f}%"
        ])


def compute_confusion_matrix(records: List[dict], num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for r in records:
        gt = int(r["gt_label"])
        pred = int(r["pred_label"])
        cm[gt, pred] += 1
    return cm


def save_confusion_matrix_figure(cm, save_path: str | Path, normalize: bool = True) -> None:
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        cm_show = cm / row_sum
    else:
        cm_show = cm

    plt.figure(figsize=(8, 7))
    plt.imshow(cm_show, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()

    ticks = np.arange(cm.shape[0])
    plt.xticks(ticks)
    plt.yticks(ticks)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm_show[i, j]:.2f}" if normalize else str(cm_show[i, j])
            plt.text(j, i, text, ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def export_thesis_reports(records: List[dict], num_classes: int, report_dir: str | Path) -> None:
    report_dir = Path(report_dir)
    ensure_dir(report_dir)

    save_predictions_csv(records, report_dir / "predictions.csv")

    cond_rows, group_rows, overall_acc = compute_condition_accuracy(records)
    save_condition_accuracy_csv(cond_rows, group_rows, overall_acc, report_dir)

    cm = compute_confusion_matrix(records, num_classes)
    save_confusion_matrix_figure(cm, report_dir / "confusion_matrix.png", normalize=True)
