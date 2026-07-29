
# -*- coding: utf-8 -*-
"""
汇总融合方式对比表。
把不同实验目录中的 best_metrics.json 读出来，生成论文可直接使用的 CSV。
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import argparse

def read_best_acc(path_str: str):
    path = Path(path_str) / "best_metrics.json"
    if not path.exists():
        return ""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return f"{data.get('best_val_acc', 0.0)*100:.2f}%"

def main(args):
    rows = [
        ["方法", "特征使用方式", "Acc"],
        ["剪影单分支", "仅剪影", read_best_acc(args.silhouette_exp)],
        ["骨架单分支", "仅骨架", read_best_acc(args.skeleton_exp)],
        ["基础拼接融合", "剪影+骨架", read_best_acc(args.fusion_exp)],
        ["冻结双分支只训练融合头", "剪影+骨架", read_best_acc(args.freeze_fusion_exp) if args.freeze_fusion_exp else ""],
    ]
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--silhouette_exp", type=str, required=True)
    parser.add_argument("--skeleton_exp", type=str, required=True)
    parser.add_argument("--fusion_exp", type=str, required=True)
    parser.add_argument("--freeze_fusion_exp", type=str, default="")
    parser.add_argument("--output_csv", type=str, default="outputs/fusion_compare.csv")
    args = parser.parse_args()
    main(args)
