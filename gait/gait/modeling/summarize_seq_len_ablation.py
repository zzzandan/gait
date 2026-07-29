
# -*- coding: utf-8 -*-
"""
汇总不同序列长度 T 下的实验结果。
运行前提：各实验目录下已经生成 best_metrics.json
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import argparse

def read_best_acc(exp_dir: Path):
    path = exp_dir / "best_metrics.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("best_val_acc", None)

def main(args):
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [["T", "剪影分支Acc", "骨架分支Acc", "融合模型Acc"]]
    for t in args.seq_lens:
        sil = read_best_acc(Path(args.silhouette_root) / f"T{t}")
        ske = read_best_acc(Path(args.skeleton_root) / f"T{t}")
        fus = read_best_acc(Path(args.fusion_root) / f"T{t}")
        rows.append([
            str(t),
            f"{sil*100:.2f}%" if sil is not None else "",
            f"{ske*100:.2f}%" if ske is not None else "",
            f"{fus*100:.2f}%" if fus is not None else "",
        ])

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--silhouette_root", type=str, required=True)
    parser.add_argument("--skeleton_root", type=str, required=True)
    parser.add_argument("--fusion_root", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="outputs/seq_len_ablation.csv")
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[10,20,30])
    args = parser.parse_args()
    main(args)
