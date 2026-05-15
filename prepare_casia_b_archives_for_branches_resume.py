# -*- coding: utf-8 -*-
"""
prepare_casia_b_archives_for_branches_resume.py

支持：
1. 直接读取 CASIA-B silhouette 的 .tar.gz 压缩包
2. 也支持已解压目录
3. 整理成当前 silhouette / skeleton 分支可直接读取的格式
4. 支持“断点继续”：
   - 如果某一帧对应的输出已经存在，就自动跳过
   - 中途中断后，重新运行会从缺失帧继续
"""

from __future__ import annotations

import json
import tarfile
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: dict, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def is_image_suffix(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTS


def parse_casia_rel_path(rel_path: str):
    parts = Path(rel_path).parts
    if len(parts) < 4:
        return None

    pid = parts[0]
    cond = parts[1]
    view = parts[2]
    fname = Path(parts[-1]).name

    if not (len(pid) == 3 and pid.isdigit()):
        return None
    if not (cond.startswith("nm-") or cond.startswith("bg-") or cond.startswith("cl-")):
        return None
    if not (len(view) == 3 and view.isdigit()):
        return None

    stem_parts = Path(fname).stem.split("-")
    if len(stem_parts) >= 5:
        try:
            frame_idx = int(stem_parts[-1])
        except ValueError:
            frame_idx = 0
    else:
        frame_idx = 0

    seq_name = f"{cond}_{view}"
    return pid, seq_name, frame_idx


def scan_from_archives(input_root: Path):
    groups = defaultdict(list)
    skipped = []

    archives = sorted(input_root.glob("*.tar.gz"))
    for archive_path in archives:
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    if not is_image_suffix(member.name):
                        continue

                    info = parse_casia_rel_path(member.name)
                    if info is None:
                        skipped.append(f"{archive_path.name}:{member.name}")
                        continue

                    pid, seq_name, frame_idx = info
                    groups[(pid, seq_name)].append((frame_idx, archive_path, member.name))
        except Exception as e:
            skipped.append(f"{archive_path.name}:<open_failed:{e}>")

    for key in groups:
        groups[key] = sorted(groups[key], key=lambda x: (x[0], x[2]))

    return groups, skipped


def scan_from_extracted_dirs(input_root: Path):
    groups = defaultdict(list)
    skipped = []

    for img_path in input_root.rglob("*"):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        try:
            rel_path = str(img_path.relative_to(input_root))
        except Exception:
            rel_path = str(img_path)

        info = parse_casia_rel_path(rel_path)
        if info is None:
            skipped.append(str(img_path))
            continue

        pid, seq_name, frame_idx = info
        groups[(pid, seq_name)].append((frame_idx, img_path))

    for key in groups:
        groups[key] = sorted(groups[key], key=lambda x: (x[0], str(x[1])))

    return groups, skipped


def read_image_from_archive(archive_path: Path, member_name: str) -> np.ndarray:
    with tarfile.open(archive_path, "r:gz") as tar:
        extracted = tar.extractfile(member_name)
        if extracted is None:
            raise ValueError(f"无法从压缩包读取成员文件: {archive_path} -> {member_name}")
        data = extracted.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法解码图像: {archive_path} -> {member_name}")
        return img


def read_image_from_path(img_path: Path) -> np.ndarray:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    return img


def find_foreground_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1
    return x1, y1, x2, y2


def normalize_one_silhouette(mask: np.ndarray, out_h: int = 64, out_w: int = 44, pad: int = 2) -> np.ndarray:
    bbox = find_foreground_bbox(mask)
    canvas = np.zeros((out_h, out_w), dtype=np.uint8)

    if bbox is None:
        return canvas

    x1, y1, x2, y2 = bbox
    person = mask[y1:y2, x1:x2]
    ph, pw = person.shape[:2]
    if ph == 0 or pw == 0:
        return canvas

    avail_h = out_h - 2 * pad
    avail_w = out_w - 2 * pad
    if avail_h <= 0 or avail_w <= 0:
        raise ValueError("输出尺寸太小或 pad 太大。")

    scale = min(avail_h / ph, avail_w / pw)
    new_h = max(1, int(round(ph * scale)))
    new_w = max(1, int(round(pw * scale)))

    resized = cv2.resize(person, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    start_x = max(0, (out_w - new_w) // 2)
    start_y = max(0, out_h - pad - new_h)
    end_x = min(out_w, start_x + new_w)
    end_y = min(out_h, start_y + new_h)

    canvas[start_y:end_y, start_x:end_x] = resized[:end_y - start_y, :end_x - start_x]
    return (canvas > 0).astype(np.uint8) * 255


def morphological_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    img = (binary_mask > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        if cv2.countNonZero(img) == 0:
            break

    return (skel > 0).astype(np.uint8) * 255


def extract_pseudo_keypoints(skeleton: np.ndarray):
    sk = (skeleton > 0).astype(np.uint8)
    sk_float = sk.astype(np.float32)

    kernel = np.ones((3, 3), dtype=np.float32)
    neighbor_sum = cv2.filter2D(sk_float, ddepth=cv2.CV_32F, kernel=kernel)
    neighbor_count = neighbor_sum - sk_float

    endpoints = ((sk == 1) & (neighbor_count == 1)).astype(np.uint8) * 255
    junctions = ((sk == 1) & (neighbor_count >= 3)).astype(np.uint8) * 255
    return endpoints, junctions


def build_pseudo_skeleton_map(norm_mask: np.ndarray, heat_sigma: float = 1.2):
    skeleton = morphological_skeleton(norm_mask)
    endpoints, junctions = extract_pseudo_keypoints(skeleton)

    keypoint_map = np.maximum(endpoints, junctions).astype(np.float32) / 255.0
    if keypoint_map.max() > 0:
        keypoint_map = cv2.GaussianBlur(keypoint_map, (0, 0), heat_sigma)
        if keypoint_map.max() > 0:
            keypoint_map = keypoint_map / keypoint_map.max()

    limb_map = (skeleton > 0).astype(np.float32)
    skeleton_map = np.stack([keypoint_map, limb_map], axis=0).astype(np.float32)

    vis = np.zeros((norm_mask.shape[0], norm_mask.shape[1], 3), dtype=np.uint8)
    vis[:, :, 1] = (limb_map * 255).astype(np.uint8)
    vis[endpoints > 0] = (0, 0, 255)
    vis[junctions > 0] = (255, 0, 0)
    return skeleton_map, vis


def should_skip_frame(sil_out_path: Path, ske_out_path: Path, vis_out_path: Path | None, save_skeleton_vis: bool) -> bool:
    if not sil_out_path.exists():
        return False
    if not ske_out_path.exists():
        return False
    if save_skeleton_vis and vis_out_path is not None and (not vis_out_path.exists()):
        return False
    return True


def write_progress(output_root: Path, mode: str, processed_sequences: int, total_sequences: int,
                   processed_frames: int, skipped_existing_frames: int, failed_frames: int) -> None:
    progress = {
        "mode": mode,
        "processed_sequences": processed_sequences,
        "total_sequences": total_sequences,
        "processed_frames": processed_frames,
        "skipped_existing_frames": skipped_existing_frames,
        "failed_frames": failed_frames
    }
    save_json(progress, output_root / "progress.json")


def process_casia_b(input_root: Path, output_root: Path, out_h: int = 64, out_w: int = 44,
                    pad: int = 2, save_skeleton_vis: bool = True) -> None:
    archive_groups, archive_skipped = scan_from_archives(input_root)
    extracted_groups, extracted_skipped = scan_from_extracted_dirs(input_root)

    if len(archive_groups) > 0:
        mode = "archives"
        groups = archive_groups
        skipped = archive_skipped
    elif len(extracted_groups) > 0:
        mode = "extracted_dirs"
        groups = extracted_groups
        skipped = extracted_skipped
    else:
        raise ValueError(f"在输入目录中没有成功解析到 CASIA-B 图像：{input_root}")

    silhouettes_root = output_root / "silhouettes"
    skeletons_root = output_root / "skeletons"
    skeleton_vis_root = output_root / "skeleton_vis"

    ensure_dir(silhouettes_root)
    ensure_dir(skeletons_root)
    if save_skeleton_vis:
        ensure_dir(skeleton_vis_root)

    total_seq = len(groups)
    processed_seq = 0
    total_frames_written = 0
    total_frames_skipped_existing = 0
    failed_frames = 0

    write_progress(output_root, mode, 0, total_seq, 0, 0, 0)

    for (pid, seq_name), items in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        sil_seq_dir = silhouettes_root / pid / seq_name
        ske_seq_dir = skeletons_root / pid / seq_name
        vis_seq_dir = skeleton_vis_root / pid / seq_name

        ensure_dir(sil_seq_dir)
        ensure_dir(ske_seq_dir)
        if save_skeleton_vis:
            ensure_dir(vis_seq_dir)

        seq_written = 0
        seq_skipped = 0

        for new_idx, item in enumerate(items):
            out_name = f"frame_{new_idx:06d}"
            sil_out_path = sil_seq_dir / f"{out_name}.png"
            ske_out_path = ske_seq_dir / f"{out_name}.npy"
            vis_out_path = (vis_seq_dir / f"{out_name}.png") if save_skeleton_vis else None

            if should_skip_frame(sil_out_path, ske_out_path, vis_out_path, save_skeleton_vis):
                seq_skipped += 1
                total_frames_skipped_existing += 1
                continue

            try:
                if mode == "archives":
                    _, archive_path, member_name = item
                    raw_img = read_image_from_archive(archive_path, member_name)
                else:
                    _, img_path = item
                    raw_img = read_image_from_path(img_path)

                mask = (raw_img > 0).astype(np.uint8) * 255
                norm_mask = normalize_one_silhouette(mask, out_h=out_h, out_w=out_w, pad=pad)
                skeleton_map, vis_img = build_pseudo_skeleton_map(norm_mask)

                cv2.imwrite(str(sil_out_path), norm_mask)
                np.save(str(ske_out_path), skeleton_map)
                if save_skeleton_vis and vis_out_path is not None:
                    cv2.imwrite(str(vis_out_path), vis_img)

                seq_written += 1
                total_frames_written += 1
            except Exception as e:
                failed_frames += 1
                print(f"[失败] {pid}/{seq_name}/{out_name}: {e}")

        processed_seq += 1
        print(f"[完成] {pid}/{seq_name} -> 总帧数 {len(items)} | 本次新写入 {seq_written} | 已存在跳过 {seq_skipped}")

        write_progress(
            output_root=output_root,
            mode=mode,
            processed_sequences=processed_seq,
            total_sequences=total_seq,
            processed_frames=total_frames_written,
            skipped_existing_frames=total_frames_skipped_existing,
            failed_frames=failed_frames
        )

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "mode": mode,
        "num_sequences": total_seq,
        "newly_written_frames": total_frames_written,
        "skipped_existing_frames": total_frames_skipped_existing,
        "failed_frames": failed_frames,
        "num_skipped_parse_files": len(skipped),
        "skipped_parse_files_preview": skipped[:50],
        "resume_supported": True
    }
    save_json(summary, output_root / "prepare_summary.json")

    print("=" * 72)
    print("CASIA-B 数据准备完成")
    print(f"模式: {mode}")
    print(f"序列数: {total_seq}")
    print(f"本次新写入帧数: {total_frames_written}")
    print(f"已存在跳过帧数: {total_frames_skipped_existing}")
    print(f"失败帧数: {failed_frames}")
    print(f"silhouettes 输出: {silhouettes_root}")
    print(f"skeletons 输出:   {skeletons_root}")
    if save_skeleton_vis:
        print(f"skeleton_vis 输出: {skeleton_vis_root}")
    print("=" * 72)


def build_parser():
    parser = argparse.ArgumentParser(description="CASIA-B 数据整理脚本（支持断点续跑）")
    parser.add_argument("--input_root", type=str, required=True, help="CASIA-B silhouette 根目录")
    parser.add_argument("--output_root", type=str, required=True, help="输出根目录")
    parser.add_argument("--img_h", type=int, default=64, help="输出高度")
    parser.add_argument("--img_w", type=int, default=44, help="输出宽度")
    parser.add_argument("--pad", type=int, default=2, help="标准化 silhouette 时边界留白")
    parser.add_argument("--no_skeleton_vis", action="store_true", help="不保存伪骨架可视化图")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    process_casia_b(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        out_h=args.img_h,
        out_w=args.img_w,
        pad=args.pad,
        save_skeleton_vis=not args.no_skeleton_vis
    )
