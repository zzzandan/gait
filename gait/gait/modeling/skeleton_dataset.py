import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SkeletonSequenceDataset(Dataset):
    """
    骨架序列数据集（Skeleton Map 版本）

    目录结构示例：
    data/skeletons/
        0001/
            seq01/
                frame_000000.npy
                frame_000001.npy
                ...
            seq02/
                ...
        0002/
            seq01/
            seq02/

    每个 .npy 文件表示一帧 Skeleton Map，
    默认 shape 为:
        [2, 64, 44]

    返回：
        x: [T, 2, 64, 44]
        y: int

    设计思路和 silhouette_dataset.py 保持一致：
    - 单个样本 = 一整个序列文件夹
    - 每次从一个序列里采样 T 帧
    - 输出给模型的格式是 [T, C, H, W]
    - DataLoader 再自动拼成 [B, T, C, H, W]
    """

    def __init__(
        self,
        root_dir: str,
        seq_len: int = 30,
        img_h: int = 64,
        img_w: int = 44,
        train: bool = True,
        return_meta: bool = True
    ):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.img_h = img_h
        self.img_w = img_w
        self.train = train
        self.return_meta = return_meta

        # 保存所有样本的索引信息
        self.samples = []

        # 身份字符串 -> 分类标签整数
        self.label_map = {}

        self._build_index()

    def _build_index(self):
        """
        扫描根目录，建立样本索引。

        逻辑：
        - 第一层子目录视为 person_id
        - 第二层子目录视为 seq_name
        - 每个 seq 文件夹对应一个样本
        """
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"找不到数据目录: {self.root_dir}")

        person_ids = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])

        # 给每个身份分配一个整数标签
        self.label_map = {pid: idx for idx, pid in enumerate(person_ids)}

        for pid in person_ids:
            person_dir = os.path.join(self.root_dir, pid)

            seq_names = sorted([
                d for d in os.listdir(person_dir)
                if os.path.isdir(os.path.join(person_dir, d))
            ])

            for seq_name in seq_names:
                seq_dir = os.path.join(person_dir, seq_name)

                frame_paths = sorted([
                    os.path.join(seq_dir, f)
                    for f in os.listdir(seq_dir)
                    if f.lower().endswith(".npy")
                ])

                # 空文件夹直接跳过
                if len(frame_paths) == 0:
                    continue

                condition = seq_name.split("_", 1)[0] if "_" in seq_name else "unknown"
                self.samples.append({
                    "person_id": pid,
                    "label": self.label_map[pid],
                    "seq_name": seq_name,
                    "condition": condition,
                    "seq_dir": seq_dir,
                    "frame_paths": frame_paths
                })

    def __len__(self):
        return len(self.samples)

    def _sample_indices(self, num_frames: int):
        """
        从一个序列中采样 seq_len 帧。

        train=True:
            - 随机采样，更适合训练
        train=False:
            - 均匀采样，更适合验证/测试

        处理帧数不足的情况：
        - 如果序列长度 < seq_len，就允许重复采样补齐
        """
        if num_frames >= self.seq_len:
            if self.train:
                # 随机采样，但保持时间顺序
                indices = np.sort(
                    np.random.choice(num_frames, self.seq_len, replace=False)
                )
            else:
                # 均匀采样
                indices = np.linspace(0, num_frames - 1, self.seq_len).astype(int)
        else:
            # 帧数不够，重复采样补齐
            if self.train:
                indices = np.sort(
                    np.random.choice(num_frames, self.seq_len, replace=True)
                )
            else:
                indices = np.linspace(0, num_frames - 1, self.seq_len)
                indices = np.round(indices).astype(int)

        return indices

    def _load_one_frame(self, npy_path: str):
        """
        读取单帧 Skeleton Map，并确保输出为 [2, H, W]

        默认输入文件是 .npy，shape 应为：
            [2, 64, 44]

        这里会做几件事：
        1. 加载 npy
        2. 检查维度
        3. 如果尺寸不是目标尺寸，则做 resize
        4. 转成 float32
        """
        arr = np.load(npy_path)

        if arr.ndim != 3:
            raise ValueError(f"文件维度错误，期望 [C, H, W]，实际为 {arr.shape}，文件: {npy_path}")

        # 期望通道在第 0 维
        c, h, w = arr.shape

        if c != 2:
            raise ValueError(f"Skeleton Map 通道数应为 2，实际为 {c}，文件: {npy_path}")

        # 如果尺寸不一致，就逐通道 resize 到统一尺寸
        if h != self.img_h or w != self.img_w:
            resized = np.zeros((2, self.img_h, self.img_w), dtype=np.float32)
            for i in range(2):
                resized[i] = self._resize_one_channel(arr[i], self.img_h, self.img_w)
            arr = resized
        else:
            arr = arr.astype(np.float32)

        return arr

    @staticmethod
    def _resize_one_channel(channel_map: np.ndarray, out_h: int, out_w: int):
        """
        对单通道骨架图做 resize。

        为什么这里用最近邻插值：
        - 骨骼线图和热图都不希望被过度平滑
        - 最近邻更简单直接，也更不容易引入奇怪模糊
        """
        import cv2
        resized = cv2.resize(channel_map, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        return resized.astype(np.float32)

    def __getitem__(self, idx: int):
        """
        返回一个样本：
            x: [T, 2, H, W]
            y: 标签（long）
        """
        sample = self.samples[idx]
        frame_paths = sample["frame_paths"]
        label = sample["label"]

        indices = self._sample_indices(len(frame_paths))

        frames = []
        for i in indices:
            frame = self._load_one_frame(frame_paths[i])   # [2, H, W]
            frames.append(frame)

        # 堆叠为 [T, 2, H, W]
        x = np.stack(frames, axis=0).astype(np.float32)

        # 转成 torch tensor
        x = torch.from_numpy(x)                  # [T, 2, H, W]
        y = torch.tensor(label, dtype=torch.long)

        if not self.return_meta:
            return x, y

        meta = {
            "person_id": sample["person_id"],
            "seq_name": sample.get("seq_name", os.path.basename(sample["seq_dir"])),
            "condition": sample.get("condition", "unknown"),
        }
        return x, y, meta


if __name__ == "__main__":
    """
    自测代码：
    1. 自动定位项目内的 data/skeletons
    2. 测试单个样本输出
    3. 测试 DataLoader 拼 batch 后的输出

    注意：
    - 这里默认脚本位于 modeling/ 目录下
    - 所以 current_dir.parent / "data" / "skeletons"
      会定位到项目里的 gait/data/skeletons
    """
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent / "data" / "skeletons"

    dataset = SkeletonSequenceDataset(
        root_dir=str(root_dir),
        seq_len=30,
        img_h=64,
        img_w=44,
        train=True
    )

    print("样本总数:", len(dataset))

    if len(dataset) > 0:
        x, y, meta = dataset[0]
        print("单个样本 x.shape:", x.shape)  # [T, 2, 64, 44]
        print("单个样本 y:", y)
        print("单个样本 meta:", meta)

        loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
        batch_x, batch_y, batch_meta = next(iter(loader))

        print("batch_x.shape:", batch_x.shape)  # [B, T, 2, 64, 44]
        print("batch_y.shape:", batch_y.shape)  # [B]

        # batch_meta 是一个字典，里面每个键对应一个长度为 B 的列表
        print("batch_meta keys:", batch_meta.keys())
        print("batch_meta['person_id']:", batch_meta["person_id"])
        print("batch_meta['condition']:", batch_meta["condition"])
        print("batch_meta['seq_name']:", batch_meta["seq_name"])
    else:
        print("数据集为空，请检查 data/skeletons 目录结构。")