import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SilhouetteSequenceDataset(Dataset):
    """
    剪影序列数据集

    目录结构示例：
    data/silhouettes/
        0001/
            seq01/
                000001.png
                000002.png
                ...
            seq02/
                ...
        0002/
            seq01/
            seq02/

    返回：
        x: [T, 1, 64, 44]
        y: int
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

        self.samples = []
        self.label_map = {}

        self._build_index()

    def _build_index(self):
        """
        扫描目录，建立样本索引。
        每个样本对应一个序列文件夹。
        """
        person_ids = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])

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
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ])

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
            随机采样（更适合训练）
        train=False:
            均匀采样（更适合验证/测试）
        """
        if num_frames >= self.seq_len:
            if self.train:
                # 随机从序列中选 seq_len 帧，并保持顺序
                indices = np.sort(
                    np.random.choice(num_frames, self.seq_len, replace=False)
                )
            else:
                # 均匀采样
                indices = np.linspace(0, num_frames - 1, self.seq_len).astype(int)
        else:
            # 如果帧数不足，就重复采样到 seq_len
            if self.train:
                indices = np.sort(
                    np.random.choice(num_frames, self.seq_len, replace=True)
                )
            else:
                indices = np.linspace(0, num_frames - 1, self.seq_len)
                indices = np.round(indices).astype(int)

        return indices

    def _load_one_frame(self, img_path: str):
        """
        读取单张剪影图，并处理成 [1, 64, 44]
        """
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")

        # 统一尺寸
        img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)

        # 转成 float32，并归一化到 [0,1]
        img = img.astype(np.float32) / 255.0

        # 如果你想强制二值化，可以取消下面注释
        # img = (img > 0.5).astype(np.float32)

        # 增加通道维，变成 [1, H, W]
        img = np.expand_dims(img, axis=0)

        return img

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        frame_paths = sample["frame_paths"]
        label = sample["label"]

        indices = self._sample_indices(len(frame_paths))

        frames = []
        for i in indices:
            frame = self._load_one_frame(frame_paths[i])   # [1, H, W]
            frames.append(frame)

        # 堆成 [T, 1, H, W]
        x = np.stack(frames, axis=0).astype(np.float32)

        # 转成 torch tensor
        x = torch.from_numpy(x)                  # [T, 1, 64, 44]
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
    dataset = SilhouetteSequenceDataset(
        root_dir="/home/zzzandan/desk/gait/gait/gait/data/silhouettes",
        seq_len=30,
        img_h=64,
        img_w=44,
        train=True
    )

    print("样本总数:", len(dataset))

    x, y, meta = dataset[0]
    print("单个样本 x.shape:", x.shape)
    print("单个样本 y:", y)
    print("单个样本 meta:", meta)

    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    batch_x, batch_y, batch_meta = next(iter(loader))
    print("batch_x.shape:", batch_x.shape)
    print("batch_y.shape:", batch_y.shape)
    print("batch_meta keys:", batch_meta.keys())