import torch
from pathlib import Path
from torch.utils.data import DataLoader

# ---------------------------------------------------------
# 从同目录导入你已经写好的模块
# 如果这个脚本放在 modeling/ 目录下，这种导入方式最直接
# ---------------------------------------------------------
from silhouette_dataset import SilhouetteSequenceDataset
from skeleton_dataset import SkeletonSequenceDataset
from silhouette_branch import SilhouetteBranch
from skeleton_branch import SkeletonBranch


def test_silhouette_pipeline(batch_size: int = 2, seq_len: int = 30):
    """
    测试：
    SilhouetteSequenceDataset -> DataLoader -> SilhouetteBranch

    目标：
    1. 检查 Dataset 输出是不是 [T, 1, 64, 44]
    2. 检查 DataLoader 输出是不是 [B, T, 1, 64, 44]
    3. 检查模型前向传播是否正常
    4. 检查输出 embedding / bn_embedding / logits 的形状
    """
    print("=" * 60)
    print("开始测试：剪影分支联调")
    print("=" * 60)

    current_dir = Path(__file__).resolve().parent
    silhouette_root = current_dir.parent / "data" / "silhouettes"

    dataset = SilhouetteSequenceDataset(
        root_dir=str(silhouette_root),
        seq_len=seq_len,
        img_h=64,
        img_w=44,
        train=True
    )

    print(f"[剪影] 数据目录: {silhouette_root}")
    print(f"[剪影] 样本总数: {len(dataset)}")

    if len(dataset) == 0:
        print("[剪影] 数据集为空，跳过测试。")
        return

    # 取单个样本检查
    x_single, y_single = dataset[0]
    print(f"[剪影] 单个样本 x.shape: {x_single.shape}")  # [T, 1, 64, 44]
    print(f"[剪影] 单个样本 y: {y_single}")

    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    batch_x, batch_y = next(iter(loader))
    print(f"[剪影] batch_x.shape: {batch_x.shape}")  # [B, T, 1, 64, 44]
    print(f"[剪影] batch_y.shape: {batch_y.shape}")  # [B]

    # 模型
    num_classes = max(len(dataset.label_map), 1)

    model = SilhouetteBranch(
        num_classes=num_classes,
        in_channels=1,
        feature_dim=256,
        num_bins=4,
        dropout=0.1
    )

    model.eval()

    with torch.no_grad():
        out = model(batch_x)

    print(f"[剪影] embedding.shape: {out['embedding'].shape}")        # [B, 256]
    print(f"[剪影] bn_embedding.shape: {out['bn_embedding'].shape}")  # [B, 256]
    print(f"[剪影] logits.shape: {out['logits'].shape}")              # [B, num_classes]")

    # 额外断言，确保维度完全符合预期
    assert batch_x.ndim == 5, "剪影 batch_x 维度应为 5"
    assert batch_x.shape[2] == 1, "剪影输入通道数应为 1"
    assert out["embedding"].shape[0] == batch_x.shape[0], "剪影 embedding 的 batch 维不匹配"
    assert out["embedding"].shape[1] == 256, "剪影 embedding 维度应为 256"
    assert out["logits"].shape[1] == num_classes, "剪影 logits 类别数不匹配"

    print("[剪影] 联调测试通过。")
    print()


def test_skeleton_pipeline(batch_size: int = 2, seq_len: int = 30):
    """
    测试：
    SkeletonSequenceDataset -> DataLoader -> SkeletonBranch

    目标：
    1. 检查 Dataset 输出是不是 [T, 2, 64, 44]
    2. 检查 DataLoader 输出是不是 [B, T, 2, 64, 44]
    3. 检查模型前向传播是否正常
    4. 检查输出 embedding / bn_embedding / logits 的形状
    """
    print("=" * 60)
    print("开始测试：骨架分支联调")
    print("=" * 60)

    current_dir = Path(__file__).resolve().parent
    skeleton_root = current_dir.parent / "data" / "skeletons"

    dataset = SkeletonSequenceDataset(
        root_dir=str(skeleton_root),
        seq_len=seq_len,
        img_h=64,
        img_w=44,
        train=True
    )

    print(f"[骨架] 数据目录: {skeleton_root}")
    print(f"[骨架] 样本总数: {len(dataset)}")

    if len(dataset) == 0:
        print("[骨架] 数据集为空，跳过测试。")
        return

    # 取单个样本检查
    x_single, y_single = dataset[0]
    print(f"[骨架] 单个样本 x.shape: {x_single.shape}")  # [T, 2, 64, 44]
    print(f"[骨架] 单个样本 y: {y_single}")

    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    batch_x, batch_y = next(iter(loader))
    print(f"[骨架] batch_x.shape: {batch_x.shape}")  # [B, T, 2, 64, 44]
    print(f"[骨架] batch_y.shape: {batch_y.shape}")  # [B]

    # 模型
    num_classes = max(len(dataset.label_map), 1)

    model = SkeletonBranch(
        num_classes=num_classes,
        in_channels=2,
        feature_dim=256,
        dropout=0.1
    )

    model.eval()

    with torch.no_grad():
        out = model(batch_x)

    print(f"[骨架] embedding.shape: {out['embedding'].shape}")        # [B, 256]
    print(f"[骨架] bn_embedding.shape: {out['bn_embedding'].shape}")  # [B, 256]
    print(f"[骨架] logits.shape: {out['logits'].shape}")              # [B, num_classes]

    # 额外断言
    assert batch_x.ndim == 5, "骨架 batch_x 维度应为 5"
    assert batch_x.shape[2] == 2, "骨架输入通道数应为 2"
    assert out["embedding"].shape[0] == batch_x.shape[0], "骨架 embedding 的 batch 维不匹配"
    assert out["embedding"].shape[1] == 256, "骨架 embedding 维度应为 256"
    assert out["logits"].shape[1] == num_classes, "骨架 logits 类别数不匹配"

    print("[骨架] 联调测试通过。")
    print()


if __name__ == "__main__":
    """
    主入口：
    依次测试剪影分支和骨架分支。
    只要都不报错，并打印出正确 shape，就说明：
    - Dataset 没问题
    - DataLoader 没问题
    - 模型前向传播没问题
    - 数据和模型已经成功接上了
    """
    test_silhouette_pipeline(batch_size=2, seq_len=30)
    test_skeleton_pipeline(batch_size=2, seq_len=30)

    print("=" * 60)
    print("所有 dataset + branch 联调测试完成。")
    print("=" * 60)