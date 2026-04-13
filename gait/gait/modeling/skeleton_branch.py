import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 1. 基础卷积模块
# =========================================================
class ConvBNReLU(nn.Module):
    """
    基本卷积块：Conv -> BN -> ReLU

    和剪影分支保持一致，方便后续统一理解和维护。

    为什么骨架分支也用 2D CNN：
    - 你现在的骨架输入不是原始关键点坐标，而是 Skeleton Map
    - Skeleton Map 本质上是“二维图像”
    - 所以直接用 2D CNN 提空间结构特征是最自然、最稳妥的做法

    输入:
        [N, C_in, H, W]

    输出:
        [N, C_out, H, W]
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# =========================================================
# 2. 骨架帧级特征提取主干网络
# =========================================================
class SkeletonBackbone(nn.Module):
    """
    骨架帧级特征提取骨干网络

    输入:
        [B*T, 2, H, W]

    输出:
        [B*T, C, H', W']

    与剪影分支相比的主要区别：
    - 输入通道数从 1 变成 2
    - 这 2 个通道分别表示：
        channel 0: 关节点热图
        channel 1: 骨骼连线图

    为什么不直接上 GCN / ST-GCN / Transformer：
    - 你现在的毕设目标不是把骨架分支做到“最先进”
    - 而是先提一个稳定、可训练、可用于融合的骨架特征
    - 当前阶段，Skeleton Map + CNN 是更稳的路线

    网络结构思路：
    - 前两层提局部骨架纹理与连线关系
    - 后两层提更高层的人体姿态结构表示
    - 中间用 MaxPool 下采样，减小特征图尺寸、增加感受野
    """
    def __init__(self, in_channels: int = 2, out_channels: int = 256):
        super().__init__()

        self.features = nn.Sequential(
            # 输入: [B*T, 2, 64, 44]
            ConvBNReLU(in_channels, 64, 3, 1),

            # 64x44 -> 32x22
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBNReLU(64, 128, 3, 1),

            # 32x22 -> 16x11
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBNReLU(128, 256, 3, 1),
            ConvBNReLU(256, out_channels, 3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


# =========================================================
# 3. 全局池化模块
# =========================================================
class GlobalPooling(nn.Module):
    """
    全局池化模块

    输入:
        [B, C, H, W]

    输出:
        [B, C]

    为什么骨架分支先用全局池化而不是水平分块池化：
    - Skeleton Map 的重点更多在“结构关系”
    - 与剪影相比，它的外形轮廓细节更少
    - 当前阶段先用全局池化更简单、更稳
    - 后面如果实验需要，也可以尝试给骨架分支加水平分块池化

    这里同样使用 avg + max 的组合：
    - avg 反映整体结构分布
    - max 强调最显著响应
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = F.adaptive_avg_pool2d(x, output_size=(1, 1)).flatten(1)
        max_pool = F.adaptive_max_pool2d(x, output_size=(1, 1)).flatten(1)
        return avg_pool + max_pool


# =========================================================
# 4. 骨架分支主模型
# =========================================================
class SkeletonBranch(nn.Module):
    """
    骨架分支（简化版）

    输入:
        x: [B, T, 2, H, W]

    输出:
        {
            "embedding": [B, feature_dim],     # 后续融合/检索/Triplet Loss 用
            "bn_embedding": [B, feature_dim],  # BNNeck 后特征，适合接分类器
            "logits": [B, num_classes]         # 分类输出
        }

    流程：
        Skeleton Map 序列
        -> 每帧骨架图经过 CNN 提帧级特征
        -> 时间维最大池化
        -> 全局池化
        -> 全连接层映射到固定维度
        -> BNNeck
        -> 分类头

    为什么时间维还是用 max pooling：
    - 简单、稳定
    - 作为基线很合适
    - 后面如果需要，可以扩展成时序卷积/注意力/Transformer
    """
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 2,
        feature_dim: int = 256,
        dropout: float = 0.0
    ):
        super().__init__()

        self.backbone = SkeletonBackbone(in_channels=in_channels, out_channels=256)
        self.global_pool = GlobalPooling()

        self.fc = nn.Linear(256, feature_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # BNNeck，和剪影分支保持一致
        self.bnneck = nn.BatchNorm1d(feature_dim)
        self.bnneck.bias.requires_grad_(False)

        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> dict:
        """
        输入:
            x: [B, T, 2, H, W]

        输出:
            dict
        """
        if x.ndim != 5:
            raise ValueError(
                f"输入张量维度应为 5，当前为 {x.ndim}，期望形状 [B, T, 2, H, W]"
            )

        b, t, c, h, w = x.shape

        # -------------------------------------------------
        # 第一步：合并 batch 和时间维
        # -------------------------------------------------
        # 把每一帧 Skeleton Map 当作独立图像处理
        x = x.view(b * t, c, h, w)  # [B*T, 2, H, W]

        # -------------------------------------------------
        # 第二步：提取帧级骨架特征
        # -------------------------------------------------
        frame_feat = self.backbone(x)  # [B*T, 256, H', W']

        # -------------------------------------------------
        # 第三步：恢复时间维
        # -------------------------------------------------
        _, c2, h2, w2 = frame_feat.shape
        frame_feat = frame_feat.view(b, t, c2, h2, w2)  # [B, T, 256, H', W']

        # -------------------------------------------------
        # 第四步：时间维最大池化
        # -------------------------------------------------
        # 聚合整段步态中的显著姿态结构模式
        seq_feat, _ = torch.max(frame_feat, dim=1)  # [B, 256, H', W']

        # -------------------------------------------------
        # 第五步：全局池化
        # -------------------------------------------------
        pooled_feat = self.global_pool(seq_feat)  # [B, 256]

        # -------------------------------------------------
        # 第六步：映射到 embedding
        # -------------------------------------------------
        embedding = self.fc(pooled_feat)  # [B, feature_dim]
        embedding = self.dropout(embedding)

        # -------------------------------------------------
        # 第七步：BNNeck + 分类头
        # -------------------------------------------------
        bn_embedding = self.bnneck(embedding)
        logits = self.classifier(bn_embedding)

        return {
            "embedding": embedding,
            "bn_embedding": bn_embedding,
            "logits": logits
        }


# =========================================================
# 5. 简单自测
# =========================================================
if __name__ == "__main__":
    """
    自测说明：
    - 构造一个假的 Skeleton Map 序列输入
    - 检查网络能否正常前向传播
    - 检查输出维度是否符合预期

    输入形状:
        [B, T, C, H, W]
        B = 2
        T = 30
        C = 2
        H = 64
        W = 44
    """
    model = SkeletonBranch(
        num_classes=100,
        in_channels=2,
        feature_dim=256,
        dropout=0.1
    )

    x = torch.randn(2, 30, 2, 64, 44)
    out = model(x)

    print("embedding:", out["embedding"].shape)        # [2, 256]
    print("bn_embedding:", out["bn_embedding"].shape)  # [2, 256]
    print("logits:", out["logits"].shape)              # [2, 100]