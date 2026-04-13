import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 1. 基础卷积模块
# =========================================================
class ConvBNReLU(nn.Module):
    """
    一个最基础的卷积块：Conv -> BatchNorm -> ReLU

    设计目的：
    1. Conv2d 负责提取空间特征
    2. BatchNorm2d 负责稳定训练、加快收敛
    3. ReLU 提供非线性表达能力

    为什么这里不用更复杂的结构：
    - 第一目标是先把“剪影单模态分支”稳定跑通；
    - 在这个阶段，不需要一开始就上残差块、注意力模块、深层网络；
    - 简单的卷积块更容易调试，也更适合后续和骨架分支保持结构对称。

    输入：
        x: [N, C_in, H, W]

    输出：
        y: [N, C_out, H, W]（如果 stride=1 且 padding 合适，高宽基本不变）
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()

        # 对于 3x3 卷积，padding=1 可以保持空间尺寸不变
        # 一般 padding = kernel_size // 2 是最常见的写法
        padding = kernel_size // 2

        self.block = nn.Sequential(
            # bias=False 是因为后面接了 BatchNorm，bias 往往可以省略
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
# 2. 剪影帧级特征提取主干网络
# =========================================================
class SilhouetteBackbone(nn.Module):
    """
    剪影帧级特征提取骨干网络（frame-level backbone）

    这里的“帧级”意思是：
    - 它一次只处理单帧剪影图
    - 不直接处理时间维
    - 时间维会在后面单独做 pooling

    输入：
        [B*T, 1, H, W]

    输出：
        [B*T, C, H', W']

    为什么先做帧级特征提取：
    - 剪影序列本质上是“很多帧单通道图像”
    - 最自然的做法就是先对每帧提空间特征
    - 然后再在时间维度上聚合，得到整段步态序列的表示

    网络结构设计思路：
    - 第一层提取低级轮廓边缘信息
    - 第二层提取更抽象的局部形状信息
    - 第三、四层提取更高层的人体结构表示
    - 中间插入 MaxPool 减小空间尺寸、增加感受野

    当前版本是“简化版 GaitBase 风格”：
    - 不追求复杂
    - 重点是稳定、能收敛、能输出可用 embedding
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 256):
        super().__init__()

        self.features = nn.Sequential(
            # 输入通常是 [B*T, 1, 64, 44]
            # 输出 [B*T, 64, 64, 44]
            ConvBNReLU(in_channels, 64, 3, 1),

            # 空间下采样：64x44 -> 32x22
            # 这样可以减少计算量，同时扩大后续特征的感受野
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 输出 [B*T, 128, 32, 22]
            ConvBNReLU(64, 128, 3, 1),

            # 空间下采样：32x22 -> 16x11
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 输出 [B*T, 256, 16, 11]
            ConvBNReLU(128, 256, 3, 1),

            # 再加一层同维度卷积，提高高层表达能力
            # 输出 [B*T, out_channels, 16, 11]
            ConvBNReLU(256, out_channels, 3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


# =========================================================
# 3. 水平分块池化模块
# =========================================================
class HorizontalPooling(nn.Module):
    """
    水平分块池化（Horizontal Pooling）

    核心思想：
    - 人体步态并不是“整个人都一样重要”
    - 上半身、躯干、腿部、脚部的运动模式不同
    - 把特征图沿“高度方向”切成若干水平块，可以让模型分别关注不同身体区域

    为什么这一步很重要：
    - 剪影任务里，腿部和下肢对步态识别通常非常关键
    - 如果直接对整张特征图做全局池化，容易丢掉部位差异
    - 水平分块池化可以保留“局部身体区域”的结构信息

    当前做法：
    1. 按高度方向把特征图分成 num_bins 块
    2. 每块分别做：
       - 自适应平均池化
       - 自适应最大池化
    3. 两者相加后作为该块特征
    4. 最后把所有块拼接起来

    输入：
        [B, C, H, W]

    输出：
        [B, C * num_bins]

    为什么用 avg + max：
    - avg pooling 更平滑，反映整体分布
    - max pooling 更强调最显著响应
    - 两者相加通常比只用一种更稳

    为什么不是更复杂的 HPP（金字塔池化）：
    - 现在阶段先做简单且稳定的版本
    - 后续如果实验需要，可以扩展成多尺度 HPP
    """
    def __init__(self, num_bins: int = 4):
        super().__init__()
        self.num_bins = num_bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # -------------------------------------------------
        # 为了能平均分块，如果 H 不能整除 num_bins，就在高度方向做补齐
        # -------------------------------------------------
        remainder = h % self.num_bins
        if remainder != 0:
            pad_h = self.num_bins - remainder

            # pad 参数顺序: (left, right, top, bottom)
            # 这里只在底部补 pad_h 行
            # mode="replicate" 表示复制边缘值，避免引入过强的人工边界
            x = F.pad(x, (0, 0, 0, pad_h), mode="replicate")
            h = h + pad_h

        bin_h = h // self.num_bins
        pooled_features = []

        for i in range(self.num_bins):
            # 取第 i 个水平块
            part = x[:, :, i * bin_h:(i + 1) * bin_h, :]   # [B, C, bin_h, W]

            # 自适应平均池化到 1x1
            avg_pool = F.adaptive_avg_pool2d(part, output_size=(1, 1)).view(b, c)

            # 自适应最大池化到 1x1
            max_pool = F.adaptive_max_pool2d(part, output_size=(1, 1)).view(b, c)

            # 两种池化结果相加
            pooled = avg_pool + max_pool

            pooled_features.append(pooled)

        # 将不同水平块的特征拼接起来
        # 最终维度：[B, C * num_bins]
        return torch.cat(pooled_features, dim=1)


# =========================================================
# 4. 剪影分支主模型
# =========================================================
class SilhouetteBranch(nn.Module):
    """
    剪影分支（简化版）

    这是一个“序列输入 -> 单个特征向量输出”的剪影特征提取器。

    输入：
        x: [B, T, 1, H, W]

    输出：
        {
            "embedding": [B, feature_dim],     # 真正用于后续融合/检索的特征
            "bn_embedding": [B, feature_dim],  # BNNeck 后的特征，适合接分类头
            "logits": [B, num_classes]         # 分类结果，用于交叉熵训练
        }

    模块逻辑：
        剪影序列
        -> 每帧用 backbone 提特征
        -> 时间维最大池化
        -> 水平分块池化
        -> FC 压缩到固定维度
        -> BNNeck
        -> 分类头

    为什么先做单模态分支，而不是一上来就做融合：
    - 因为你必须先确认“剪影本身能不能学到有效特征”
    - 单模态模型跑通以后，后面和骨架融合才有意义
    - 否则你根本不知道融合后提升来自哪里

    为什么输出分成 embedding / bn_embedding / logits：
    - embedding：后面真正做融合、检索、triplet loss 用这个
    - bn_embedding：分类前常做 BNNeck，训练更稳
    - logits：做分类监督（CrossEntropyLoss）
    """
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        feature_dim: int = 256,
        num_bins: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()

        # 帧级特征提取网络
        self.backbone = SilhouetteBackbone(in_channels=in_channels, out_channels=256)

        # 水平分块池化模块
        self.hpm = HorizontalPooling(num_bins=num_bins)

        # 全连接层，将 [256 * num_bins] 压缩到 feature_dim
        # 例如 num_bins=4 时，输入维度是 1024，输出维度是 256
        self.fc = nn.Linear(256 * num_bins, feature_dim)

        # dropout 不是必须的，但有时能缓解过拟合
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # -------------------------------------------------
        # BNNeck 风格
        # -------------------------------------------------
        # 这是很多识别任务中常见的技巧：
        # - embedding 用于度量学习（triplet loss）
        # - BN 后的特征用于分类头
        #
        # bias 不更新是常见做法，因为分类前的 BN 偏置对识别贡献不大，
        # 且固定它通常更稳定。
        self.bnneck = nn.BatchNorm1d(feature_dim)
        self.bnneck.bias.requires_grad_(False)

        # 分类头
        # 这里只是最简单的线性分类器，不加 bias 是较常见选择
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> dict:
        """
        前向传播

        输入：
            x: [B, T, 1, H, W]

        输出：
            dict，包含 embedding / bn_embedding / logits

        详细步骤：
        1. 检查输入维度
        2. 合并 batch 和时间维，对每帧提特征
        3. 恢复时间维
        4. 在时间维上做最大池化，得到序列级特征图
        5. 做水平分块池化，得到全局序列特征
        6. 用 FC 映射到固定维度
        7. BNNeck + 分类头
        """
        if x.ndim != 5:
            raise ValueError(
                f"输入张量维度应为 5，当前为 {x.ndim}，期望形状 [B, T, 1, H, W]"
            )

        b, t, c, h, w = x.shape

        # -------------------------------------------------
        # 第一步：把 [B, T, C, H, W] 展平成 [B*T, C, H, W]
        # -------------------------------------------------
        # 这样就可以把每一帧当作独立图像送进 2D CNN
        x = x.view(b * t, c, h, w)  # [B*T, 1, H, W]

        # -------------------------------------------------
        # 第二步：提取帧级空间特征
        # -------------------------------------------------
        frame_feat = self.backbone(x)  # [B*T, 256, H', W']

        # -------------------------------------------------
        # 第三步：恢复时间维
        # -------------------------------------------------
        _, c2, h2, w2 = frame_feat.shape
        frame_feat = frame_feat.view(b, t, c2, h2, w2)  # [B, T, 256, H', W']

        # -------------------------------------------------
        # 第四步：时间维最大池化（Temporal Max Pooling）
        # -------------------------------------------------
        # 这一步的含义：
        # - 对整段序列，在时间维上取最强响应
        # - 相当于聚合整段步态序列中的显著运动模式
        #
        # 为什么用 max：
        # - 简单
        # - 稳定
        # - 是很多步态方法里的常见基线操作
        #
        # 这里输出：
        #   seq_feat: [B, 256, H', W']
        seq_feat, _ = torch.max(frame_feat, dim=1)

        # -------------------------------------------------
        # 第五步：水平分块池化
        # -------------------------------------------------
        # 输出维度：[B, 256*num_bins]
        pooled_feat = self.hpm(seq_feat)

        # -------------------------------------------------
        # 第六步：映射到 embedding 空间
        # -------------------------------------------------
        # 这就是后面真正要拿去做融合的剪影特征
        embedding = self.fc(pooled_feat)   # [B, feature_dim]
        embedding = self.dropout(embedding)

        # -------------------------------------------------
        # 第七步：BNNeck + 分类头
        # -------------------------------------------------
        bn_embedding = self.bnneck(embedding)
        logits = self.classifier(bn_embedding)

        return {
            # 这个特征最重要：
            # 后面做 triplet loss、融合模型，都建议优先用它
            "embedding": embedding,

            # 这个特征一般给分类器使用
            "bn_embedding": bn_embedding,

            # 分类输出，用于交叉熵损失
            "logits": logits
        }


# =========================================================
# 5. 简单自测
# =========================================================
if __name__ == "__main__":
    """
    自测目的：
    1. 检查网络能否正常实例化
    2. 检查输入输出维度是否符合预期

    这里构造一个随机输入：
        B=2
        T=30
        C=1
        H=64
        W=44
    这和你当前剪影序列的目标输入格式一致。
    """
    model = SilhouetteBranch(
        num_classes=100,   # 假设有 100 个身份类别
        in_channels=1,
        feature_dim=256,
        num_bins=4,
        dropout=0.1
    )

    x = torch.randn(2, 30, 1, 64, 44)  # [B, T, C, H, W]
    out = model(x)

    print("embedding:", out["embedding"].shape)        # [2, 256]
    print("bn_embedding:", out["bn_embedding"].shape)  # [2, 256]
    print("logits:", out["logits"].shape)              # [2, 100]