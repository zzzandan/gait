# -*- coding: utf-8 -*-
"""
fusion_model.py

多特征融合步态识别模型

当前版本采用最简单、最稳妥的“特征层拼接融合”：
1. 剪影序列 -> SilhouetteBranch -> silhouette_embedding
2. 骨架序列 -> SkeletonBranch -> skeleton_embedding
3. 两个 embedding 在特征维度拼接
4. 经过融合头（FC + BNNeck + Classifier）
5. 输出最终融合分类结果

这样做的好处是：
- 结构清晰
- 容易训练
- 容易写进论文
- 便于后续扩展成注意力融合、门控融合等版本
"""

from __future__ import annotations

import torch
import torch.nn as nn

from silhouette_branch import SilhouetteBranch
from skeleton_branch import SkeletonBranch


# =========================================================
# 1. 融合头模块
# =========================================================
class FusionHead(nn.Module):
    """
    融合头模块

    输入：
        silhouette_embedding: [B, silhouette_dim]
        skeleton_embedding:   [B, skeleton_dim]

    处理流程：
        拼接 -> FC -> Dropout -> BNNeck -> Classifier

    输出：
        {
            "fused_embedding": [B, fusion_dim],
            "bn_embedding":    [B, fusion_dim],
            "logits":          [B, num_classes]
        }
    """

    def __init__(
        self,
        silhouette_dim: int,
        skeleton_dim: int,
        fusion_dim: int,
        num_classes: int,
        dropout: float = 0.1
    ):
        super().__init__()

        fusion_in_dim = silhouette_dim + skeleton_dim

        self.fc = nn.Linear(fusion_in_dim, fusion_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # BNNeck 风格，和你前面的单分支保持一致
        self.bnneck = nn.BatchNorm1d(fusion_dim)
        self.bnneck.bias.requires_grad_(False)

        self.classifier = nn.Linear(fusion_dim, num_classes, bias=False)

    def forward(
        self,
        silhouette_embedding: torch.Tensor,
        skeleton_embedding: torch.Tensor
    ) -> dict:
        if silhouette_embedding.ndim != 2:
            raise ValueError(
                f"silhouette_embedding 维度应为 2，当前为 {silhouette_embedding.ndim}"
            )

        if skeleton_embedding.ndim != 2:
            raise ValueError(
                f"skeleton_embedding 维度应为 2，当前为 {skeleton_embedding.ndim}"
            )

        if silhouette_embedding.shape[0] != skeleton_embedding.shape[0]:
            raise ValueError(
                "剪影 embedding 和骨架 embedding 的 batch 大小不一致，"
                f"分别为 {silhouette_embedding.shape[0]} 和 {skeleton_embedding.shape[0]}"
            )

        # 特征拼接
        fused = torch.cat([silhouette_embedding, skeleton_embedding], dim=1)

        # 融合映射
        fused_embedding = self.fc(fused)
        fused_embedding = self.dropout(fused_embedding)

        # BNNeck + 分类器
        bn_embedding = self.bnneck(fused_embedding)
        logits = self.classifier(bn_embedding)

        return {
            "fused_embedding": fused_embedding,
            "bn_embedding": bn_embedding,
            "logits": logits
        }


# =========================================================
# 2. 多特征融合主模型
# =========================================================
class GaitFusionModel(nn.Module):
    """
    多特征融合步态识别模型

    输入：
        x_silhouette: [B, T, 1, H, W]
        x_skeleton:   [B, T, 2, H, W]

    输出：
        {
            "silhouette_embedding": [B, silhouette_feature_dim],
            "skeleton_embedding":   [B, skeleton_feature_dim],
            "fused_embedding":      [B, fusion_dim],
            "bn_embedding":         [B, fusion_dim],
            "logits":               [B, num_classes],
            "silhouette_logits":    [B, num_classes],
            "skeleton_logits":      [B, num_classes]
        }
    """

    def __init__(
        self,
        num_classes: int,
        silhouette_feature_dim: int = 256,
        skeleton_feature_dim: int = 256,
        fusion_dim: int = 256,
        silhouette_num_bins: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        # 剪影分支
        self.silhouette_branch = SilhouetteBranch(
            num_classes=num_classes,
            in_channels=1,
            feature_dim=silhouette_feature_dim,
            num_bins=silhouette_num_bins,
            dropout=dropout
        )

        # 骨架分支
        self.skeleton_branch = SkeletonBranch(
            num_classes=num_classes,
            in_channels=2,
            feature_dim=skeleton_feature_dim,
            dropout=dropout
        )

        # 融合头
        self.fusion_head = FusionHead(
            silhouette_dim=silhouette_feature_dim,
            skeleton_dim=skeleton_feature_dim,
            fusion_dim=fusion_dim,
            num_classes=num_classes,
            dropout=dropout
        )

    def set_branch_trainable(
        self,
        silhouette_trainable: bool = True,
        skeleton_trainable: bool = True
    ) -> None:
        """
        控制两个单分支是否参与训练。

        用途：
        - 如果后面你想加载单分支预训练权重，然后冻结两个 branch，
          只训练 fusion head，就可以用这个函数
        """
        for p in self.silhouette_branch.parameters():
            p.requires_grad = silhouette_trainable

        for p in self.skeleton_branch.parameters():
            p.requires_grad = skeleton_trainable

    def forward(
        self,
        x_silhouette: torch.Tensor,
        x_skeleton: torch.Tensor
    ) -> dict:
        if x_silhouette.ndim != 5:
            raise ValueError(
                f"x_silhouette 维度应为 5，当前为 {x_silhouette.ndim}，"
                "期望形状 [B, T, 1, H, W]"
            )

        if x_skeleton.ndim != 5:
            raise ValueError(
                f"x_skeleton 维度应为 5，当前为 {x_skeleton.ndim}，"
                "期望形状 [B, T, 2, H, W]"
            )

        if x_silhouette.shape[0] != x_skeleton.shape[0]:
            raise ValueError(
                "剪影输入和骨架输入的 batch 大小不一致，"
                f"分别为 {x_silhouette.shape[0]} 和 {x_skeleton.shape[0]}"
            )

        # 两个单分支各自前向
        sil_out = self.silhouette_branch(x_silhouette)
        ske_out = self.skeleton_branch(x_skeleton)

        silhouette_embedding = sil_out["embedding"]
        skeleton_embedding = ske_out["embedding"]

        # 融合输出
        fusion_out = self.fusion_head(
            silhouette_embedding=silhouette_embedding,
            skeleton_embedding=skeleton_embedding
        )

        return {
            "silhouette_embedding": silhouette_embedding,
            "skeleton_embedding": skeleton_embedding,
            "fused_embedding": fusion_out["fused_embedding"],
            "bn_embedding": fusion_out["bn_embedding"],
            "logits": fusion_out["logits"],

            # 保留单分支 logits，便于后续做辅助监督
            "silhouette_logits": sil_out["logits"],
            "skeleton_logits": ske_out["logits"]
        }


# =========================================================
# 3. 简单自测
# =========================================================
if __name__ == "__main__":
    model = GaitFusionModel(
        num_classes=4,
        silhouette_feature_dim=256,
        skeleton_feature_dim=256,
        fusion_dim=256,
        silhouette_num_bins=4,
        dropout=0.1
    )

    x_silhouette = torch.randn(2, 30, 1, 64, 44)
    x_skeleton = torch.randn(2, 30, 2, 64, 44)

    out = model(x_silhouette, x_skeleton)

    print("silhouette_embedding:", out["silhouette_embedding"].shape)
    print("skeleton_embedding:", out["skeleton_embedding"].shape)
    print("fused_embedding:", out["fused_embedding"].shape)
    print("bn_embedding:", out["bn_embedding"].shape)
    print("logits:", out["logits"].shape)
    print("silhouette_logits:", out["silhouette_logits"].shape)
    print("skeleton_logits:", out["skeleton_logits"].shape)
