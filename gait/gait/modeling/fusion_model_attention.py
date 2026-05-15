# -*- coding: utf-8 -*-
"""
fusion_model_attention.py

一个“可直接替换现有 fusion_model.py 使用”的备选融合模型。
核心思想：不用最基础的直接拼接，而改成“注意力融合（Attention Fusion）”。

=========================================================
一、这个模型和你原来的融合模型有什么区别
=========================================================
原来的基础版融合通常是：
    1. 剪影分支提特征
    2. 骨架分支提特征
    3. 直接 concat
    4. 过一层/几层全连接得到最终分类结果

这类方法简单，但缺点是：
- 两个模态之间没有显式交互
- 没有建模“谁该关注谁”
- 对互补信息的利用比较有限

这里给你一版“注意力融合”：
- 先把 silhouette / skeleton 的 embedding 映射到同一维度
- 再把两个模态看作长度为 2 的 token 序列
- 用 Multi-Head Self-Attention 做模态间交互
- 再做残差 + 前馈网络
- 最后把两个 token 聚合成一个 fused embedding

这样模型就能学到：
- 剪影该从骨架里关注什么
- 骨架该从剪影里关注什么
- 哪些维度在两种模态之间是互补的

=========================================================
二、兼容性说明
=========================================================
这份代码尽量保持和你之前 fusion_model.py 相同的使用习惯：

1. 仍然包含：
   - self.silhouette_branch
   - self.skeleton_branch
   因此可以继续加载你已有的单分支 best.pth

2. forward 返回字段尽量与之前保持一致：
   - silhouette_embedding
   - skeleton_embedding
   - fused_embedding
   - bn_embedding
   - logits
   - silhouette_logits
   - skeleton_logits

3. 仍然提供：
   - set_branch_trainable(...)
   方便冻结 / 解冻分支

=========================================================
三、如何接入你现有训练脚本
=========================================================
方法 1：最直接
- 把这个文件保存为 fusion_model.py
- 覆盖原来的 fusion_model.py
- 然后你原来的 train_fusion_condition_split.py 就不用改 import

方法 2：保留原文件，新增一个新文件
- 把这个文件命名为 fusion_model_attention.py
- 然后把 train_fusion_condition_split.py 里的：
      from fusion_model import GaitFusionModel
  改成：
      from fusion_model_attention import GaitFusionModel

=========================================================
四、注意
=========================================================
这个“attention 融合版”并不保证一定比基础拼接更强，
但它比简单拼接更像真正的跨模态融合，也适合作为论文中的对比实验。
"""

from __future__ import annotations

from typing import Dict, Any

import torch
import torch.nn as nn

from silhouette_branch import SilhouetteBranch
from skeleton_branch import SkeletonBranch


class CrossModalAttentionFusion(nn.Module):
    """
    注意力融合模块。

    输入：
        sil_feat: [B, D1]
        ske_feat: [B, D2]

    处理思路：
        1. 先把两个模态投影到统一维度 fusion_dim
        2. 把两个模态堆叠成 token 序列，长度为 2
              tokens = [sil_token, ske_token]
        3. 用多头自注意力建模两个模态之间的交互
        4. 经过残差、LayerNorm、前馈网络
        5. 对两个 token 做平均池化，得到 fused embedding

    输出：
        fused_feat: [B, D]
        tokens_out: [B, 2, D]
        attn_weights: 注意力权重
    """

    def __init__(
        self,
        in_dim_sil: int,
        in_dim_ske: int,
        fusion_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.fusion_dim = fusion_dim

        self.sil_proj = nn.Sequential(
            nn.Linear(in_dim_sil, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.ske_proj = nn.Sequential(
            nn.Linear(in_dim_ske, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.modality_embed = nn.Parameter(torch.randn(1, 2, fusion_dim) * 0.02)

        self.attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(fusion_dim)

        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Dropout(dropout)
        )

        self.norm2 = nn.LayerNorm(fusion_dim)

        self.out_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, sil_feat: torch.Tensor, ske_feat: torch.Tensor):
        sil_token = self.sil_proj(sil_feat)
        ske_token = self.ske_proj(ske_feat)

        tokens = torch.stack([sil_token, ske_token], dim=1)
        tokens = tokens + self.modality_embed

        attn_out, attn_weights = self.attn(
            query=tokens,
            key=tokens,
            value=tokens,
            need_weights=True,
            average_attn_weights=False
        )

        tokens = self.norm1(tokens + attn_out)

        ffn_out = self.ffn(tokens)
        tokens = self.norm2(tokens + ffn_out)

        fused = tokens.mean(dim=1)
        fused = self.out_proj(fused)

        return fused, tokens, attn_weights


class GaitFusionModel(nn.Module):
    """
    备选融合模型：Attention Fusion 版
    """

    def __init__(
        self,
        num_classes: int,
        silhouette_feature_dim: int = 256,
        skeleton_feature_dim: int = 256,
        fusion_dim: int = 256,
        silhouette_num_bins: int = 4,
        dropout: float = 0.1,
        num_heads: int = 4
    ):
        super().__init__()

        self.silhouette_branch = SilhouetteBranch(
            num_classes=num_classes,
            in_channels=1,
            feature_dim=silhouette_feature_dim,
            num_bins=silhouette_num_bins,
            dropout=dropout
        )

        self.skeleton_branch = SkeletonBranch(
            num_classes=num_classes,
            in_channels=2,
            feature_dim=skeleton_feature_dim,
            dropout=dropout
        )

        self.fusion_head = CrossModalAttentionFusion(
            in_dim_sil=silhouette_feature_dim,
            in_dim_ske=skeleton_feature_dim,
            fusion_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.bn = nn.BatchNorm1d(fusion_dim)
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def set_branch_trainable(self, silhouette_trainable: bool = True, skeleton_trainable: bool = True) -> None:
        for p in self.silhouette_branch.parameters():
            p.requires_grad = silhouette_trainable
        for p in self.skeleton_branch.parameters():
            p.requires_grad = skeleton_trainable

    def _extract_branch_outputs(self, out: Any, name: str) -> Dict[str, torch.Tensor]:
        if isinstance(out, dict):
            if "embedding" in out:
                embedding = out["embedding"]
            elif "bn_embedding" in out:
                embedding = out["bn_embedding"]
            else:
                raise KeyError(f"{name} branch 输出中找不到 embedding / bn_embedding 字段。")

            logits = out.get("logits", None)
            bn_embedding = out.get("bn_embedding", embedding)

            return {
                "embedding": embedding,
                "bn_embedding": bn_embedding,
                "logits": logits
            }

        if torch.is_tensor(out):
            return {
                "embedding": out,
                "bn_embedding": out,
                "logits": None
            }

        raise TypeError(f"{name} branch 的输出类型不受支持：{type(out)}")

    def forward(self, x_silhouette: torch.Tensor, x_skeleton: torch.Tensor) -> Dict[str, torch.Tensor]:
        sil_out_raw = self.silhouette_branch(x_silhouette)
        ske_out_raw = self.skeleton_branch(x_skeleton)

        sil_out = self._extract_branch_outputs(sil_out_raw, name="silhouette")
        ske_out = self._extract_branch_outputs(ske_out_raw, name="skeleton")

        sil_emb = sil_out["embedding"]
        ske_emb = ske_out["embedding"]

        fused_emb, token_embeddings, attn_weights = self.fusion_head(sil_emb, ske_emb)
        bn_embedding = self.bn(fused_emb)
        logits = self.classifier(bn_embedding)

        return {
            "silhouette_embedding": sil_emb,
            "skeleton_embedding": ske_emb,
            "fused_embedding": fused_emb,
            "bn_embedding": bn_embedding,
            "logits": logits,
            "silhouette_logits": sil_out["logits"],
            "skeleton_logits": ske_out["logits"],
            "fusion_tokens": token_embeddings,
            "fusion_attention": attn_weights
        }


if __name__ == "__main__":
    model = GaitFusionModel(
        num_classes=11,
        silhouette_feature_dim=256,
        skeleton_feature_dim=256,
        fusion_dim=256,
        silhouette_num_bins=4,
        dropout=0.1,
        num_heads=4
    )

    x_silhouette = torch.randn(2, 30, 1, 64, 44)
    x_skeleton = torch.randn(2, 30, 2, 64, 44)

    out = model(x_silhouette, x_skeleton)

    print("silhouette_embedding:", out["silhouette_embedding"].shape)
    print("skeleton_embedding:", out["skeleton_embedding"].shape)
    print("fused_embedding:", out["fused_embedding"].shape)
    print("bn_embedding:", out["bn_embedding"].shape)
    print("logits:", out["logits"].shape)

    if out["silhouette_logits"] is not None:
        print("silhouette_logits:", out["silhouette_logits"].shape)
    if out["skeleton_logits"] is not None:
        print("skeleton_logits:", out["skeleton_logits"].shape)

    print("fusion_tokens:", out["fusion_tokens"].shape)

    attn = out["fusion_attention"]
    if attn is not None:
        print("fusion_attention:", attn.shape)
