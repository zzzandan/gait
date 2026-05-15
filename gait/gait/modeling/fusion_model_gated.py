# -*- coding: utf-8 -*-
"""
fusion_model_gated.py

一个“可直接替换现有 fusion_model.py 使用”的备选融合模型。
核心思想：不用最基础的直接拼接，而改成“门控融合（Gated Fusion）”。

=========================================================
一、这个模型和你原来的融合模型有什么区别
=========================================================
原来的基础版融合通常是：
    1. 剪影分支提特征
    2. 骨架分支提特征
    3. 直接 concat
    4. 过一层/几层全连接得到最终分类结果

这种方法简单、稳定，但问题是：
- 它默认两个模态同等重要
- 没有显式告诉网络：什么时候更信剪影，什么时候更信骨架
- 当一个模态噪声较大时，直接拼接不一定能自动抑制噪声

因此，这里给你一个“门控融合版”：
- 先把 silhouette / skeleton 的 embedding 映射到同一维度
- 再根据两个模态的联合特征，学习一个 gate
- gate 的每一维都在 [0,1] 内
- 最终：
      fused = gate * sil_proj + (1 - gate) * ske_proj

这样模型就能自己学：
- 哪些维度更应该相信剪影
- 哪些维度更应该相信骨架

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
- 把这个文件命名为 fusion_model_gated.py
- 然后把 train_fusion_condition_split.py 里的：
      from fusion_model import GaitFusionModel
  改成：
      from fusion_model_gated import GaitFusionModel

=========================================================
四、注意
=========================================================
这个“别的模型”并不保证一定比原来的拼接融合更强，
但它比简单拼接更有“融合味道”，也更适合写进论文中做对比实验。
"""

from __future__ import annotations

from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from silhouette_branch import SilhouetteBranch
from skeleton_branch import SkeletonBranch


# =========================================================
# 1. 门控融合头
# =========================================================
class GatedFusionHead(nn.Module):
    """
    门控融合模块。

    输入：
        sil_feat: [B, D]
        ske_feat: [B, D]

    处理思路：
        1. 先把两个模态映射到同一维度
        2. 再构造联合特征：
              [sil, ske, |sil-ske|, sil*ske]
        3. 用一个小 MLP 生成 gate
        4. 用 gate 做逐维加权融合

    输出：
        fused_feat: [B, D]
        gate:       [B, D]
    """

    def __init__(self, in_dim_sil: int, in_dim_ske: int, fusion_dim: int = 256, dropout: float = 0.1):
        super().__init__()

        # 先把两个模态映射到统一维度
        self.sil_proj = nn.Sequential(
            nn.Linear(in_dim_sil, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.ske_proj = nn.Sequential(
            nn.Linear(in_dim_ske, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # 联合特征维度：4 * fusion_dim
        gate_in_dim = fusion_dim * 4

        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid()   # gate 每一维限制在 [0,1]
        )

        # 再接一个 refinement 层，让融合后的特征再做一次非线性变换
        self.refine = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, sil_feat: torch.Tensor, ske_feat: torch.Tensor):
        sil_proj = self.sil_proj(sil_feat)   # [B, D]
        ske_proj = self.ske_proj(ske_feat)   # [B, D]

        joint_feat = torch.cat(
            [
                sil_proj,
                ske_proj,
                torch.abs(sil_proj - ske_proj),
                sil_proj * ske_proj
            ],
            dim=1
        )  # [B, 4D]

        gate = self.gate_mlp(joint_feat)     # [B, D]

        fused = gate * sil_proj + (1.0 - gate) * ske_proj
        fused = self.refine(fused)

        return fused, gate


# =========================================================
# 2. 整体融合模型
# =========================================================
class GaitFusionModel(nn.Module):
    """
    备选融合模型：Gated Fusion 版

    结构：
        silhouette_branch -> silhouette embedding
        skeleton_branch   -> skeleton embedding
                           ↓
                     gated fusion
                           ↓
                       classifier
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

        # -------------------------------------------------
        # 两个单分支
        # -------------------------------------------------
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

        # -------------------------------------------------
        # 门控融合头
        # -------------------------------------------------
        self.fusion_head = GatedFusionHead(
            in_dim_sil=silhouette_feature_dim,
            in_dim_ske=skeleton_feature_dim,
            fusion_dim=fusion_dim,
            dropout=dropout
        )

        # 融合后的 BN + 分类器
        self.bn = nn.BatchNorm1d(fusion_dim)
        self.classifier = nn.Linear(fusion_dim, num_classes)

    # -----------------------------------------------------
    # 兼容训练脚本：冻结 / 解冻分支
    # -----------------------------------------------------
    def set_branch_trainable(self, silhouette_trainable: bool = True, skeleton_trainable: bool = True) -> None:
        for p in self.silhouette_branch.parameters():
            p.requires_grad = silhouette_trainable

        for p in self.skeleton_branch.parameters():
            p.requires_grad = skeleton_trainable

    # -----------------------------------------------------
    # 兼容不同 branch 返回格式的辅助函数
    # -----------------------------------------------------
    def _extract_branch_outputs(self, out: Any, name: str) -> Dict[str, torch.Tensor]:
        """
        统一提取 branch 的 embedding / logits。

        兼容两类情况：
        1. branch.forward 返回 dict
        2. branch.forward 直接返回 tensor（极少见，但这里做防御性处理）
        """
        if isinstance(out, dict):
            # 常见情况：out["embedding"], out["logits"]
            if "embedding" in out:
                embedding = out["embedding"]
            elif "bn_embedding" in out:
                # 如果某个分支只给了 bn_embedding，就退化使用它
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

        elif torch.is_tensor(out):
            return {
                "embedding": out,
                "bn_embedding": out,
                "logits": None
            }

        else:
            raise TypeError(f"{name} branch 的输出类型不受支持：{type(out)}")

    # -----------------------------------------------------
    # 前向传播
    # -----------------------------------------------------
    def forward(self, x_silhouette: torch.Tensor, x_skeleton: torch.Tensor) -> Dict[str, torch.Tensor]:
        sil_out_raw = self.silhouette_branch(x_silhouette)
        ske_out_raw = self.skeleton_branch(x_skeleton)

        sil_out = self._extract_branch_outputs(sil_out_raw, name="silhouette")
        ske_out = self._extract_branch_outputs(ske_out_raw, name="skeleton")

        sil_emb = sil_out["embedding"]   # [B, D1]
        ske_emb = ske_out["embedding"]   # [B, D2]

        fused_emb, gate = self.fusion_head(sil_emb, ske_emb)   # [B, D]
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
            "fusion_gate": gate
        }


# =========================================================
# 3. 自测代码
# =========================================================
if __name__ == "__main__":
    # 这里只是做 shape 级别的快速自测
    model = GaitFusionModel(
        num_classes=11,
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

    if out["silhouette_logits"] is not None:
        print("silhouette_logits:", out["silhouette_logits"].shape)
    if out["skeleton_logits"] is not None:
        print("skeleton_logits:", out["skeleton_logits"].shape)

    print("fusion_gate:", out["fusion_gate"].shape)
