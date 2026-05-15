# -*- coding: utf-8 -*-
"""
fusion_model_residual.py

剪影主导 + 骨架残差修正 融合模型
---------------------------------------------------------
设计目标：
- 你现在的实验里，剪影单分支最强，骨架单分支略弱
- 直接拼接 / 门控 / attention 都没有稳定超过剪影单分支
- 因此这版模型不再追求“两个模态平权融合”
- 而是明确采用“剪影主导，骨架修正”的思路

核心思想：
    silhouette embedding 作为主特征
    skeleton embedding    作为补充修正特征

公式直观写法：
    s = phi_s(e_s)
    k = phi_k(e_k)

    g = sigmoid(MLP([s, k, |s-k|]))
    c = max(softmax(y_k))

    delta = psi(k)
    f = s + c * g * delta

其中：
- s     : 剪影主特征
- k     : 骨架投影特征
- g     : 逐维门控，控制骨架修正的注入强度
- c     : 骨架置信度，若骨架分支自己都不自信，就少修正
- delta : 骨架提供的残差修正量
- f     : 最终融合特征

兼容性：
- 保留 self.silhouette_branch / self.skeleton_branch
- 保留 set_branch_trainable(...)
- forward 返回字段尽量和原 fusion_model.py 保持兼容：
    silhouette_embedding
    skeleton_embedding
    fused_embedding
    bn_embedding
    logits
    silhouette_logits
    skeleton_logits
- 额外返回：
    skeleton_confidence
    fusion_gate
    residual_delta

如何接入现有训练脚本：
1. 最直接：
   把这个文件重命名为 fusion_model.py 覆盖原文件
2. 更稳妥：
   保留文件名 fusion_model_residual.py
   然后把 train_fusion_condition_split.py 里的
       from fusion_model import GaitFusionModel
   改成
       from fusion_model_residual import GaitFusionModel
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from silhouette_branch import SilhouetteBranch
from skeleton_branch import SkeletonBranch


# =========================================================
# 1. 残差融合头
# =========================================================
class SilhouetteDominantResidualFusion(nn.Module):
    """
    剪影主导 + 骨架残差修正 融合头

    输入：
        sil_feat: [B, D_s]
        ske_feat: [B, D_k]
        ske_logits: [B, C] 或 None

    输出：
        fused_feat: [B, D]
        gate: [B, D]
        delta: [B, D]
        skeleton_conf: [B, 1]
    """

    def __init__(
        self,
        in_dim_sil: int,
        in_dim_ske: int,
        fusion_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 剪影主分支投影：尽量保留为“干净主干”
        self.sil_proj = nn.Sequential(
            nn.Linear(in_dim_sil, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # 骨架分支投影：作为补充信息
        self.ske_proj = nn.Sequential(
            nn.Linear(in_dim_ske, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # 骨架残差修正量
        self.delta_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )

        # 门控：学习“哪些维度应该让骨架来修正剪影”
        gate_in_dim = fusion_dim * 3  # [s, k, |s-k|]
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid()
        )

        # 输出 refinement，避免纯线性相加后表达过于粗糙
        self.refine = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def _estimate_skeleton_confidence(self, ske_logits: torch.Tensor | None, device: torch.device, batch_size: int):
        """
        根据 skeleton branch 的分类输出估计一个 [B,1] 的骨架置信度。
        若没有 skeleton_logits，则退化为全 1。
        """
        if ske_logits is None:
            return torch.ones(batch_size, 1, device=device)

        # softmax 后取最大类别概率，作为骨架当前样本的自信度
        prob = torch.softmax(ske_logits, dim=1)
        conf = torch.max(prob, dim=1, keepdim=True)[0]  # [B,1]
        return conf

    def forward(
        self,
        sil_feat: torch.Tensor,
        ske_feat: torch.Tensor,
        ske_logits: torch.Tensor | None = None
    ):
        batch_size = sil_feat.size(0)
        device = sil_feat.device

        s = self.sil_proj(sil_feat)  # [B, D]
        k = self.ske_proj(ske_feat)  # [B, D]

        joint_feat = torch.cat([s, k, torch.abs(s - k)], dim=1)
        gate = self.gate_mlp(joint_feat)  # [B, D]

        delta = self.delta_proj(k)        # [B, D]
        skeleton_conf = self._estimate_skeleton_confidence(
            ske_logits=ske_logits,
            device=device,
            batch_size=batch_size
        )  # [B,1]

        # residual fusion:
        # 主干保持为 silhouette
        # 骨架只作为修正项参与
        fused = s + skeleton_conf * gate * delta
        fused = self.refine(fused)

        return fused, gate, delta, skeleton_conf


# =========================================================
# 2. 整体融合模型
# =========================================================
class GaitFusionModel(nn.Module):
    """
    剪影主导残差融合模型
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

        self.fusion_head = SilhouetteDominantResidualFusion(
            in_dim_sil=silhouette_feature_dim,
            in_dim_ske=skeleton_feature_dim,
            fusion_dim=fusion_dim,
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
        """
        兼容分支 forward 返回 dict / tensor 两种情况
        """
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

        elif torch.is_tensor(out):
            return {
                "embedding": out,
                "bn_embedding": out,
                "logits": None
            }

        else:
            raise TypeError(f"{name} branch 的输出类型不受支持：{type(out)}")

    def forward(self, x_silhouette: torch.Tensor, x_skeleton: torch.Tensor) -> Dict[str, torch.Tensor]:
        sil_out_raw = self.silhouette_branch(x_silhouette)
        ske_out_raw = self.skeleton_branch(x_skeleton)

        sil_out = self._extract_branch_outputs(sil_out_raw, name="silhouette")
        ske_out = self._extract_branch_outputs(ske_out_raw, name="skeleton")

        sil_emb = sil_out["embedding"]
        ske_emb = ske_out["embedding"]
        ske_logits = ske_out["logits"]

        fused_emb, gate, delta, skeleton_conf = self.fusion_head(
            sil_feat=sil_emb,
            ske_feat=ske_emb,
            ske_logits=ske_logits
        )

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
            "fusion_gate": gate,
            "residual_delta": delta,
            "skeleton_confidence": skeleton_conf
        }


# =========================================================
# 3. 自测
# =========================================================
if __name__ == "__main__":
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
    print("residual_delta:", out["residual_delta"].shape)
    print("skeleton_confidence:", out["skeleton_confidence"].shape)
