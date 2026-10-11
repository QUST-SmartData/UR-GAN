# UR-GAN: Feature-Adaptive Transfer Module (FATM) — 权重对齐阶段（特征空间自适应融合）
# 对应论文 3.2：在选定分辨率层对基准路径与参考路径的特征做通道拼接与空间门控融合。

from __future__ import annotations

import copy
from typing import List, Optional, Sequence

import torch
from torch_utils import misc
from torch_utils import persistence

from training.networks import Conv2dLayer, SynthesisNetwork


def default_fatm_resolutions(block_resolutions: Sequence[int]) -> List[int]:
    """与论文一致：低分辨率若干层不融合，最后一层输出不融合，中间层引入 FATM。"""
    br = list(block_resolutions)
    if len(br) <= 4:
        return br[2:-1] if len(br) > 3 else []
    return br[3:-1]


@persistence.persistent_class
class FeatureAdaptiveTransferModule(torch.nn.Module):
    """对两路同分辨率特征做通道拼接，经卷积预测空间门控，再逐像素加权融合。"""

    def __init__(self, channels: int, hidden_ratio: int = 2):
        super().__init__()
        hidden = max(channels * hidden_ratio // 2, 32)
        self.conv = torch.nn.Sequential(
            Conv2dLayer(in_channels=channels * 2, out_channels=hidden, kernel_size=3, activation='lrelu'),
            Conv2dLayer(in_channels=hidden, out_channels=1, kernel_size=3, activation='linear'),
        )

    def forward(self, f_a: torch.Tensor, f_b: torch.Tensor) -> torch.Tensor:
        x = torch.cat([f_a, f_b], dim=1)
        gate = torch.sigmoid(self.conv(x))
        return gate * f_a + (1.0 - gate) * f_b


@persistence.persistent_class
class DualSynthesisNetworkFATM(torch.nn.Module):
    """在单套 ws 下并行两条合成路径，于指定分辨率对主路径特征与参考路径特征做 FATM 融合。"""

    def __init__(
        self,
        synthesis_trainable: SynthesisNetwork,
        synthesis_ref_frozen: SynthesisNetwork,
        fatm_resolutions: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        assert synthesis_trainable.block_resolutions == synthesis_ref_frozen.block_resolutions
        assert synthesis_trainable.num_ws == synthesis_ref_frozen.num_ws
        self.syn_a = synthesis_trainable
        self.syn_b = synthesis_ref_frozen
        self.block_resolutions = list(synthesis_trainable.block_resolutions)
        self.num_ws = synthesis_trainable.num_ws
        self.w_dim = synthesis_trainable.w_dim
        self.img_resolution = synthesis_trainable.img_resolution
        self.img_channels = synthesis_trainable.img_channels

        if fatm_resolutions is None:
            fatm_resolutions = default_fatm_resolutions(self.block_resolutions)
        self.fatm_res_set = set(int(r) for r in fatm_resolutions)

        self.fatm = torch.nn.ModuleDict()
        for res in self.fatm_res_set:
            if res not in self.block_resolutions:
                continue
            block = getattr(self.syn_a, f'b{res}')
            c_out = int(block.conv1.weight.shape[0])
            self.fatm[str(res)] = FeatureAdaptiveTransferModule(c_out)

        for p in self.syn_b.parameters():
            p.requires_grad = False
        self.syn_b.eval()

    def forward(self, ws, **block_kwargs):
        block_ws: List[torch.Tensor] = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self.syn_a, f'b{res}')
                n = block.num_conv + block.num_torgb
                block_ws.append(ws.narrow(1, w_idx, n))
                w_idx += block.num_conv

        x_a = img_a = None
        x_b = img_b = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block_a = getattr(self.syn_a, f'b{res}')
            block_b = getattr(self.syn_b, f'b{res}')
            x_a, img_a = block_a(x_a, img_a, cur_ws, **block_kwargs)
            with torch.no_grad():
                x_b, img_b = block_b(x_b, img_b, cur_ws, **block_kwargs)

            if res in self.fatm_res_set and str(res) in self.fatm:
                x_a = self.fatm[str(res)](x_a, x_b.detach())

        return img_a


def patch_generator_with_fatm(G, ref_generator_b, fatm_resolutions=None, device=None):
    """将已有 Generator 的 synthesis 替换为 DualSynthesisNetworkFATM（参考路径来自第二个模型）。"""
    syn_b = copy.deepcopy(ref_generator_b.synthesis).eval().requires_grad_(False)
    if device is not None:
        syn_b = syn_b.to(device)
    G.synthesis = DualSynthesisNetworkFATM(G.synthesis, syn_b, fatm_resolutions=fatm_resolutions)
    if device is not None:
        G.synthesis.to(device)
    return G
