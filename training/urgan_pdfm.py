# UR-GAN: Parameter-Level Dynamic Fusion (PDFM) — 权重融合阶段（由 z、c 生成逐层融合系数）
# 对应论文 3.3：条件编码后生成系数，对各层激活做逐层凸组合（参考路径冻结，系数网络可训练）。

from __future__ import annotations

import copy
from typing import List, Optional

import torch
import torch.nn as nn
from torch_utils import misc
from torch_utils import persistence

from training.networks import Generator, SynthesisNetwork


def _average_mapping_into(dst_gen: Generator, sources: List[Generator]) -> None:
    """将多个 Generator 的 mapping 参数均值写入 dst_gen.mapping。"""
    def pairs_to_dict(pairs_or_dict):
        if isinstance(pairs_or_dict, dict):
            return pairs_or_dict
        return {k: v for k, v in pairs_or_dict}

    from torch_utils import misc as misc_mod
    dst_params = pairs_to_dict(misc_mod.named_params_and_buffers(dst_gen.mapping))
    arrs = [pairs_to_dict(misc_mod.named_params_and_buffers(s.mapping)) for s in sources]
    for name, tensor in dst_params.items():
        stack = []
        for a in arrs:
            t = a.get(name, None)
            if t is not None and tuple(t.shape) == tuple(tensor.shape):
                stack.append(t.detach())
        if stack:
            with torch.no_grad():
                tensor.copy_(sum(stack) / float(len(stack)))


@persistence.persistent_class
class LayerBlendSynthesis(torch.nn.Module):
    """两套冻结 synthesis 在每一尺度上用 α_l(z,c) 做特征与 RGB 累积的凸组合。"""

    def __init__(
        self,
        synthesis_a: SynthesisNetwork,
        synthesis_b: SynthesisNetwork,
        z_dim: int,
        c_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        assert synthesis_a.block_resolutions == synthesis_b.block_resolutions
        assert synthesis_a.num_ws == synthesis_b.num_ws
        self.syn_a = synthesis_a
        self.syn_b = synthesis_b
        self.block_resolutions = list(synthesis_a.block_resolutions)
        self.num_ws = synthesis_a.num_ws
        self.img_resolution = synthesis_a.img_resolution
        self.img_channels = synthesis_a.img_channels
        self.z_dim = z_dim
        self.c_dim = c_dim

        k = len(self.block_resolutions)
        in_dim = z_dim + (c_dim if c_dim > 0 else 0)
        self.coeff_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, k),
        )

        for p in self.syn_a.parameters():
            p.requires_grad = False
        for p in self.syn_b.parameters():
            p.requires_grad = False
        self.syn_a.eval()
        self.syn_b.eval()

    def forward(self, ws, z: Optional[torch.Tensor] = None, c: Optional[torch.Tensor] = None, **block_kwargs):
        block_ws: List[torch.Tensor] = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, None])
            ws32 = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self.syn_a, f'b{res}')
                n = block.num_conv + block.num_torgb
                block_ws.append(ws32.narrow(1, w_idx, n))
                w_idx += block.num_conv

        B = int(ws.shape[0])
        device = ws.device
        if z is None:
            z = torch.zeros(B, self.z_dim, device=device)
        parts = [z]
        if self.c_dim > 0:
            if c is None:
                raise ValueError('PDFM synthesis expects class tensor c when c_dim > 0')
            parts.append(c.to(torch.float32))
        alpha_all = torch.sigmoid(self.coeff_mlp(torch.cat(parts, dim=1)))

        x_a = img_a = None
        x_b = img_b = None
        for blk_idx, (res, cur_ws) in enumerate(zip(self.block_resolutions, block_ws)):
            a = alpha_all[:, blk_idx].view(B, 1, 1, 1)
            block_a = getattr(self.syn_a, f'b{res}')
            block_b = getattr(self.syn_b, f'b{res}')
            x_a, img_a = block_a(x_a, img_a, cur_ws, **block_kwargs)
            with torch.no_grad():
                x_b, img_b = block_b(x_b, img_b, cur_ws, **block_kwargs)

            x_a = a * x_a + (1.0 - a) * x_b.detach()

            if img_a is None and img_b is None:
                pass
            elif img_a is None:
                img_a = img_b
            elif img_b is None:
                pass
            else:
                img_a = a * img_a + (1.0 - a) * img_b

        return img_a


@persistence.persistent_class
class PDFMLayerHybridGenerator(Generator):
    """Hybrid 架构 + PDFM：mapping 可训练（由源模型均值初始化），synthesis 为双路径逐层系数融合。"""

    def __init__(
        self,
        source_hybrid_generators: List[Generator],
        z_dim,
        c_dim,
        w_dim,
        img_resolution,
        img_channels,
        mapping_kwargs=None,
        synthesis_kwargs=None,
    ):
        mapping_kwargs = mapping_kwargs or {}
        synthesis_kwargs = synthesis_kwargs or {}
        super().__init__(
            z_dim=z_dim,
            c_dim=c_dim,
            w_dim=w_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            mapping_kwargs=mapping_kwargs,
            synthesis_kwargs=synthesis_kwargs,
        )
        assert len(source_hybrid_generators) >= 2
        _average_mapping_into(self, source_hybrid_generators)
        syn_a = copy.deepcopy(source_hybrid_generators[0].synthesis).eval().requires_grad_(False)
        syn_b = copy.deepcopy(source_hybrid_generators[1].synthesis).eval().requires_grad_(False)
        self.synthesis = LayerBlendSynthesis(syn_a, syn_b, z_dim=z_dim, c_dim=c_dim)
        self.num_ws = self.synthesis.num_ws

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        noise_mode = synthesis_kwargs.get('noise_mode', 'random')
        fused = {k: v for k, v in synthesis_kwargs.items()}
        fused.setdefault('noise_mode', noise_mode)
        return self.synthesis(ws, z=z, c=c, **fused)
