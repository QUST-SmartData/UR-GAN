from typing import List, Optional
import torch

from training.networks import Generator, Discriminator
from torch_utils import misc

class HybridGenerator(Generator):
    def __init__(self,
                 source_hybrid_generators: List[Generator],
                 z_dim,
                 c_dim,
                 w_dim,
                 img_resolution,
                 img_channels,
                 mapping_kwargs      = {},
                 synthesis_kwargs    = {},
                 blend_weights: Optional[List[float]] = None) -> None:
        super().__init__(z_dim = z_dim,
                         c_dim = c_dim,
                         w_dim = w_dim,
                         img_resolution = img_resolution,
                         img_channels = img_channels,
                         mapping_kwargs = mapping_kwargs,
                         synthesis_kwargs = synthesis_kwargs)

        def _pairs_to_dict(pairs_or_dict):
            if isinstance(pairs_or_dict, dict):
                return pairs_or_dict
            return {k: v for k, v in pairs_or_dict}

        generators_parameters = []
        for generator in source_hybrid_generators:
            generators_parameters.append(_pairs_to_dict(misc.named_params_and_buffers(generator)))

        n_src = len(generators_parameters)
        if blend_weights is not None and len(blend_weights) == n_src:
            sw = sum(float(w) for w in blend_weights)
            wts = [float(w) / sw for w in blend_weights]
        else:
            wts = [1.0 / n_src] * n_src

        self.requires_grad_(False)
        # Average source tensors with matching shapes.
        self_params = _pairs_to_dict(misc.named_params_and_buffers(self))
        first_params = generators_parameters[0]
        for name, tensor in self_params.items():
            candidates = []
            for gp in generators_parameters:
                t = gp.get(name, None)
                if (t is not None) and (tuple(t.shape) == tuple(tensor.shape)):
                    candidates.append(t)
            if len(candidates) >= 1:
                if len(candidates) == n_src:
                    avg = sum(wts[i] * candidates[i].detach() for i in range(n_src))
                else:
                    avg = sum(c.detach() for c in candidates) / float(len(candidates))
                tensor.copy_(avg)
            else:
                # Use the root tensor when its shape matches.
                t0 = first_params.get(name, None)
                if (t0 is not None) and (tuple(t0.shape) == tuple(tensor.shape)):
                    tensor.copy_(t0.detach())


class HybridDiscriminator(Discriminator):
    def __init__(self,
        source_hybrid_discriminators: List[Discriminator],
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ) -> None:
        super().__init__(c_dim=c_dim, img_resolution=img_resolution, img_channels=img_channels, architecture=architecture,
                         channel_base=channel_base, channel_max=channel_max, num_fp16_res=num_fp16_res, conv_clamp=conv_clamp,
                         cmap_dim=cmap_dim, block_kwargs=block_kwargs, mapping_kwargs=mapping_kwargs, epilogue_kwargs=epilogue_kwargs)

        def _pairs_to_dict(pairs_or_dict):
            if isinstance(pairs_or_dict, dict):
                return pairs_or_dict
            return {k: v for k, v in pairs_or_dict}

        discriminators_parameters = []
        for discriminator in source_hybrid_discriminators:
            discriminators_parameters.append(_pairs_to_dict(misc.named_params_and_buffers(discriminator)))

        self.requires_grad_(False)
        self_params = _pairs_to_dict(misc.named_params_and_buffers(self))
        first_params = discriminators_parameters[0]
        for name, tensor in self_params.items():
            candidates = []
            for dp in discriminators_parameters:
                t = dp.get(name, None)
                if (t is not None) and (tuple(t.shape) == tuple(tensor.shape)):
                    candidates.append(t)
            if len(candidates) >= 1:
                avg = sum(c.detach() for c in candidates) / float(len(candidates))
                tensor.copy_(avg)
            else:
                t0 = first_params.get(name, None)
                if (t0 is not None) and (tuple(t0.shape) == tuple(tensor.shape)):
                    tensor.copy_(t0.detach())
