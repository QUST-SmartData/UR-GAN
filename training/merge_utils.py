"""Model merging utilities for binary image GANs"""

import torch
import torch.nn.functional as F
from typing import List
import dnnlib
import legacy
from training.networks import Generator, Discriminator
from torch_utils import misc

def get_models(model_paths: List[str], new_n_classes: int, device):
    """Load multiple models and extend their class dimensions if needed"""
    generators = []
    discriminators = []

    for model_path in model_paths:
        generator, discriminator = get_extended_model(model_path=model_path, 
                                                      new_n_classes=new_n_classes,
                                                      device=device)
        generators.append(generator)
        discriminators.append(discriminator)

    return generators, discriminators

def get_extended_model(model_path: str, new_n_classes: int, device):
    """Load a model and extend its class dimensions if needed"""
    with dnnlib.util.open_url(model_path) as f:
        model_data = legacy.load_network_pkl(f)

        generator = model_data["G_ema"].eval().requires_grad_(False).cpu()
        discriminator = model_data["D"].eval().requires_grad_(False).cpu()

        if generator.c_dim < new_n_classes:
            generator, discriminator = extend_GAN_embeddings(input_G=generator, 
                                                             input_D=discriminator, 
                                                             new_n_classes=new_n_classes)

        return generator.to(device), discriminator.to(device)

def get_one_hot_batch(index_value, num_classes, batch_size):
    """Generate one-hot encoded batch for class conditioning"""
    c = F.one_hot(torch.tensor(index_value), num_classes=num_classes)
    c = c.unsqueeze(0).repeat(batch_size, 1)
    return c

@torch.no_grad()
def sample_merge_generators(merge_generators, batch_size, class_percentages, device):
    """Sample from multiple generators based on class percentages"""
    samples = []
    labels = []

    for generator_idx, generator in enumerate(merge_generators):
        num_samples = int(batch_size * class_percentages[generator_idx])
        if num_samples == 0:
            continue
            
        z = torch.randn([num_samples, generator.z_dim], device=device)
        c = get_one_hot_batch(generator_idx, len(merge_generators), num_samples).to(device)

        on_cpu = next(generator.parameters()).device.type == 'cpu'
        gen = generator.to(device)
        with torch.no_grad():
            img = gen(z, c, noise_mode='const')
        if on_cpu:
            generator.cpu()
        samples.append(img)
        labels.append(c)

    samples = torch.cat(samples, dim=0)
    labels = torch.cat(labels, dim=0)

    return samples, labels

def extend_GAN_embeddings(input_G, input_D, new_n_classes):
    """Extend GAN embeddings to support more classes (model rooting)"""
    input_G.cpu()
    input_D.cpu()

    output_G = Generator(z_dim=input_G.z_dim,
                         c_dim=new_n_classes,
                         w_dim=input_G.w_dim,
                         img_channels=input_G.img_channels,
                         img_resolution=input_G.img_resolution,
                         mapping_kwargs={'num_layers': 8},
                         synthesis_kwargs={'channel_base': 32768, 'channel_max': 512, 'conv_clamp': 256, 'num_fp16_res': 4})
    output_G = output_G.eval().requires_grad_(False).cpu()

    output_D = Discriminator(c_dim=new_n_classes,
                             img_resolution=input_D.img_resolution,
                             img_channels=input_D.img_channels,
                             num_fp16_res=4,
                             conv_clamp=256,
                             epilogue_kwargs={'mbstd_group_size': 4})
    output_D = output_D.eval().requires_grad_(False).cpu()

    assert input_G.c_dim < output_G.c_dim, "New number of classes must to be bigger than the old one"

    # 安全复制：仅在名称相同且张量形状一致时复制权重/缓冲，避免因类嵌入大小改变而报错
    def _pairs_to_dict(pairs_or_dict):
        if isinstance(pairs_or_dict, dict):
            return pairs_or_dict
        return {k: v for k, v in pairs_or_dict}

    def _safe_copy(src_module, dst_module):
        src_tensors = _pairs_to_dict(misc.named_params_and_buffers(src_module))
        for name, dst_tensor in _pairs_to_dict(misc.named_params_and_buffers(dst_module)).items():
            src_tensor = src_tensors.get(name, None)
            if (src_tensor is not None) and (tuple(src_tensor.shape) == tuple(dst_tensor.shape)):
                dst_tensor.copy_(src_tensor.detach())

    _safe_copy(input_G, output_G)
    _safe_copy(input_D, output_D)

    return output_G, output_D
