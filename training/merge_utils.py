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

    generator_kwargs = dict(input_G.init_kwargs)
    generator_kwargs['c_dim'] = new_n_classes
    output_G = Generator(**generator_kwargs)
    output_G = output_G.eval().requires_grad_(False).cpu()

    discriminator_kwargs = dict(input_D.init_kwargs)
    discriminator_kwargs['c_dim'] = new_n_classes
    output_D = Discriminator(**discriminator_kwargs)
    output_D = output_D.eval().requires_grad_(False).cpu()

    assert input_G.c_dim < output_G.c_dim, "Target class count must exceed the source class count"

    # Preserve matching tensors while initializing expanded class embeddings.
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
