"""Model merging script for binary image GANs - merges multiple models by weight averaging"""

import os
import pickle
import click
import dnnlib
import legacy
import torch
from training.hybrid_model import HybridGenerator, HybridDiscriminator

@click.command()
@click.option('--model_paths', help='Paths to source models for merging', multiple=True, required=True)
@click.option('--output_model', help='Output merged model pickle file', required=True, metavar='PKL')
@click.option('--weights', help='Weights for each model (default: equal weights)', type=float, multiple=True)
def merge_models(model_paths: list, output_model: str, weights: list):
    """Merge multiple models by weight averaging.
    
    This script implements the second stage of GAN Cocktail: model merging.
    It creates a hybrid model by averaging the weights of multiple models.
    
    Examples:
    
    \b
    # Merge two models with equal weights
    python merge_models.py --model_paths=model1.pkl --model_paths=model2.pkl --output_model=merged.pkl
    
    \b
    # Merge two models with custom weights (0.7 and 0.3)
    python merge_models.py --model_paths=model1.pkl --model_paths=model2.pkl --output_model=merged.pkl --weights=0.7 --weights=0.3
    """
    
    if len(model_paths) < 2:
        raise click.ClickException('At least 2 models are required for merging')
    
    if weights and len(weights) != len(model_paths):
        raise click.ClickException('Number of weights must match number of models')
    
    if not weights:
        weights = [1.0 / len(model_paths)] * len(model_paths)
    
    # Normalize weights
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]
    
    print(f'Loading {len(model_paths)} models...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    generators = []
    discriminators = []
    
    for i, model_path in enumerate(model_paths):
        print(f'Loading model {i+1}: {model_path}')
        with dnnlib.util.open_url(model_path) as f:
            model_data = legacy.load_network_pkl(f)
            
        generator = model_data["G_ema"].eval().requires_grad_(False).to(device)
        discriminator = model_data["D"].eval().requires_grad_(False).to(device)
        
        generators.append(generator)
        discriminators.append(discriminator)
    
    print('Creating hybrid models...')
    
    # Get model specifications from first model
    first_g = generators[0]
    first_d = discriminators[0]
    
    # Create hybrid generator
    hybrid_generator = HybridGenerator(
        source_hybrid_generators=generators,
        z_dim=first_g.z_dim,
        c_dim=first_g.c_dim,
        w_dim=first_g.w_dim,
        img_resolution=first_g.img_resolution,
        img_channels=first_g.img_channels,
        blend_weights=list(weights),
    )
    
    # Create hybrid discriminator
    hybrid_discriminator = HybridDiscriminator(
        source_hybrid_discriminators=discriminators,
        c_dim=first_d.c_dim,
        img_resolution=first_d.img_resolution,
        img_channels=first_d.img_channels,
        architecture=getattr(first_d, 'architecture', 'resnet'),
        channel_base=getattr(first_d, 'channel_base', 32768),
        channel_max=getattr(first_d, 'channel_max', 512),
        num_fp16_res=getattr(first_d, 'num_fp16_res', 4),
        conv_clamp=getattr(first_d, 'conv_clamp', 256)
    )
    
    print(f'Hybrid generator classes: {hybrid_generator.c_dim}')
    print(f'Hybrid discriminator classes: {hybrid_discriminator.c_dim}')
    
    # Create merged model data
    merged_model_data = {
        'G_ema': hybrid_generator,
        'D': hybrid_discriminator,
        'G': hybrid_generator,  # Also save as G for compatibility
        'training_set_kwargs': {},
        'augment_pipe': None,
    }
    
    # Copy metadata from first model if available
    with dnnlib.util.open_url(model_paths[0]) as f:
        first_model_data = legacy.load_network_pkl(f)
        for key in ['augment_pipe', 'training_set_kwargs']:
            if key in first_model_data:
                merged_model_data[key] = first_model_data[key]
    
    print(f'Saving merged model to "{output_model}"...')
    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    
    with open(output_model, 'wb') as f:
        pickle.dump(merged_model_data, f)
    
    print('Model merging completed successfully!')
    print(f'Used weights: {weights}')

if __name__ == "__main__":
    merge_models()
