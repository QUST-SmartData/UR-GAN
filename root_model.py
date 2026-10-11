"""Model rooting script for binary image GANs - extends model class dimensions"""

import os
import pickle
import click
import dnnlib
import legacy
import torch
from training.merge_utils import extend_GAN_embeddings

@click.command()
@click.option('--input_model', help='Input model pickle file', required=True, metavar='PKL')
@click.option('--output_model', help='Output rooted model pickle file', required=True, metavar='PKL')
@click.option('--new_classes', help='New number of classes', type=int, required=True, metavar='INT')
def root_model(input_model: str, output_model: str, new_classes: int):
    """Root a model by extending its class dimensions.
    
    This script implements the first stage of GAN Cocktail: model rooting.
    It extends the class dimensions of a trained model to support more classes.
    
    Examples:
    
    \b
    # Root a model to support 2 classes
    python root_model.py --input_model=model1.pkl --output_model=rooted_model1.pkl --new_classes=2
    """
    
    print(f'Loading input model from "{input_model}"...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with dnnlib.util.open_url(input_model) as f:
        model_data = legacy.load_network_pkl(f)
        
    input_G = model_data["G_ema"].eval().requires_grad_(False).cpu()
    input_D = model_data["D"].eval().requires_grad_(False).cpu()
    
    print(f'Original model classes: {input_G.c_dim}')
    print(f'Target classes: {new_classes}')
    
    if input_G.c_dim >= new_classes:
        print('Warning: Model already has enough classes. No rooting needed.')
        return
    
    print('Extending model embeddings...')
    output_G, output_D = extend_GAN_embeddings(input_G, input_D, new_classes)
    
    print(f'Rooted model classes: {output_G.c_dim}')
    
    # Create new model data
    rooted_model_data = {
        'G_ema': output_G,
        'D': output_D,
        'G': output_G,  # Also save as G for compatibility
        'training_set_kwargs': model_data.get('training_set_kwargs', {}),
        'augment_pipe': model_data.get('augment_pipe', None),
    }
    
    # Copy other metadata if available
    for key in ['augment_pipe', 'training_set_kwargs']:
        if key in model_data:
            rooted_model_data[key] = model_data[key]
    
    print(f'Saving rooted model to "{output_model}"...')
    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    
    with open(output_model, 'wb') as f:
        pickle.dump(rooted_model_data, f)
    
    print('Model rooting completed successfully!')

if __name__ == "__main__":
    root_model()
