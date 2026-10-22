#!/usr/bin/env python3
"""Generate mixed images using soft label interpolation between A and B styles."""

import torch
import numpy as np
import dnnlib
import legacy
import PIL.Image
import os
import click

@click.command()
@click.option('--network', 'network_path', help='Path to the trained model pickle', required=True)
@click.option('--outdir', help='Output directory for mixed images', required=True)
@click.option('--seeds', help='Random seeds (e.g., 0-9 or 0,1,2,3)', default='0-9')
@click.option('--alphas', help='Mixing ratios (e.g., 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)', default='0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0')
@click.option('--is3D', 'is3d_steps', type=int, default=0, show_default=True, help='If >0, ignore --alphas and generate a 0..1 linspace with N steps (including alpha=0 and alpha=1). Saves stitched transition row + individual frames under outdir/3D/.')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1.0)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const')
def generate_mixed_images(network_path, outdir, seeds, alphas, is3d_steps, truncation_psi, noise_mode):
    """Generate mixed images using soft label interpolation.
    
    Examples:
    
    \b
    # Basic usage with default parameters
    python generate_mixed.py --network=model.pkl --outdir=output/mixed
    
    \b
    # Custom seeds and mixing ratios
    python generate_mixed.py --network=model.pkl --outdir=output/mixed --seeds=0-4 --alphas=0.0,0.5,1.0
    
    \b
    # With truncation
    python generate_mixed.py --network=model.pkl --outdir=output/mixed --trunc=0.7
    """
    
    if '-' in seeds:
        start, end = map(int, seeds.split('-'))
        seed_list = list(range(start, end + 1))
    else:
        seed_list = [int(x.strip()) for x in seeds.split(',')]
    
    if is3d_steps and is3d_steps > 0:
        if is3d_steps < 2:
            raise click.ClickException('--is3D must be >= 2 (need to include alpha=0 and alpha=1).')
        alpha_list = np.linspace(0.0, 1.0, int(is3d_steps)).tolist()
    else:
        alpha_list = [float(x.strip()) for x in alphas.split(',')]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading network from "{network_path}"...')
    
    with dnnlib.util.open_url(network_path) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    
    os.makedirs(outdir, exist_ok=True)
    outdir_3d = os.path.join(outdir, '3D')
    if is3d_steps and is3d_steps > 0:
        os.makedirs(outdir_3d, exist_ok=True)
    
    print(f'Generating mixed images...')
    print(f'Seeds: {seed_list}')
    print(f'Alphas: {alpha_list}')
    print(f'Output directory: {outdir}')
    if is3d_steps and is3d_steps > 0:
        print(f'3D output directory: {outdir_3d}')
    
    for seed in seed_list:
        print(f'Generating row for seed {seed}...')
        
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        
        row_images = []
        # When --is3D is enabled, also save individual frames in-order under outdir/3D/
        if is3d_steps and is3d_steps > 0:
            frames_dir = os.path.join(outdir_3d, f'seed{seed:04d}')
            os.makedirs(frames_dir, exist_ok=True)
            frame_digits = max(4, len(str(len(alpha_list) - 1)))
        
        for alpha_idx, alpha in enumerate(alpha_list):
            # Create soft label: [1-alpha, alpha] for A->B interpolation
            c = torch.tensor([[1-alpha, alpha]], dtype=torch.float32, device=device)
            
            with torch.no_grad():
                img = G(z, c, truncation_psi=truncation_psi, noise_mode=noise_mode)
                img = (img.permute(0,2,3,1) * 127.5 + 128).clamp(0,255).to(torch.uint8)[0].cpu().numpy()
                
                # Convert to PIL Image with binary thresholding
                if img.shape[2] == 1:  # Single channel (grayscale)
                    img = img.squeeze(2)
                    # Apply binary thresholding (threshold = 128)
                    img_binary = (img > 128).astype(np.uint8) * 255
                    pil_img = PIL.Image.fromarray(img_binary, 'L')
                elif img.shape[2] == 3:  # RGB
                    # Convert to grayscale first, then apply binary thresholding
                    img_gray = np.mean(img, axis=2)
                    img_binary = (img_gray > 128).astype(np.uint8) * 255
                    pil_img = PIL.Image.fromarray(img_binary, 'L')
                else:
                    raise ValueError(f"Unexpected image shape: {img.shape}")
                
                row_images.append(pil_img)

                if is3d_steps and is3d_steps > 0:
                    # Index-prefixed names preserve frame order.
                    frame_name = f'frame{alpha_idx:0{frame_digits}d}_alpha{alpha:.6f}.png'
                    pil_img.save(os.path.join(frames_dir, frame_name))
        
        # Create and save row image
        if is3d_steps and is3d_steps > 0:
            create_single_row_image(outdir_3d, seed, row_images, alpha_list)
        else:
            create_single_row_image(outdir, seed, row_images, alpha_list)
    
    if is3d_steps and is3d_steps > 0:
        print(f'Generated {len(seed_list)} 3D rows + frames in "{outdir_3d}"')
    else:
        print(f'Generated {len(seed_list)} row images in "{outdir}"')
    print(f'Alpha values: {alpha_list}')
    print(f'  - Alpha 0.0 = Pure A style')
    print(f'  - Alpha 1.0 = Pure B style') 
    print(f'  - Alpha 0.5 = 50% A + 50% B')
    print(f'Each row shows one seed with all alpha interpolations')

def create_single_row_image(outdir, seed, row_images, alpha_list):
    """Create a single row image for one seed."""
    if not row_images:
        print(f"No images for seed {seed}")
        return
    
    img_width, img_height = row_images[0].size
    
    cols = len(row_images)
    row_width = cols * img_width
    row_height = img_height
    
    if row_images[0].mode == 'L':  # Grayscale
        row_img = PIL.Image.new('L', (row_width, row_height), 255)
    else:  # RGB
        row_img = PIL.Image.new('RGB', (row_width, row_height), (255, 255, 255))
    
    for col_idx, img in enumerate(row_images):
        x = col_idx * img_width
        y = 0
        row_img.paste(img, (x, y))
    
    row_filename = f'seed{seed:04d}_row.png'
    row_path = os.path.join(outdir, row_filename)
    row_img.save(row_path)
    print(f'  Saved: {row_filename} ({cols} images in row)')

if __name__ == "__main__":
    generate_mixed_images()
