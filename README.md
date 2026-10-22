# UR-GAN

This is the official repository of **Union Rock Generative Adversarial Network
(UR-GAN)**. If you find this repository useful for your research or project,
please cite the related work.

Digital rock generative models are usually trained separately for different
lithologies and datasets. Due to differences in training data, lithology
characteristics, and data distributions, existing pre-trained models lack a
unified representation and reuse mechanism, making them difficult to transfer
and fuse directly. To address this problem, we propose a weight-space fusion
and reuse framework for pre-trained digital rock models. In this project,
UR-GAN uses StyleGAN2-ADA as an example to map different digital rock models
into a unified representation space and support model alignment, weight
fusion, conditional generation, and cross-lithology interpolation.

UR-GAN contains two main modules:

- **FATM (Feature-Adaptive Transfer Module)**: extracts and transfers
  structural features from different lithologies in feature space for
  progressive cross-lithology weight alignment.
- **PDFM (Parameter-Level Dynamic Fusion Module)**: dynamically fuses the
  parameters of pre-trained models layer by layer under conditional control.

The current project uses StyleGAN2-ADA as an example. FATM and PDFM can also be
adapted to other GAN projects according to their network structures, training
procedures, and model formats.

## Usage

### Requirements

This project requires:

```plain
Python
pytorch
torchvision
numpy
scipy
Pillow
Click
psutil
```

### Datasets and pretrained models

Place your datasets in the `datasets` directory. The dataset can be an image
directory or a `.zip` file in the StyleGAN2-ADA format.


### Train new models

First, train two digital rock generative models:

```bash
python train.py --outdir=output/model_A --data=datasets/dataset_A \
    --cfg=auto --gpus=1 --kimg=1000 --metrics=none

python train.py --outdir=output/model_B --data=datasets/dataset_B \
    --cfg=auto --gpus=1 --kimg=1000 --metrics=none
```

Record the paths of the two models:

```text
MODEL_A = output/model_A/.../network-snapshot-XXXXXX.pkl
MODEL_B = output/model_B/.../network-snapshot-XXXXXX.pkl
DATA_A  = datasets/dataset_A
DATA_B  = datasets/dataset_B
```

### Weight alignment

Use FATM to align the weights of the two pretrained models:

```bash
python train.py --outdir=output/alignment --gpus=1 --cfg=auto \
    --kimg=500 --metrics=none --merge_mode=True --use_fatm=True \
    --data_paths=DATA_A --data_paths=DATA_B --num_classes=2 \
    --class_percentages=0 --class_percentages=1 \
    --merge_model_paths=MODEL_A --merge_model_paths=MODEL_B \
    --root_model_path=MODEL_A
```

Use the checkpoint generated during alignment as:

```text
ALIGNED_B = output/alignment/.../network-snapshot-XXXXXX.pkl
```

### Weight fusion

Use PDFM to dynamically fuse the aligned models:

```bash
python train.py --outdir=output/fusion --gpus=1 --cfg=auto \
    --kimg=500 --metrics=none --merge_mode=True \
    --hybrid_mode=True --use_pdfm=True \
    --data_paths=DATA_A --data_paths=DATA_B --num_classes=2 \
    --merge_model_paths=MODEL_A --merge_model_paths=MODEL_B \
    --hybrid_model_paths=ALIGNED_B
```

### Generate images

Generate images under a specified condition using the fused model:

```bash
python generate.py \
    --network=output/fusion/.../network-snapshot-XXXXXX.pkl \
    --outdir=output/generated \
    --seeds=0-9 \
    --class=0
```

`--class=0` represents the condition of the first dataset, while `--class=1`
represents the condition of the second dataset.

### Interpolation

Perform continuous interpolation between the two lithology conditions:

```bash
python generate_mixed.py \
    --network=output/fusion/.../network-snapshot-XXXXXX.pkl \
    --outdir=output/interpolation \
    --seeds=0-4 \
    --alphas=0.0,...,1.0
```

Here, `alpha=0.0` represents the first dataset and `alpha=1.0` represents the
second dataset.

### Adapting to other projects

When applying FATM and PDFM to another GAN project, the following components
need to be adapted according to the target project:

1. Model loading and weight-saving formats.
2. Intermediate feature or layer-wise parameter structures.
3. Training losses and optimizer parameters.
4. Checkpoint saving and model inference scripts.


## Acknowledgements

The base code of this project is mainly based on the official PyTorch
implementation of StyleGAN2-ADA released by NVIDIA.

Related paper:

- Karras T, Aittala M, Hellsten J, Laine S, Lehtinen J, Aila T. Training
  Generative Adversarial Networks with Limited Data[C]. Advances in Neural
  Information Processing Systems, 2020.

