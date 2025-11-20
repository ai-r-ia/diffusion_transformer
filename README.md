# Diffusion Transformer (DiT) — PyTorch Implementation & Experiments

This repository contains a full PyTorch implementation of the Diffusion Transformer (DiT) architecture introduced in DiT: Diffusion Transformers for Image Generation (Peebles & Xie, 2022).
The project includes:

## Core Components

 - Forward diffusion process
 - Noise schedulers (linear, cosine)
 - Sinusoidal timestep embeddings
 - Image patchification & sequence representation
 - Modular DiT blocks (attention, FFN, layer norms)

## Training

 - Full DDPM-style training loop
 - Configurable hyperparameters (depth, patch size, timesteps)
 - Training curves + logging utilities
 - FID scoring pipeline

## Sampling

 - DDIM sampling
 - Reverse diffusion visualization
 - Global 10×10 grids showing progression over timesteps
 - Classifier-free guidance (CFG) with scale sensitivity analysis

## Experiments

 - Impact of number of timesteps (T=100, 500, 1000)
 - Impact of DiT depth (4, 6, 8 blocks)
 - Attention head ablations
 - CFG sensitivity (w = 0 → 10)

## Artifacts Included

 - Training loss curves - training_loss_trend.png
 - Sampling progression grids -  DDIM_Progression_Grid_10x10.png
 - Report summarizing architecture decisions, challenges, and observations - DiT_report.pdf
 - FID comparisons for major configurations


## Usage (check config.py for additional config options):

Training:
python train/train_script.py \
--epochs 350 \
--learning_rate 3e-4 \
--img_size 32 \
--batch_size 8 \
--embed_dim 256
--depth 4

Reverse Sampling (update based on your model checkpoint path): 
python -m train.sample \
     --resume_ckpt ./checkpoints/dit_ckpt_epoch_250.pt \
     --embed_dim 256 \
     --depth 4 \
     --guidance_scale 1.0 \
     --num_sampling_steps 500 \
     --num_samples 10 \
     --output_dir samples

for FID scores (replace directory paths based on your pwd):
python -m pytorch_fid \
    /Users/riash/Desktop/dit/fid_data/real_images_128x128 \
    /Users/riash/Desktop/dit/Diffusion-DiT-Assignment/fid_data/generated_w12.0/ \
    --device mps \
    --batch-size 64 \
    --num-workers 0
