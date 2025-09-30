# Implementation for Diffusion-Transformer

Diffusion Transformer (DiT) implementation for image generation based on the tasks given in Assignment-Diffusion-PA.pdf
Please find task specific code in the following files: 

Task 1: 
    --utils/modules.py
    --utils/schedulers.py

Task 2:
    --models/dit.py
    --utils/modules.py

Task 3:
    --train/train_script.py
    --utils/loss_visualizer.py

Task 4:
    --train/sample.py
    --train/make_grid.py

Task 5:
    --train/train_script.py
    --train/sample.py

Additionally, please find a report in the following document: DiT_report.pdf
Loss curve can be found in: training_loss_trend.png
100 image grid for different time step progression is available at: DDIM_Progression_Grid_10x10.png

for running the code please use the following commands (check config.py for additional config options):

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