import torch
import os
import argparse
import sys
from models.dit import DiT 
from utils.schedulers import NoiseScheduler
from torchvision.utils import save_image
from diffusers import AutoencoderKL 
from tqdm import tqdm 
from PIL import Image
from torchvision import transforms

sys.path.append('.')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


VAE_SCALING_FACTOR = 0.18215 
VAE_LATENT_SCALE_FACTOR = 8 
IMG_PIXEL_SIZE = 32
LATENT_SIZE = IMG_PIXEL_SIZE // VAE_LATENT_SCALE_FACTOR # 4
PATCH_SIZE = 4 


def load_config():
    parser = argparse.ArgumentParser(description="DiT Sampling Script")
    parser.add_argument("--resume_ckpt", default="./checkpoints/dit_ckpt_epoch_250.pt", type=str, help="Path to the trained model checkpoint.")
    parser.add_argument("--target_class", default=1, type=int, help="Target class ID for generation (0-C).")
    parser.add_argument("--guidance_scale", default=4.0, type=float, help="Classifier-Free Guidance weight (w).")
    parser.add_argument("--num_samples", default=8, type=int, help="Number of images to generate.")
    parser.add_argument("--num_sampling_steps", default=1000, type=int, help="Number of steps for the DDIM sampler (T/N).")
    parser.add_argument("--output_dir", default="./samples", type=str)
    parser.add_argument("--embed_dim", default=256, type=int)
    parser.add_argument("--depth", default=4, type=int)
    
    args = parser.parse_args()
    return args

def load_vae_model(device):
    
    VAE_MODEL_ID = "stabilityai/sd-vae-ft-mse"
    try:
        vae = AutoencoderKL.from_pretrained(VAE_MODEL_ID).to(device).eval()
        vae.scale_factor = VAE_SCALING_FACTOR
        vae.latent_scale_factor = VAE_LATENT_SCALE_FACTOR
        for param in vae.parameters():
            param.requires_grad = False
        return vae
    except Exception as e:
        print(f" ERROR: Could not load VAE. Error: {e}")
        sys.exit(1)

def load_model_and_scheduler(args, device):
    
    checkpoint = torch.load(args.resume_ckpt, map_location=device)
    total_num_classes = checkpoint['num_classes']
    
    vae = load_vae_model(device)

    model = DiT(
        img_size=LATENT_SIZE,     # 4 (32/8)
        in_channels=4,            
        patch_size=PATCH_SIZE,    
        embed_dim=args.embed_dim, 
        num_heads=4,              
        depth=args.depth, 
        num_classes=total_num_classes
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    scheduler = NoiseScheduler(noise_steps=1000, device=device) 
    
    return model, scheduler, vae, total_num_classes 

@torch.no_grad()
def apply_cfg(model, x_t, t, y_target, w, unconditional_idx):
    
    epsilon_cond = model(x_t, t, y_target)
    y_uncond = torch.full_like(y_target, fill_value=unconditional_idx)
    epsilon_uncond = model(x_t, t, y_uncond)
    
    epsilon_final = epsilon_uncond + w * (epsilon_cond - epsilon_uncond)
    
    return epsilon_final

def decode_and_denormalize(latent_tensor, vae):
        
    vae_input = latent_tensor / vae.scale_factor     
    with torch.no_grad():
        decoded_image = vae.decode(vae_input).sample

    decoded_image = (decoded_image / 2) + 0.5 

    decoded_image = decoded_image.clamp(0, 1) 
    
    return decoded_image # Tensor (B, 3, H, W) in range [0, 1]



@torch.no_grad()
def ddim_sample_loop_grid(model, scheduler, vae, args, total_num_classes, device):
    os.makedirs(args.output_dir, exist_ok=True)
    
    VIS_TIMESTEPS_GRID = [990, 880, 770, 660, 550, 440, 330, 220, 110, 0] 

    UNCONDITIONAL_IDX = total_num_classes - 1 

    img_size = LATENT_SIZE
    num_samples_to_process = args.num_samples
    
    x = torch.randn((num_samples_to_process, 4, img_size, img_size), device=device)
    
    y_target = torch.full((num_samples_to_process,), fill_value=args.target_class, dtype=torch.long, device=device)
    T = scheduler.noise_steps

    num_sampling_steps = args.num_sampling_steps
    skip = T // num_sampling_steps
    time_steps = list(range(0, T, skip))
    time_steps_rev = time_steps[::-1]

    TARGET_SIZE = 128
    
    for i, t_val in tqdm(enumerate(time_steps_rev), desc="DDIM Sampling", total=num_sampling_steps):
        t = torch.tensor([t_val] * num_samples_to_process, device=device)
        
        next_t_val = time_steps_rev[i + 1] if i < len(time_steps_rev) - 1 else 0
        
        epsilon_theta = apply_cfg(model, x, t, y_target, args.guidance_scale, UNCONDITIONAL_IDX)
        
        alpha_t = scheduler.alpha_bars[t_val].view(-1, 1, 1, 1)
        alpha_t_prev = scheduler.alpha_bars[next_t_val].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
        
        x_0_pred = (x - sqrt_one_minus_alpha_t * epsilon_theta) / torch.sqrt(alpha_t)
        
        if t_val in VIS_TIMESTEPS_GRID:
            
            decoded_batch = decode_and_denormalize(x_0_pred, vae) 
            
            for sample_idx in range(num_samples_to_process):
                
                if sample_idx >= 10:
                    break 

                decoded_img = decoded_batch[sample_idx]
                pil_image_32 = transforms.ToPILImage()(decoded_img)
                final_pil_image = pil_image_32.resize(
                    (TARGET_SIZE, TARGET_SIZE), 
                    resample=Image.Resampling.BICUBIC
                )

                filename = f"grid_t{t_val:04d}_n{sample_idx:02d}.png"
                final_pil_image.save(os.path.join(args.output_dir, filename))
        
        dir_xt = torch.sqrt(1.0 - alpha_t_prev) * epsilon_theta
        x_prev = torch.sqrt(alpha_t_prev) * x_0_pred + dir_xt
        
        x = x_prev
        
    final_latent = x
    TARGET_SIZE = 128
    
    decoded_images = decode_and_denormalize(final_latent, vae)
    
    for idx in range(num_samples_to_process):
        
        image_tensor = decoded_images[idx]
        
        pil_image_32 = transforms.ToPILImage()(image_tensor)
        
        final_pil_image = pil_image_32.resize(
            (TARGET_SIZE, TARGET_SIZE), 
            resample=Image.Resampling.BICUBIC
        )

        filename = f"c{args.target_class}_w{args.guidance_scale:.1f}_n{idx}_up{TARGET_SIZE}.png"
        
        final_pil_image.save(os.path.join(args.output_dir, filename))
    
    print(f"Sampling complete. Images saved to {args.output_dir} at {TARGET_SIZE}x{TARGET_SIZE}.")
    
@torch.no_grad()
def ddim_sample_loop(model, scheduler, vae, args, total_num_classes, device):
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Timesteps for DDIM Evolution Visualization (Task 4)
    VIS_TIMESTEPS = [950, 750, 500, 250, 50, 0] 

    UNCONDITIONAL_IDX = total_num_classes - 1 

    img_size = LATENT_SIZE 
    num_samples_to_process = args.num_samples
    
    x = torch.randn((num_samples_to_process, 4, img_size, img_size), device=device)
    
    y_target = torch.full((num_samples_to_process,), fill_value=args.target_class, dtype=torch.long, device=device)
    T = scheduler.noise_steps # 1000
    
    num_sampling_steps = args.num_sampling_steps
    skip = T // num_sampling_steps
    time_steps = list(range(0, T, skip))
    time_steps_rev = time_steps[::-1]

    TARGET_SIZE = 128
    
    for i, t_val in tqdm(enumerate(time_steps_rev), desc="DDIM Sampling", total=num_sampling_steps):
        t = torch.tensor([t_val] * num_samples_to_process, device=device)
            
        next_t_val = time_steps_rev[i + 1] if i < len(time_steps_rev) - 1 else 0
        
        epsilon_theta = apply_cfg(model, x, t, y_target, args.guidance_scale, UNCONDITIONAL_IDX)        
        
        alpha_t = scheduler.alpha_bars[t_val].view(-1, 1, 1, 1)
        alpha_t_prev = scheduler.alpha_bars[next_t_val].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
                
        x_0_pred = (x - sqrt_one_minus_alpha_t * epsilon_theta) / torch.sqrt(alpha_t)
        
        if t_val in VIS_TIMESTEPS:
        
            sample_latent = x_0_pred[0:1] 
        
            decoded_img = decode_and_denormalize(sample_latent, vae)[0] 
            pil_image_32 = transforms.ToPILImage()(decoded_img)
            final_pil_image = pil_image_32.resize(
                (TARGET_SIZE, TARGET_SIZE), 
                resample=Image.Resampling.BICUBIC
            )
            # final_pil_image = pil_image_32

            filename = f"vis_t{t_val}_w{args.guidance_scale:.1f}_up{TARGET_SIZE}.png"
            final_pil_image.save(os.path.join(args.output_dir, filename))
        
        dir_xt = torch.sqrt(1.0 - alpha_t_prev) * epsilon_theta
        x_prev = torch.sqrt(alpha_t_prev) * x_0_pred + dir_xt
        
        x = x_prev
        
    final_latent = x
    decoded_images = decode_and_denormalize(final_latent, vae)    
    
    for idx in range(num_samples_to_process):        
        image_tensor = decoded_images[idx]
            
        pil_image_32 = transforms.ToPILImage()(image_tensor)
        
        final_pil_image = pil_image_32.resize(
            (TARGET_SIZE, TARGET_SIZE), 
            resample=Image.Resampling.BICUBIC 
        )

        filename = f"c{args.target_class}_w{args.guidance_scale:.1f}_n{idx}_up{TARGET_SIZE}_{args.num_sampling_steps}.png"
        
        final_pil_image.save(os.path.join(args.output_dir, filename))
    
    print(f"Sampling complete. Images saved to {args.output_dir} at {TARGET_SIZE}x{TARGET_SIZE}.")
    
    # # Save the images
    # for idx in range(args.num_samples):
    #     filename = f"c{args.target_class}_w{args.guidance_scale:.1f}_n{idx}.png"
    #     save_image(decoded_images[idx], os.path.join(args.output_dir, filename))
    
    # print(f"Sampling complete. Images saved to {args.output_dir}")


if __name__ == '__main__':
    args = load_config()
    
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    elif torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
    else:
        DEVICE = torch.device('cpu')

    model, scheduler, vae, total_num_classes = load_model_and_scheduler(args, DEVICE)
    
    print(f"Generating {args.num_samples} samples for class {args.target_class} with w={args.guidance_scale} in {args.num_sampling_steps} DDIM steps.")
    # ddim_sample_loop(model, scheduler, vae, args, total_num_classes, DEVICE)
    ddim_sample_loop_grid(model, scheduler, vae, args, total_num_classes, DEVICE)
        
# python -m train.sample \
# --resume_ckpt ./checkpoints/dit_ckpt_epoch_70.pt \
# --embed_dim 128 \
# --depth 2 \
# --guidance_scale 12.0 \
# --num_samples 8

    
# python -m train.sample \
# --resume_ckpt ./checkpoints/dit_ckpt_epoch_250.pt \
# --embed_dim 256 \
# --depth 4 \
# --guidance_scale 1.0 \
# --num_samples 8

# python -m train.sample \
# --resume_ckpt ./checkpoints_64/dit_ckpt_epoch_40.pt \
# --embed_dim 256 \
# --depth 6 \
# --guidance_scale 1.0 \
# --num_samples 8

# python -m train.sample \
#     --resume_ckpt ./checkpoints/dit_ckpt_epoch_250.pt \
#     --embed_dim 256 \
#     --depth 4 \
#     --guidance_scale 1.0 \
#     --num_sampling_steps 500 \
#     --num_samples 1 \
#     --output_dir samples_timesteps 

# python -m train.sample \
#     --resume_ckpt ./checkpoints/dit_ckpt_epoch_250.pt \
#     --embed_dim 256 \
#     --depth 4 \
#     --guidance_scale 1.0 \
#     --num_sampling_steps 500 \
#     --num_samples 10 \
#     --output_dir sample_grid