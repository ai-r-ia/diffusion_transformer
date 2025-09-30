# Task 3: Main training loop
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
import sys
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from diffusers import AutoencoderKL 

sys.path.append('.')

from models.dit import DiT
from utils.schedulers import NoiseScheduler
from data.data import LandscapeDataset
from configs.config import create_parser, get_logger 
from configs.seed import set_seed


VAE_SCALING_FACTOR = 0.18215 
VAE_LATENT_SCALE_FACTOR = 8  

def load_vae_model(device):

    VAE_MODEL_ID = "stabilityai/sd-vae-ft-mse"
    
    try:
        vae = AutoencoderKL.from_pretrained(
            VAE_MODEL_ID
        ).to(device).eval()

        vae.scale_factor = VAE_SCALING_FACTOR
        vae.latent_scale_factor = VAE_LATENT_SCALE_FACTOR
        
        for param in vae.parameters():
            param.requires_grad = False
            
        return vae
    
    except Exception as e:
        print(f" ERROR: Could not load VAE from HuggingFace.")
        print(f"Error details: {e}")
        sys.exit(1) 


def train_one_epoch(model, scheduler, dataloader, optimizer, device, unconditional_idx, vae):
    model.train()
    total_loss = 0
    
    for step, (x_0, y) in tqdm(enumerate(dataloader)): 
        x_0 = x_0.to(device) # x_0 is the clean IMAGE (B, 3, H, W)
        y = y.to(device)
        B = x_0.shape[0]
        
        with torch.no_grad():
            latent = vae.encode(x_0).latent_dist.sample()         
            latent = latent * vae.scale_factor 
                        
        t = torch.randint(0, scheduler.noise_steps, (B,), device=device).long()
        
        x_t, target_noise = scheduler.q_sample(latent, t) # x_t is the noisy latent

        p_uncond = 0.1 
        mask = torch.rand(B, device=device) < p_uncond
        y[mask] = unconditional_idx 
        
        predicted_noise = model(x_t, t, y) 

        loss = MSELoss()(predicted_noise, target_noise) 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
    return avg_loss
    
    
if __name__ == "__main__":
    set_seed()
    parser = create_parser()
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    LOGGER = get_logger("DiT_Trainer", args.log_dir)
    
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    elif torch.backends.mps.is_available():
        DEVICE = torch.device('mps') 
    else:
        DEVICE = torch.device('cpu')
        
    
    T = args.time_steps 
    IMG_SIZE = args.img_size 
    PATCH_SIZE = args.patch_size 
    EMBED_DIM = args.embed_dim   
    NUM_HEADS = args.num_heads   
    DEPTH = args.depth           
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    NUM_EPOCHS = args.epochs
    DATA_ROOT = args.data_root
    NUM_DATA_CLASSES = args.num_classes

    LOGGER.info(f"Starting training on device: {DEVICE}")
    LOGGER.info(f"Configuration: Epochs={NUM_EPOCHS}, BatchSize={BATCH_SIZE}, LR={LEARNING_RATE}")

    try:
        dataset = LandscapeDataset(root_dir=DATA_ROOT, img_size=IMG_SIZE)            
        NUM_DATA_CLASSES = dataset.num_classes 
        
    except FileNotFoundError as e:
        print(e)
        print("Failed to load dataset.")
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers= 8)
    scheduler = NoiseScheduler(noise_steps=T, device = DEVICE)
    
    vae = load_vae_model(DEVICE)
    LATENT_SIZE = IMG_SIZE // VAE_LATENT_SCALE_FACTOR
    LOGGER.info(f"VAE Loaded. Pixel Size: {IMG_SIZE}x{IMG_SIZE}, Latent Size: {LATENT_SIZE}x{LATENT_SIZE}")

    UNCONDITIONAL_IDX = NUM_DATA_CLASSES  
    TOTAL_NUM_CLASSES = NUM_DATA_CLASSES + 1 

    model = DiT(
        img_size=LATENT_SIZE,
        in_channels=4,       
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        depth=DEPTH,
        num_classes=TOTAL_NUM_CLASSES
    ).to(DEVICE)
    
    START_EPOCH = 0
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)


    if args.resume_ckpt: 
        try:
            checkpoint = torch.load(args.resume_ckpt, map_location=DEVICE)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            START_EPOCH = checkpoint['epoch']
            
            LOGGER.info(f"Successfully resumed training from checkpoint: {args.resume_ckpt}")
            LOGGER.info(f"Continuing from Epoch {START_EPOCH + 1}.")

        except FileNotFoundError:
            LOGGER.error(f"Checkpoint file not found at {args.resume_ckpt}. ")
        except Exception as e:
            LOGGER.error(f"Error loading checkpoint: {e}.")
        
    LOGGER.info(f"Starting CFG training run for {NUM_EPOCHS} epochs.")
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-7)

    for epoch in range(START_EPOCH, NUM_EPOCHS):
        LOGGER.info(f"--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        
        avg_loss = train_one_epoch(model, scheduler, dataloader, optimizer, DEVICE, UNCONDITIONAL_IDX, vae)
        
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        LOGGER.info(f"Epoch {epoch + 1} finished. Avg Loss: {avg_loss:.4f} | Current LR: {current_lr:.6e}") 
    
        if (epoch + 1) % 10 == 0 or (epoch + 1) == NUM_EPOCHS:
            ckpt_path = os.path.join(args.ckpt_dir, f"dit_ckpt_epoch_{epoch + 1}.pt")
            torch.save(
                {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'num_classes': TOTAL_NUM_CLASSES
                },
                ckpt_path
            )
            LOGGER.info(f"Checkpoint saved to {ckpt_path}")

    LOGGER.info("Training complete.")    
    
#     python train/train_script.py \
# --resume_ckpt ./checkpoints/dit_ckpt_epoch_40.pt \
# --epochs 80 \
# --img_size 32 \
# --batch_size 8 

# python train/train_script.py \
# --resume_ckpt ./checkpoints/dit_ckpt_epoch_20.pt \
# --epochs 80 \
# --learning_rate 1e-5 \
# --img_size 32 \
# --batch_size 8 \
# --embed_dim 128 \
# --depth 2

# python train/train_script.py \
# --resume_ckpt ./checkpoints/dit_ckpt_epoch_40_fixed.pt \
# --epochs 350 \
# --learning_rate 3e-6 \
# --img_size 32 \
# --batch_size 8 \
# --embed_dim 256
# --depth 4