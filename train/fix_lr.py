# import torch
# import os


# CHECKPOINT_PATH = './checkpoints/dit_ckpt_epoch_40.pt'  

# NEW_LEARNING_RATE = 1e-5                               

# OUTPUT_PATH = './checkpoints/dit_ckpt_epoch_40_fixed.pt' 


# def fix_checkpoint_lr(ckpt_path, new_lr, output_path):
    
#     if not os.path.exists(ckpt_path):
#         print(f"Error: Checkpoint file not found at {ckpt_path}")
#         return

#     checkpoint = torch.load(ckpt_path, map_location='cpu')
    
#     for group in checkpoint['optimizer_state_dict']['param_groups']:
#         old_lr = group['lr']
#         group['lr'] = new_lr
#         print(f"   -> LR changed from {old_lr:.6e} to {new_lr:.6e}")

#     torch.save(checkpoint, output_path)

# if __name__ == '__main__':
#     os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
#     fix_checkpoint_lr(CHECKPOINT_PATH, NEW_LEARNING_RATE, OUTPUT_PATH)
    
    
import os
from PIL import Image

INPUT_DIR = '/Users/riash/Downloads/archive/'
OUTPUT_DIR = '/Users/riash/Desktop/dit/fid_data/real_images_128x128/'
TARGET_SIZE = (128, 128)


os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Starting resize from {INPUT_DIR} to {OUTPUT_DIR}...")

for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        try:
            img = Image.open(input_path).convert('RGB') 
            img_resized = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            img_resized.save(output_path)
        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")

print("Resizing complete.")