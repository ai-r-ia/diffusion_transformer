import matplotlib.pyplot as plt
import pandas as pd
import re
import os

LOG_FILE_PATH = '/Users/riash/Desktop/dit/Diffusion-DiT-Assignment/logs/dit_train/DiT_Trainer_20250930-085107.log' 
OUTPUT_PLOT_PATH = 'training_loss_trend.png'


def parse_epoch_log(log_path):
    data = {'epoch': [], 'loss': []}
    
    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(r'Epoch\s+(\d+).*?Avg Loss:\s+([\d\.]+)', line)
            if match:
                data['epoch'].append(int(match.group(1)))
                data['loss'].append(float(match.group(2)))
                
    return pd.DataFrame(data)

try:
    df = parse_epoch_log(LOG_FILE_PATH)
except FileNotFoundError:
    print(f"Error: Log file not found at {LOG_FILE_PATH}. Cannot generate plot.")
    exit()

plt.figure(figsize=(10, 6))

plt.plot(df['epoch'], df['loss'], color='blue', linewidth=2, label='Average Epoch Loss')

plt.title(f'DiT Training Loss Trend (Depth=4, Embed=256)')
plt.xlabel('Epochs')
plt.ylabel('Average Loss (MSE)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.savefig(OUTPUT_PLOT_PATH)
print(f"Loss trend visualization saved to {OUTPUT_PLOT_PATH}")