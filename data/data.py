import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import re 

class LandscapeDataset(Dataset):
    def __init__(self, root_dir: str, img_size: int):
        self.root_dir = root_dir
        
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, '**', '*.jpg'), recursive=True) + 
                                  glob.glob(os.path.join(root_dir, '**', '*.jpeg'), recursive=True))
        
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in the directory: {root_dir}.")
    
        self.labeled_data = []
        pattern = re.compile(r'_\((\d)\)\.jpe?g$', re.IGNORECASE)
        
        for path in self.image_paths:
            filename = os.path.basename(path)
            match = pattern.search(filename)
            
            label_int = -1 

            if match:                
                suffix_num = int(match.group(1))
                if 2 <= suffix_num <= 7:
                    label_int = suffix_num - 1 
            else:
                label_int = 0
            
            if label_int == -1:
                print(f"Warning: Skipping file with unhandled suffix: {filename}")
                continue

            self.labeled_data.append((path, label_int))

        self.num_classes = 7 

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size), 
            transforms.ToTensor(),           
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.labeled_data)

    def __getitem__(self, idx):
        img_path, label_int = self.labeled_data[idx]
        
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(label_int, dtype=torch.long)
        
        return self.transform(image), label