import os
import glob
from PIL import Image

def create_progression_grid(input_dir, output_filename, grid_size=(10, 10), target_size=(128, 128)):    
    
    required_images = grid_size[0] * grid_size[1]
    image_paths = glob.glob(os.path.join(input_dir, f"grid_t*_n*.png"))
    
    if len(image_paths) != required_images:
        print(f"ERROR: Found {len(image_paths)} grid images. Need exactly {required_images}.")
        
        return

    def sort_key(path):
        filename = os.path.basename(path)
        
        try:
            sample_idx = int(filename.split('_n')[1].split('.')[0])
            timestep = int(filename.split('_t')[1].split('_')[0])
        
            return (sample_idx, -timestep) 
        except Exception as e:
            print(f"Skipping malformed filename {filename}: {e}")
            return (1000, 1000) 
        
    sorted_paths = sorted(image_paths, key=sort_key)
    
    grid_width = grid_size[0] * target_size[0]
    grid_height = grid_size[1] * target_size[1]
    
    grid_image = Image.new('RGB', (grid_width, grid_height))

    for i, path in enumerate(sorted_paths):
        img = Image.open(path)
                
        if img.size != target_size:
            img = img.resize(target_size)
            
        row = i // grid_size[0] 
        col = i % grid_size[0]  
        
        x_offset = col * target_size[0]
        y_offset = row * target_size[1]
        
        grid_image.paste(img, (x_offset, y_offset))

    
    grid_image.save(output_filename)
    print(f"\n--- SUCCESS ---")
    print(f"10x10 Progression Grid created successfully and saved to {output_filename}")

if __name__ == '__main__':
    
    INPUT_DIR = "./sample_grid" 
    TARGET_GRID_FILE = "DDIM_Progression_Grid_10x10.png"
    
    create_progression_grid(INPUT_DIR, TARGET_GRID_FILE)
        
