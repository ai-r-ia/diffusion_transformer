import argparse
import logging
import os
import time

def get_logger(name: str, log_dir: str) -> logging.Logger:
    
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

    
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DiT Training Configuration")

    parser.add_argument(
        "--epochs", default=50, type=int, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--batch_size", default=32, type=int, help="Batch size for training."
    )
    parser.add_argument(
        "--learning_rate", default=1e-4, type=float, help="Learning rate for Adam optimizer."
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="Random seed for reproducibility."
    )

    parser.add_argument(
        "--img_size", default=64, type=int, help="Image resolution to train on (H=W)."
    )
    parser.add_argument(
        "--data_root", 
        default="/Users/riash/Downloads/archive", 
        type=str, 
        help="Root directory of the dataset."
    )
    parser.add_argument(
        "--log_dir", 
        default="./logs/dit_train", 
        type=str, 
        help="Directory to save logs."
    )
    parser.add_argument(
        "--ckpt_dir", 
        default="./checkpoints", 
        type=str, 
        help="Directory to save model checkpoints."
    )
    parser.add_argument(
    "--resume_ckpt",
    default=None, #'./checkpoints/dit_ckpt_epoch_250.pt'
    type=str,
    help="Path to a checkpoint file to resume training from.",
)
    
    parser.add_argument("--patch_size", default=4, type=int, help="Size of ViT patches.")
    parser.add_argument("--embed_dim", default=256, type=int, help="Transformer token embedding dimension.")
    parser.add_argument("--num_heads", default=4, type=int, help="Number of attention heads.")
    parser.add_argument("--depth", default=4, type=int, help="Number of DiT blocks (depth of transformer).")
    parser.add_argument("--time_steps", default=1000, type=int, help="Number of diffusion time steps (T).")
    parser.add_argument("--num_classes", default=7, type=int, help="Number of classes in dataset.")
    
    return parser