#!/usr/bin/env python3
"""
Utility script to extract and save face embeddings from child images.
This allows for preprocessing child embeddings before training.
"""

import os
import torch
import argparse
from tqdm import tqdm
from PIL import Image
import pandas as pd
from model import LatentWeightTrainer
from differentiable_pipeline import DifferentiableFaceEncoder

def parse_args():
    parser = argparse.ArgumentParser(description='Extract and save face embeddings')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to family data CSV')
    parser.add_argument('--base_path', type=str, required=True, help='Base path for images')
    parser.add_argument('--output_dir', type=str, default='./embeddings', help='Output directory')
    parser.add_argument('--output_name', type=str, default='child_embeddings.pt', help='Output filename')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)
    
    # Load family data from CSV
    print(f"Loading family data from {args.csv_path}...")
    df = pd.read_csv(args.csv_path)
    
    # Get child image paths
    child_images = []
    for _, row in df.iterrows():
        try:
            # Extract family paths from CSV
            family_id = row['family_id']
            child_id = row['child_id']
            
            # Construct child image path
            child_img_path = os.path.join(args.base_path, f"{family_id}/{child_id}.jpg")
            
            if os.path.exists(child_img_path):
                child_images.append(child_img_path)
            else:
                print(f"Warning: Child image not found: {child_img_path}")
                child_images.append(None)
        except Exception as e:
            print(f"Error processing row {row}: {e}")
            child_images.append(None)
    
    print(f"Found {len([x for x in child_images if x is not None])} valid child images")
    
    # Initialize the trainer with minimal components needed for embedding extraction
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = LatentWeightTrainer(
        processor=None,  # We don't need the processor for embedding extraction
        latent_shape=(18, 512),  # Shape doesn't matter for extraction
        learning_rate=0.0001,
        save_dir='./embeddings',
        device=device
    )
    
    # Extract and save embeddings
    print(f"Extracting embeddings and saving to {output_path}...")
    trainer.extract_face_embeddings(child_images, save_path=output_path)
    
    print("Done!")

if __name__ == "__main__":
    main() 