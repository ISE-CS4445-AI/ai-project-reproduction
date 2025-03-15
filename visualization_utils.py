import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
import os
from datetime import datetime

def plot_training_history(train_losses, val_losses, save_dir='plots'):
    """Plot training and validation loss history."""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

def visualize_family_generation(father_img, mother_img, generated_child, actual_child=None, save_path=None):
    """Visualize a family's images including generated child."""
    num_images = 4 if actual_child is not None else 3
    plt.figure(figsize=(5 * num_images, 5))
    
    plt.subplot(1, num_images, 1)
    plt.imshow(father_img)
    plt.title("Father")
    
    plt.subplot(1, num_images, 2)
    plt.imshow(mother_img)
    plt.title("Mother")
    
    plt.subplot(1, num_images, 3)
    plt.imshow(generated_child)
    plt.title("Generated Child")
    
    if actual_child is not None:
        plt.subplot(1, num_images, 4)
        plt.imshow(actual_child)
        plt.title("Actual Child")
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_weight_analysis(model_weights, save_dir='plots'):
    """Analyze and visualize model weights."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot full heatmap
    plt.figure(figsize=(14, 8))
    plt.imshow(model_weights.cpu().numpy(), cmap='coolwarm', vmin=0, vmax=1)
    plt.colorbar(label="Weight Value (0=mother, 1=father)")
    plt.title("Model-Learned Parent Weights Across Layers and Dimensions")
    plt.xlabel("Latent Dimension (0-511)")
    plt.ylabel("StyleGAN Layer (0=coarse, 17=fine)")
    plt.savefig(os.path.join(save_dir, 'weight_heatmap.png'))
    plt.close()
    
    # Plot weight distribution
    plt.figure(figsize=(12, 6))
    plt.hist(model_weights.cpu().flatten().numpy(), bins=50, alpha=0.7)
    plt.axvline(x=0.5, color='red', linestyle='--', label='Equal Weight (0.5)')
    plt.title("Distribution of Model-Learned Weights")
    plt.xlabel("Weight Value (0=mother, 1=father)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'weight_distribution.png'))
    plt.close()

def analyze_weights_across_samples(weights_list, save_dir='plots'):
    """Analyze weights across multiple samples."""
    os.makedirs(save_dir, exist_ok=True)
    stacked_weights = torch.stack(weights_list)
    avg_weights = stacked_weights.mean(dim=0)
    weight_variance = stacked_weights.var(dim=0)
    
    # Plot average weights
    plt.figure(figsize=(14, 8))
    plt.imshow(avg_weights.cpu().numpy(), cmap='coolwarm', vmin=0, vmax=1)
    plt.colorbar(label="Average Weight (0=mother, 1=father)")
    plt.title(f"Average Weights Across {len(weights_list)} Samples")
    plt.xlabel("Latent Dimension")
    plt.ylabel("StyleGAN Layer")
    plt.savefig(os.path.join(save_dir, 'avg_weights_across_samples.png'))
    plt.close()
    
    # Plot weight variance
    plt.figure(figsize=(14, 8))
    plt.imshow(weight_variance.cpu().numpy(), cmap='viridis')
    plt.colorbar(label="Weight Variance Across Samples")
    plt.title(f"Weight Consistency Across {len(weights_list)} Samples")
    plt.xlabel("Latent Dimension")
    plt.ylabel("StyleGAN Layer")
    plt.savefig(os.path.join(save_dir, 'weight_variance.png'))
    plt.close()
    
    return avg_weights, weight_variance

def plot_layer_weights(layer_weights, save_dir='plots'):
    """Plot average weights per StyleGAN layer."""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(layer_weights, marker='o', linestyle='-', color='blue')
    plt.axhline(y=0.5, color='red', linestyle='--', label='Equal Weight (0.5)')
    plt.title("Average Weight per StyleGAN Layer")
    plt.xlabel("StyleGAN Layer")
    plt.ylabel("Average Weight (0=mother, 1=father)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'layer_weights.png'))
    plt.close() 