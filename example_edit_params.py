#!/usr/bin/env python3
# Example script demonstrating how to use both weight-based and edit-parameter-based models

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
from model import LatentWeightTrainer, LatentWeightGenerator, EditParamGenerator
from e4e_lib import E4EProcessor

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_family_data(data_dir):
    """
    Load family data from a directory structure:
    data_dir/
        family_1/
            father.jpg
            mother.jpg
            child.jpg
        family_2/
            ...
    
    Returns:
        tuple: (father_images, mother_images, child_images, family_indices)
    """
    father_images = []
    mother_images = []
    child_images = []
    family_indices = []
    
    # Iterate through family directories
    for i, family_dir in enumerate(sorted(os.listdir(data_dir))):
        family_path = os.path.join(data_dir, family_dir)
        if not os.path.isdir(family_path):
            continue
            
        # Check if this family has all required images
        father_path = os.path.join(family_path, 'father.jpg')
        mother_path = os.path.join(family_path, 'mother.jpg')
        child_path = os.path.join(family_path, 'child.jpg')
        
        if os.path.exists(father_path) and os.path.exists(mother_path) and os.path.exists(child_path):
            father_images.append(father_path)
            mother_images.append(mother_path)
            child_images.append(child_path)
            family_indices.append(i)
    
    print(f"Loaded {len(family_indices)} valid families")
    return father_images, mother_images, child_images, family_indices

def extract_latents(processor, image_paths, cache_dir='cache'):
    """
    Extract latent codes from a list of images.
    
    Args:
        processor: E4E processor
        image_paths: List of image paths
        cache_dir: Directory to cache latent codes
        
    Returns:
        list: List of latent codes
    """
    os.makedirs(cache_dir, exist_ok=True)
    latents = []
    
    for i, image_path in enumerate(tqdm(image_paths, desc="Extracting latents")):
        # Create cached filename
        basename = os.path.basename(os.path.dirname(image_path)) + '_' + os.path.basename(image_path)
        cache_path = os.path.join(cache_dir, f"{basename.replace('.jpg', '.pt')}")
        
        # Check if latent is already cached
        if os.path.exists(cache_path):
            print(f"Loading cached latent for {image_path}")
            latent = torch.load(cache_path)
        else:
            # Process image and extract latent
            try:
                result = processor.process_image(image_path)
                latent = result['latent']
                
                # Cache the latent
                torch.save(latent, cache_path)
                print(f"Saved latent to {cache_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                latent = None
                
        latents.append(latent)
    
    return latents

def compare_models(processor, father_latent, mother_latent, weight_trainer, edit_trainer, output_dir='comparison'):
    """
    Compare outputs from weight-based and edit-parameter-based models.
    
    Args:
        processor: E4E processor
        father_latent: Father's latent code
        mother_latent: Mother's latent code
        weight_trainer: Trained weight model trainer
        edit_trainer: Trained edit parameter model trainer
        output_dir: Directory to save comparison images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate father and mother images for reference
    father_img = processor.combiner.generate_from_latent(father_latent)
    mother_img = processor.combiner.generate_from_latent(mother_latent)
    
    # Generate child image using weight-based model
    weight_child_img = weight_trainer.generate_child_image(father_latent, mother_latent)
    
    # Generate child image using edit-parameter-based model
    edit_child_img = edit_trainer.generate_child_image(father_latent, mother_latent)
    
    # Get edit parameters for visualization
    edit_params = edit_trainer.predict_edit_parameters(father_latent, mother_latent)
    
    # Create comparison figure
    fig = plt.figure(figsize=(15, 10))
    
    # Add images
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(np.array(father_img))
    ax1.set_title('Father')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(np.array(mother_img))
    ax2.set_title('Mother')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(2, 3, 4)
    ax3.imshow(np.array(weight_child_img))
    ax3.set_title('Weight-based Child')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(2, 3, 5)
    ax4.imshow(np.array(edit_child_img))
    ax4.set_title('Edit-parameter-based Child')
    ax4.axis('off')
    
    # Add edit parameters visualization
    ax5 = fig.add_subplot(2, 3, 3)
    directions = list(edit_params.keys())
    values = list(edit_params.values())
    bars = ax5.bar(directions, values, color=['skyblue', 'lightgreen', 'salmon'])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        offset = 0.1 if height >= 0 else -0.3
        ax5.text(bar.get_x() + bar.get_width()/2., height + offset,
               f'{height:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    ax5.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax5.set_ylim(-3.5, 3.5)
    ax5.set_title('Edit Parameters')
    ax5.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    timestamp = int(time.time())
    plt.savefig(os.path.join(output_dir, f'comparison_{timestamp}.png'), bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison to {output_dir}/comparison_{timestamp}.png")
    
    # Also save individual images
    father_img.save(os.path.join(output_dir, f'father_{timestamp}.jpg'))
    mother_img.save(os.path.join(output_dir, f'mother_{timestamp}.jpg'))
    weight_child_img.save(os.path.join(output_dir, f'weight_child_{timestamp}.jpg'))
    edit_child_img.save(os.path.join(output_dir, f'edit_child_{timestamp}.jpg'))
    
    return {
        'father_img': father_img,
        'mother_img': mother_img,
        'weight_child_img': weight_child_img,
        'edit_child_img': edit_child_img,
        'edit_params': edit_params
    }

def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    # Configuration
    data_dir = 'family_data'
    cache_dir = 'latent_cache'
    weight_model_dir = 'weight_model'
    edit_model_dir = 'edit_model'
    comparison_dir = 'comparison_results'
    batch_size = 8
    num_epochs = 30
    
    # Create directories
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(weight_model_dir, exist_ok=True)
    os.makedirs(edit_model_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize E4E processor
    print("Initializing E4E processor...")
    processor = E4EProcessor(experiment_type='ffhq_encode')
    
    # Load family data
    print("Loading family data...")
    father_images, mother_images, child_images, family_indices = load_family_data(data_dir)
    
    # Split data into train and test sets
    train_indices, test_indices = train_test_split(family_indices, test_size=0.2, random_state=42)
    print(f"Split data into {len(train_indices)} training and {len(test_indices)} testing families")
    
    # Extract latent codes
    print("Extracting latent codes...")
    father_latents = extract_latents(processor, father_images, cache_dir=os.path.join(cache_dir, 'father'))
    mother_latents = extract_latents(processor, mother_images, cache_dir=os.path.join(cache_dir, 'mother'))
    
    # Initialize the trainers for both models
    weight_trainer = LatentWeightTrainer(
        processor=processor,
        model_type='weight',
        latent_shape=(18, 512),
        learning_rate=0.00005,
        device=device,
        save_dir=weight_model_dir
    )
    
    edit_trainer = LatentWeightTrainer(
        processor=processor,
        model_type='edit',
        latent_shape=(18, 512),
        learning_rate=0.00005,
        device=device,
        save_dir=edit_model_dir
    )
    
    # Extract child face embeddings (shared between both models)
    print("Extracting child face embeddings...")
    embeddings_save_path = os.path.join(cache_dir, 'child_embeddings.pt')
    
    # Train the weight-based model
    print("\n===== Training Weight-Based Model =====")
    weight_model, weight_history = weight_trainer.train(
        father_latents=father_latents,
        mother_latents=mother_latents,
        child_images=child_images,
        train_indices=train_indices,
        test_indices=test_indices,
        num_epochs=num_epochs,
        batch_size=batch_size,
        embeddings_save_path=embeddings_save_path
    )
    
    # Train the edit-parameter-based model
    print("\n===== Training Edit-Parameter-Based Model =====")
    edit_model, edit_history = edit_trainer.train(
        father_latents=father_latents,
        mother_latents=mother_latents,
        child_images=child_images,
        train_indices=train_indices,
        test_indices=test_indices,
        num_epochs=num_epochs,
        batch_size=batch_size,
        load_embeddings_from=embeddings_save_path  # Reuse embeddings extracted for the weight model
    )
    
    # Compare the results of both models on test samples
    print("\n===== Comparing Models =====")
    for idx in test_indices[:3]:  # Compare first 3 test samples
        if father_latents[idx] is not None and mother_latents[idx] is not None:
            print(f"Comparing models on test family {idx}...")
            results = compare_models(
                processor=processor,
                father_latent=father_latents[idx],
                mother_latent=mother_latents[idx],
                weight_trainer=weight_trainer,
                edit_trainer=edit_trainer,
                output_dir=comparison_dir
            )
    
    # Cleanup
    processor.cleanup()
    print("\nDone!")

if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    print(f"Total execution time: {time.time() - start_time:.2f} seconds") 