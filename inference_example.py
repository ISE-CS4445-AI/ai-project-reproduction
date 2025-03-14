#!/usr/bin/env python3
# Example script demonstrating how to use a pre-trained EditParamGenerator model for inference

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from model import LatentWeightTrainer
from e4e_lib import E4EProcessor

def visualize_edit_parameters(edit_params, output_path=None):
    """Visualize edit parameters in a bar chart"""
    plt.figure(figsize=(10, 6))
    directions = list(edit_params.keys())
    values = list(edit_params.values())
    
    bars = plt.bar(directions, values, color=['skyblue', 'lightgreen', 'salmon'])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        offset = 0.1 if height >= 0 else -0.3
        plt.text(bar.get_x() + bar.get_width()/2., height + offset,
               f'{height:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.ylim(-3.5, 3.5)
    plt.title('Predicted Edit Parameters')
    plt.ylabel('Parameter Value (-3 to +3)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved edit parameters visualization to {output_path}")
    
    plt.show()

def main():
    # Configuration
    model_path = 'edit_model/best_model.pt'  # Path to pre-trained model
    output_dir = 'inference_results'
    father_image = 'test_images/father.jpg'
    mother_image = 'test_images/mother.jpg'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize E4E processor
    print("Initializing E4E processor...")
    processor = E4EProcessor(experiment_type='ffhq_encode')
    
    # Initialize trainer with edit-parameter model type
    trainer = LatentWeightTrainer(
        processor=processor,
        model_type='edit',  # Specify 'edit' model type
        latent_shape=(18, 512),
        device=device,
        save_dir='edit_model'
    )
    
    # Load pre-trained model
    print(f"Loading pre-trained model from {model_path}...")
    trainer.load_model(os.path.basename(model_path))
    
    # Process input images to get latent codes
    print(f"Processing father image: {father_image}")
    father_result = processor.process_image(father_image)
    father_latent = father_result['latent']
    
    print(f"Processing mother image: {mother_image}")
    mother_result = processor.process_image(mother_image)
    mother_latent = mother_result['latent']
    
    # Get edit parameters from the model
    print("Predicting edit parameters...")
    edit_params = trainer.predict_edit_parameters(father_latent, mother_latent)
    print("Predicted edit parameters:")
    for direction, value in edit_params.items():
        print(f"  {direction}: {value:.4f}")
    
    # Visualize edit parameters
    visualize_edit_parameters(
        edit_params,
        output_path=os.path.join(output_dir, 'edit_parameters.png')
    )
    
    # Generate child image
    print("Generating child image using edit parameters...")
    child_img = trainer.generate_child_image(father_latent, mother_latent)
    
    # Save input and output images
    father_img = processor.combiner.generate_from_latent(father_latent)
    mother_img = processor.combiner.generate_from_latent(mother_latent)
    
    father_img.save(os.path.join(output_dir, 'father.jpg'))
    mother_img.save(os.path.join(output_dir, 'mother.jpg'))
    child_img.save(os.path.join(output_dir, 'child.jpg'))
    
    # Display images in a grid
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(np.array(father_img))
    plt.title('Father')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(np.array(mother_img))
    plt.title('Mother')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(np.array(child_img))
    plt.title('Generated Child')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'family_result.png'))
    plt.show()
    
    # Cleanup
    processor.cleanup()
    print("\nDone!")

if __name__ == "__main__":
    main() 