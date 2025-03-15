import os
import sys
import argparse
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

# Import local modules
from model import LatentWeightTrainer
from e4e_lib import E4EProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments for inference."""
    parser = argparse.ArgumentParser(description='Genetic Face Generation - Inference Script')
    
    # Required arguments - parent images
    parser.add_argument('parent1', type=str, help='Path to the first parent image')
    parser.add_argument('parent2', type=str, help='Path to the second parent image')
    
    # Optional arguments
    parser.add_argument('--output', type=str, default='outputs/inference', 
                        help='Directory to save the generated child image')
    parser.add_argument('--model', type=str, default='family_models/latest_model.pt',
                        help='Path to the trained model weights')
    parser.add_argument('--uniform-weights', action='store_true', 
                        help='Use uniform 50/50 weights instead of model weights')
    parser.add_argument('--custom-weights', type=float, nargs='+',
                        help='Custom weight value(s) for parent1 (0.0-1.0). If a single value is provided, '
                             'it applies uniformly. Multiple values create a custom weight pattern.')
    parser.add_argument('--save-latents', action='store_true',
                        help='Save the generated latent codes')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use for inference (cpu, cuda, cuda:0, etc.)')
    parser.add_argument('--visualize-weights', action='store_true',
                        help='Visualize the weights used for blending')
    parser.add_argument('--display', action='store_true',
                        help='Display the generated image')
    
    return parser.parse_args()

def check_inputs(args):
    """Validate input arguments."""
    # Check if parent images exist
    if not os.path.isfile(args.parent1):
        logger.error(f"Parent image 1 not found: {args.parent1}")
        return False
    
    if not os.path.isfile(args.parent2):
        logger.error(f"Parent image 2 not found: {args.parent2}")
        return False
    
    # Check if model exists when not using uniform weights
    if not args.uniform_weights and not os.path.isfile(args.model):
        logger.error(f"Model file not found: {args.model}")
        logger.info("You can use --uniform-weights to generate without a model.")
        return False
    
    # Check custom weights
    if args.custom_weights:
        for w in args.custom_weights:
            if w < 0.0 or w > 1.0:
                logger.error(f"Custom weight value {w} is outside valid range [0.0-1.0]")
                return False
    
    return True

def create_uniform_weights(latent_shape):
    """Create uniform weights (50/50 blend)."""
    weights = torch.ones(latent_shape) * 0.5
    return weights

def create_custom_weights(latent_shape, weight_values):
    """Create custom weights based on provided values."""
    if len(weight_values) == 1:
        # Single value - apply uniformly
        weights = torch.ones(latent_shape) * weight_values[0]
    else:
        # Multiple values - create a pattern
        # Resize to match expected dimensions
        weights = torch.zeros(latent_shape)
        
        # If there are exactly as many weights as layers (18 for StyleGAN)
        if len(weight_values) == latent_shape[0]:
            for i, w in enumerate(weight_values):
                weights[i, :] = w
        else:
            # Interpolate to fit the number of layers
            import numpy as np
            from scipy.interpolate import interp1d
            
            x_original = np.linspace(0, 1, len(weight_values))
            x_target = np.linspace(0, 1, latent_shape[0])
            f = interp1d(x_original, weight_values, kind='linear')
            interpolated_weights = f(x_target)
            
            for i, w in enumerate(interpolated_weights):
                weights[i, :] = w
    
    return weights

def visualize_results(parent1_img, parent2_img, child_img, weights=None, save_path=None):
    """Visualize and save the results."""
    plt.figure(figsize=(15, 10))
    
    # Display parent images and generated child
    plt.subplot(1, 3, 1)
    plt.imshow(parent1_img)
    plt.title("Parent 1")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(parent2_img)
    plt.title("Parent 2")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(child_img)
    plt.title("Generated Child")
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the visualization if a path is provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved visualization to {save_path}")
    
    return plt.gcf()

def visualize_weights(weights, save_path=None):
    """Visualize the weights used for blending."""
    if weights.dim() != 2:
        weights = weights.reshape(-1, weights.size(-1))
    
    plt.figure(figsize=(12, 6))
    plt.imshow(weights.cpu().numpy(), cmap='viridis', aspect='auto')
    plt.colorbar(label='Weight for Parent 1')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Style Layer')
    plt.title('Weight Distribution Across Latent Space')
    
    # Add horizontal lines to separate style layers
    for i in range(1, weights.size(0)):
        plt.axhline(i - 0.5, color='white', linestyle='-', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved weight visualization to {save_path}")
    
    return plt.gcf()

def main():
    """Main function for inference."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Validate inputs
    if not check_inputs(args):
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize E4E processor for encoding images
    logger.info("Initializing E4E processor...")
    processor = E4EProcessor(
        experiment_type='ffhq_encode',
        memory_efficient=True,
        enable_mixed_precision=True,
        max_batch_size=1
    )
    
    # Process parent images to get latent codes
    logger.info(f"Processing parent image 1: {args.parent1}")
    _, parent1_latent, _ = processor.process_image(args.parent1)
    
    logger.info(f"Processing parent image 2: {args.parent2}")
    _, parent2_latent, _ = processor.process_image(args.parent2)
    
    # Initialize the LatentWeightTrainer (no need to train, just for inference)
    latent_shape = parent1_latent.shape
    trainer = LatentWeightTrainer(
        processor=processor,
        latent_shape=latent_shape,
        device=device
    )
    
    # Load model if not using uniform weights
    if not args.uniform_weights and not args.custom_weights:
        logger.info(f"Loading model from {args.model}")
        try:
            trainer.load_model(args.model)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Falling back to uniform weights")
            args.uniform_weights = True
    
    # Generate child latent
    if args.uniform_weights:
        logger.info("Using uniform 50/50 weights for blending")
        weights = create_uniform_weights(latent_shape)
        weights = weights.to(device)
        
        # Move latents to the right device
        parent1_latent = parent1_latent.to(device)
        parent2_latent = parent2_latent.to(device)
        
        # Combine latents manually
        child_latent = trainer._combine_latents_with_weights(
            parent1_latent, parent2_latent, weights
        )
    elif args.custom_weights:
        logger.info(f"Using custom weights: {args.custom_weights}")
        weights = create_custom_weights(latent_shape, args.custom_weights)
        weights = weights.to(device)
        
        # Move latents to the right device
        parent1_latent = parent1_latent.to(device)
        parent2_latent = parent2_latent.to(device)
        
        # Combine latents manually
        child_latent = trainer._combine_latents_with_weights(
            parent1_latent, parent2_latent, weights
        )
    else:
        logger.info("Using trained model to predict weights")
        child_latent = trainer.generate_child_latent(parent1_latent, parent2_latent)
        # Get the weights for visualization
        weights = trainer.predict_weights(parent1_latent, parent2_latent)
    
    # Generate child image
    logger.info("Generating child image")
    child_image = trainer.generate_child_image(parent1_latent, parent2_latent)
    
    # Get parent images as PIL Images
    parent1_img = Image.open(args.parent1).convert('RGB')
    parent2_img = Image.open(args.parent2).convert('RGB')
    
    # Save the child image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    child_image_path = os.path.join(args.output, f"child_{timestamp}.png")
    child_image.save(child_image_path)
    logger.info(f"Saved child image to {child_image_path}")
    
    # Save latents if requested
    if args.save_latents:
        latents_dir = os.path.join(args.output, "latents")
        os.makedirs(latents_dir, exist_ok=True)
        
        torch.save(parent1_latent.cpu(), os.path.join(latents_dir, f"parent1_{timestamp}.pt"))
        torch.save(parent2_latent.cpu(), os.path.join(latents_dir, f"parent2_{timestamp}.pt"))
        torch.save(child_latent.cpu(), os.path.join(latents_dir, f"child_{timestamp}.pt"))
        torch.save(weights.cpu(), os.path.join(latents_dir, f"weights_{timestamp}.pt"))
        
        logger.info(f"Saved latent codes to {latents_dir}")
    
    # Visualize results
    visualization_path = os.path.join(args.output, f"result_{timestamp}.png")
    fig = visualize_results(parent1_img, parent2_img, child_image, weights, visualization_path)
    
    # Visualize weights if requested
    if args.visualize_weights:
        weights_path = os.path.join(args.output, f"weights_{timestamp}.png")
        weight_fig = visualize_weights(weights, weights_path)
    
    # Display images if requested
    if args.display:
        plt.show()
    else:
        plt.close('all')
    
    logger.info("Inference completed successfully")
    
if __name__ == "__main__":
    main() 