import os
import torch
import argparse
from model import LatentWeightTrainer
from visualization_utils import (
    plot_weight_analysis,
    analyze_weights_across_samples,
    plot_layer_weights
)

def main():
    parser = argparse.ArgumentParser(description='Visualize model weights')
    parser.add_argument('--model', type=str, required=True, help='Path to the model weights')
    parser.add_argument('--output', type=str, default='plots', help='Output directory for visualizations')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load model weights
    print(f"Loading model from {args.model}")
    model_weights = torch.load(args.model, map_location='cpu')
    
    # Extract the weights tensor (adjust this based on your model structure)
    if isinstance(model_weights, dict) and 'model_state_dict' in model_weights:
        # If it's a checkpoint with state_dict
        weights = None
        for key, value in model_weights['model_state_dict'].items():
            if 'weight' in key and value.dim() == 2:
                weights = value
                break
    elif isinstance(model_weights, dict):
        # If it's just a state_dict
        weights = None
        for key, value in model_weights.items():
            if 'weight' in key and value.dim() == 2:
                weights = value
                break
    else:
        # If it's directly the weights tensor
        weights = model_weights
    
    if weights is None:
        print("Could not find weights in the model file")
        return
    
    # Generate visualizations
    print("Generating weight analysis plots...")
    plot_weight_analysis(weights, save_dir=args.output)
    
    # If you have multiple weight samples, you could analyze them
    # analyze_weights_across_samples([weights], save_dir=args.output)
    
    # Plot average weights per layer
    layer_weights = weights.mean(dim=1)  # Average across dimensions
    plot_layer_weights(layer_weights, save_dir=args.output)
    
    print(f"Visualizations saved to {args.output}")

if __name__ == "__main__":
    main()