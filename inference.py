import os
import sys
import argparse
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

# Add encoder4editing to the Python path
sys.path.append(os.path.abspath('encoder4editing'))

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
    
    # Create a mutually exclusive group for the input methods
    input_group = parser.add_mutually_exclusive_group()
    
    # Parent images as positional arguments (normal mode)
    input_group.add_argument('--parent-images', type=str, nargs=2, metavar=('parent1', 'parent2'),
                        help='Paths to the two parent images (normal mode)')
    
    # Web application mode
    input_group.add_argument('--webapp', action='store_true', 
                        help='Run as a web application with UI for uploading images')
    
    # Optional arguments
    parser.add_argument('--output', type=str, default='outputs/inference', 
                        help='Directory to save the generated child image')
    parser.add_argument('--model', type=str, default='family_models/content/family_models/best_model.pt',
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
    parser.add_argument('--make-younger', action='store_true',
                        help='Apply the young edit to make the output image look younger')
    parser.add_argument('--young-factor', type=float, default=-2.5,
                        help='Factor for the young edit (negative values make younger, positive make older)')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port for the web application')
    
    return parser.parse_args()

def check_inputs(args):
    """Validate input arguments."""
    # Skip validation for webapp mode
    if args.webapp:
        return True
        
    # Check if parent images exist
    if args.parent_images is None:
        logger.error("No parent images specified. Use --parent-images or --webapp")
        return False
        
    parent1 = args.parent_images[0]
    parent2 = args.parent_images[1]
    
    if not os.path.isfile(parent1):
        logger.error(f"Parent image 1 not found: {parent1}")
        return False
    
    if not os.path.isfile(parent2):
        logger.error(f"Parent image 2 not found: {parent2}")
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

def generate_child(parent1_path, parent2_path, args):
    """Generate a child image from two parent images."""
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
    logger.info(f"Processing parent image 1: {parent1_path}")
    _, parent1_latent, _ = processor.process_image(parent1_path)
    
    logger.info(f"Processing parent image 2: {parent2_path}")
    _, parent2_latent, _ = processor.process_image(parent2_path)
    
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
    parent1_img = Image.open(parent1_path).convert('RGB')
    parent2_img = Image.open(parent2_path).convert('RGB')
    
    # Apply young edit if requested
    if args.make_younger:
        logger.info(f"Applying young edit with factor {args.young_factor}...")
        try:
            # Use the editor component of the processor with 'age' direction
            edited_image = processor.editor.apply_interfacegan(
                child_latent,
                direction_name='age',  # Use 'age' instead of 'young'
                factor=args.young_factor  # Negative factor makes the image younger
            )
            child_image = edited_image
            logger.info("Young edit applied successfully")
        except Exception as e:
            logger.error(f"Error applying young edit: {e}")
            logger.warning("Proceeding with original image")
            logger.info("Available directions: age, smile, pose")
    
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
    
    # Visualize weights if requested
    if args.visualize_weights:
        weights_path = os.path.join(args.output, f"weights_{timestamp}.png")
        weight_fig = visualize_weights(weights, weights_path)
    
    return parent1_img, parent2_img, child_image, child_image_path, weights

def run_webapp(args):
    """Run the web application for face generation."""
    from flask import Flask, request, render_template, url_for, redirect, flash, send_from_directory
    import io
    import base64
    
    # Create the Flask app
    app = Flask(__name__)
    app.secret_key = os.urandom(24)
    
    # Create upload directory if it doesn't exist
    upload_dir = os.path.join(args.output, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    # Create templates directory if it doesn't exist
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create the HTML template if it doesn't exist
    template_path = os.path.join(templates_dir, "index.html")
    if not os.path.exists(template_path):
        with open(template_path, 'w') as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Genetic Face Generator</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
        }
        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .upload-form {
            background-color: #f5f5f5;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
            width: 100%;
            max-width: 800px;
        }
        .input-group {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        .input-section {
            width: 48%;
        }
        .preview-section {
            display: flex;
            justify-content: space-between;
            width: 100%;
            max-width: 1000px;
        }
        .preview-container {
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 10px;
            width: 30%;
            text-align: center;
        }
        .preview-image {
            max-width: 100%;
            max-height: 300px;
            margin-bottom: 10px;
        }
        .buttons {
            margin-top: 20px;
            text-align: center;
        }
        .btn {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .btn:hover {
            background-color: #45a049;
        }
        .file-input {
            margin-top: 10px;
            width: 100%;
        }
        .options {
            margin-top: 20px;
        }
        .flash-messages {
            margin-bottom: 20px;
        }
        .flash-message {
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .flash-error {
            background-color: #ffcccc;
            color: #cc0000;
        }
        .flash-success {
            background-color: #ccffcc;
            color: #006600;
        }
    </style>
</head>
<body>
    <h1>Genetic Face Generator</h1>
    
    <div class="container">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                <div class="flash-messages">
                    {% for category, message in messages %}
                        <div class="flash-message flash-{{ category }}">{{ message }}</div>
                    {% endfor %}
                </div>
            {% endif %}
        {% endwith %}
        
        <div class="upload-form">
            <form method="POST" enctype="multipart/form-data" action="{{ url_for('upload') }}">
                <div class="input-group">
                    <div class="input-section">
                        <h3>Parent 1</h3>
                        <input type="file" name="parent1" class="file-input" accept="image/*" required>
                        {% if parent1_img %}
                            <img src="data:image/png;base64,{{ parent1_img }}" class="preview-image">
                        {% endif %}
                    </div>
                    
                    <div class="input-section">
                        <h3>Parent 2</h3>
                        <input type="file" name="parent2" class="file-input" accept="image/*" required>
                        {% if parent2_img %}
                            <img src="data:image/png;base64,{{ parent2_img }}" class="preview-image">
                        {% endif %}
                    </div>
                </div>
                
                <div class="options">
                    <h3>Options</h3>
                    <label>
                        <input type="checkbox" name="uniform_weights" {% if uniform_weights %}checked{% endif %}>
                        Use uniform 50/50 weights (simpler blend)
                    </label>
                    <br>
                    <label>
                        <input type="checkbox" name="make_younger" {% if make_younger %}checked{% endif %}>
                        Make the child look younger
                    </label>
                </div>
                
                <div class="buttons">
                    <button type="submit" class="btn">Upload & Generate</button>
                </div>
            </form>
        </div>
        
        {% if child_img %}
            <div class="preview-section">
                <div class="preview-container">
                    <h3>Parent 1</h3>
                    <img src="data:image/png;base64,{{ parent1_img }}" class="preview-image">
                </div>
                
                <div class="preview-container">
                    <h3>Parent 2</h3>
                    <img src="data:image/png;base64,{{ parent2_img }}" class="preview-image">
                </div>
                
                <div class="preview-container">
                    <h3>Generated Child</h3>
                    <img src="data:image/png;base64,{{ child_img }}" class="preview-image">
                    <a href="{{ url_for('download_child') }}" class="btn">Download</a>
                </div>
            </div>
        {% endif %}
    </div>
</body>
</html>
            """)
    
    # Store the current child image path for download
    current_child_path = [None]
    
    @app.route('/')
    def index():
        """Render the main page."""
        return render_template('index.html')
    
    @app.route('/upload', methods=['POST'])
    def upload():
        """Handle image uploads and generation."""
        try:
            # Check if files were provided
            if 'parent1' not in request.files or 'parent2' not in request.files:
                flash('Both parent images are required', 'error')
                return redirect(url_for('index'))
            
            parent1_file = request.files['parent1']
            parent2_file = request.files['parent2']
            
            # Check if files are valid
            if parent1_file.filename == '' or parent2_file.filename == '':
                flash('Both parent images are required', 'error')
                return redirect(url_for('index'))
            
            # Get form options
            uniform_weights = 'uniform_weights' in request.form
            make_younger = 'make_younger' in request.form
            
            # Save uploaded files temporarily
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            parent1_path = os.path.join(upload_dir, f"parent1_{timestamp}.png")
            parent2_path = os.path.join(upload_dir, f"parent2_{timestamp}.png")
            
            parent1_file.save(parent1_path)
            parent2_file.save(parent2_path)
            
            # Set options in args
            args.uniform_weights = uniform_weights
            args.make_younger = make_younger
            
            # Generate child
            parent1_img, parent2_img, child_img, child_path, _ = generate_child(parent1_path, parent2_path, args)
            
            # Convert images to base64 for display
            def img_to_base64(img):
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            parent1_b64 = img_to_base64(parent1_img)
            parent2_b64 = img_to_base64(parent2_img)
            child_b64 = img_to_base64(child_img)
            
            # Store current child path for download
            current_child_path[0] = child_path
            
            # Render template with images
            return render_template('index.html', 
                                 parent1_img=parent1_b64, 
                                 parent2_img=parent2_b64, 
                                 child_img=child_b64,
                                 uniform_weights=uniform_weights,
                                 make_younger=make_younger)
        
        except Exception as e:
            logger.error(f"Error in upload: {e}")
            flash(f'Error generating child: {str(e)}', 'error')
            return redirect(url_for('index'))
    
    @app.route('/download')
    def download_child():
        """Download the generated child image."""
        if current_child_path[0] is None:
            flash('No child image generated yet', 'error')
            return redirect(url_for('index'))
        
        directory = os.path.dirname(current_child_path[0])
        filename = os.path.basename(current_child_path[0])
        return send_from_directory(directory, filename, as_attachment=True)
    
    # Run the Flask app
    logger.info(f"Starting webapp on port {args.port}. Open http://localhost:{args.port} in your browser.")
    app.run(host='0.0.0.0', port=args.port, debug=False)

def main():
    """Main function for inference."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Run webapp if requested
    if args.webapp:
        logger.info("Starting web application mode")
        run_webapp(args)
        return
    
    # Validate inputs for normal mode
    if not check_inputs(args):
        sys.exit(1)
    
    # In normal mode, extract parent image paths
    parent1_path = args.parent_images[0]
    parent2_path = args.parent_images[1]
    
    # Generate child
    parent1_img, parent2_img, child_img, child_path, weights = generate_child(
        parent1_path, parent2_path, args
    )
    
    # Visualize results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    visualization_path = os.path.join(args.output, f"result_{timestamp}.png")
    fig = visualize_results(parent1_img, parent2_img, child_img, weights, visualization_path)
    
    # Display images if requested
    if args.display:
        plt.show()
    else:
        plt.close('all')
    
    logger.info("Inference completed successfully")
    
if __name__ == "__main__":
    main() 