from model import LatentWeightTrainer
from e4e_lib import E4EProcessor
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import torch
import logging
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from visualization_utils import (
    plot_training_history,
    visualize_family_generation,
    plot_weight_analysis,
    analyze_weights_across_samples,
    plot_layer_weights
)

from data_loader import FamilyDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define utility functions
def get_image_from_path(image_path):
    """Load an image from the given path."""
    try:
        img = Image.open(image_path)
        return img
    except FileNotFoundError:
        logger.error(f"File not found: {image_path}")
        # Look for the file in alternative locations
        filename = os.path.basename(image_path)
        alt_locations = [
            os.path.join("AlignedTest2", filename),  # Try direct in AlignedTest2
            os.path.join("sample_images", "fathers", filename),
            os.path.join("sample_images", "mothers", filename),
            os.path.join("sample_images", "children", filename)
        ]
        
        for alt_path in alt_locations:
            if os.path.exists(alt_path):
                logger.info(f"Found alternative for {image_path} at {alt_path}")
                return Image.open(alt_path)
        
        logger.warning(f"No alternative found for {image_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        return None

def get_family(family_id, base_path, csv_path):
    """
    Retrieves family information for a given family ID.

    Args:
        family_id: The ID of the family.
        base_path: Base path for images
        csv_path: Path to the CSV file with family data

    Returns:
        A dictionary containing family information, or None if the family ID is not found.
    """
    try:
        df = pd.read_csv(csv_path)
        for list_col_name in ["mother_images", "father_images", "child_images"]:
            df[list_col_name] = df[list_col_name].map(eval)

        family_data = df[df['family_id'] == family_id].iloc[0]

        no_father_images = len(family_data['father_images'])
        no_mother_images = len(family_data['mother_images'])
        no_child_images = len(family_data['child_images'])

        no_of_images = min(no_father_images, no_mother_images, no_child_images)

        father_images = random.sample(family_data['father_images'], k=min(no_of_images, len(family_data['father_images'])))
        mother_images = random.sample(family_data['mother_images'], k=min(no_of_images, len(family_data['mother_images'])))
        child_images = random.sample(family_data['child_images'], k=min(no_of_images, len(family_data['child_images'])))

        # Prepend base path to image paths if they don't include it already
        father_images = [os.path.join(base_path, img) if not os.path.isabs(img) else img for img in father_images]
        mother_images = [os.path.join(base_path, img) if not os.path.isabs(img) else img for img in mother_images]
        child_images = [os.path.join(base_path, img) if not os.path.isabs(img) else img for img in child_images]

        family_info = {
            'father_images': father_images,
            'mother_images': mother_images,
            'child_images': child_images,
            'Father_name': family_data['father_name'],
            'Mother_name': family_data['mother_name'],
            'Child_name': family_data['child_name']
        }
        return family_info
    except IndexError:
        print(f"Family with ID {family_id} not found.")
        return None
    except Exception as e:
        print(f"Error loading family {family_id}: {e}")
        return None

# Default configuration
DEFAULT_CONFIG = {
    'base_path': "./AlignedTest2",  # Base path for images
    'csv_path': "./CSVs/checkpoint10.csv",  # Path to family data CSV
    'output_dir': './outputs',  # Output directory for generated images
    'e4e_base_dir': None,  # Let E4EProcessor find the model files automatically
    'latent_dir': './latents',  # Directory containing pre-computed latent codes
    'embeddings_dir': './embeddings',  # Directory for storing face embeddings
    'model_dir': 'family_models',
    'learning_rate': 0.00001,
    'num_epochs': 80,
    'batch_size': 4,
    'use_scheduler': True
}

def prepare_environment(config=None):
    """
    Prepare the environment by creating necessary directories.
    
    Args:
        config (dict): Configuration dictionary with paths
        
    Returns:
        dict: Updated configuration
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    # Create output directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['model_dir'], exist_ok=True)
    os.makedirs(config['embeddings_dir'], exist_ok=True)
    
    # Check if the paths exist and provide warnings if they don't
    for path_name, path in [('base_path', config['base_path']), ('csv_path', config['csv_path'])]:
        if not os.path.exists(path):
            logger.warning(f"Path '{path}' specified in {path_name} does not exist. Please check your configuration.")
    
    return config

def verify_family_images(father_images, mother_images, children_images):
    """
    Verify that all images in a family exist and are valid.
    
    Args:
        father_images (list): List of father image paths
        mother_images (list): List of mother image paths
        children_images (list): List of child image paths
        
    Returns:
        tuple: (is_valid, valid_indices) 
               is_valid: True if at least one set of images is valid
               valid_indices: List of indices for valid image sets
    """
    valid_indices = []
    
    # Find the minimum length of the three lists
    min_len = min(len(father_images), len(mother_images), len(children_images))
    
    for j in range(min_len):
        # Check all three images for this family set
        f_img = get_image_from_path(father_images[j]) 
        m_img = get_image_from_path(mother_images[j])
        c_img = get_image_from_path(children_images[j])
        
        # Only consider valid if all three images exist
        if f_img is not None and m_img is not None and c_img is not None:
            valid_indices.append(j)
    
    return len(valid_indices) > 0, valid_indices

def load_family_data(config):
    """
    Load family data from the CSV file or sample data.
    
    Args:
        config (dict): Configuration dictionary with paths
        
    Returns:
        tuple: (father_images, mother_images, child_images, train_indices, test_indices)
    """
    logger.info("Loading family data...")
    fathers = []
    mothers = []
    children = []
    valid_family_count = 0
    invalid_family_count = 0

    # Try to load from the CSV path
    try:
        # Attempt to load using pandas
        df = pd.read_csv(config['csv_path'])
        max_family_id = df['family_id'].max()
        
        logger.info(f"Found {max_family_id + 1} families in the CSV")
        
        for i in range(max_family_id + 1):
            family = get_family(i, config['base_path'], config['csv_path'])
            if family:
                father_images = family.get('father_images')
                mother_images = family.get('mother_images')
                children_images = family.get('child_images')

                # Verify and only add valid image sets
                is_valid, valid_indices = verify_family_images(father_images, mother_images, children_images)
                
                if is_valid:
                    valid_family_count += 1
                    # Only add the valid image sets
                    for j in valid_indices:
                        fathers.append(father_images[j])
                        mothers.append(mother_images[j])
                        children.append(children_images[j])
                else:
                    invalid_family_count += 1
                    logger.warning(f"Family {i} has no valid image sets, skipping...")
        
        logger.info(f"Loaded {len(fathers)} family image sets from {valid_family_count} valid families")
        if invalid_family_count > 0:
            logger.warning(f"Skipped {invalid_family_count} families with missing or invalid images")

    except FileNotFoundError:
        logger.warning(f"CSV file not found at '{config['csv_path']}'")
        logger.info("Using sample data instead...")
        
        # If CSV doesn't exist, create some sample data for testing
        sample_dir = "sample_images"
        if os.path.exists(sample_dir):
            sample_fathers = [f for f in os.listdir(os.path.join(sample_dir, "fathers")) 
                            if f.endswith(('.jpg', '.jpeg', '.png'))]
            sample_mothers = [f for f in os.listdir(os.path.join(sample_dir, "mothers")) 
                            if f.endswith(('.jpg', '.jpeg', '.png'))]
            sample_children = [f for f in os.listdir(os.path.join(sample_dir, "children")) 
                            if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            # Use the minimum number of images available
            num_samples = min(len(sample_fathers), len(sample_mothers), len(sample_children))
            
            fathers = [os.path.join(sample_dir, "fathers", f) for f in sample_fathers[:num_samples]]
            mothers = [os.path.join(sample_dir, "mothers", f) for f in sample_mothers[:num_samples]]
            children = [os.path.join(sample_dir, "children", f) for f in sample_children[:num_samples]]
            
            logger.info(f"Created {len(fathers)} sample family image sets")
        else:
            logger.warning("No sample directory found. Please provide valid data paths.")

    # If no data could be loaded, exit
    if len(fathers) == 0:
        logger.error("No family data could be loaded. Please check your paths.")
        return None, None, None, None, None

    # Create train-test split (80% train, 20% test)
    indices = np.arange(len(fathers))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

    logger.info(f"Training on {len(train_indices)} families, testing on {len(test_indices)} families")

    return fathers, mothers, children, train_indices, test_indices

def initialize_processor(config):
    """
    Initialize the E4E processor.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        E4EProcessor: Initialized processor
    """
    logger.info("Initializing E4E processor...")
    processor = E4EProcessor(
        experiment_type='ffhq_encode',
        base_dir=config['e4e_base_dir'],
        memory_efficient=False,
        enable_mixed_precision=False,
        max_batch_size=1
    )
    return processor

def load_latents(config, child_images):
    """
    Load pre-computed latent codes.
    
    Args:
        config (dict): Configuration dictionary
        child_images (list): List of child image paths to match length with
        
    Returns:
        tuple: (father_latents, mother_latents, train_indices, test_indices)
    """
    logger.info("Loading pre-computed latent codes...")
    father_latents = []
    mother_latents = []

    # Check if the latent directory exists
    if not os.path.exists(config['latent_dir']):
        logger.error(f"Latent directory '{config['latent_dir']}' does not exist. Please check the path.")
        return None, None, None, None

    # Count the number of available latent pairs
    num_latent_pairs = 0
    while os.path.exists(os.path.join(config['latent_dir'], f'father_latent_{num_latent_pairs}.pt')) and \
          os.path.exists(os.path.join(config['latent_dir'], f'mother_latent_{num_latent_pairs}.pt')):
        num_latent_pairs += 1

    logger.info(f"Found {num_latent_pairs} latent pairs in {config['latent_dir']}")

    # Load the latent codes
    for i in range(num_latent_pairs):
        father_latent_path = os.path.join(config['latent_dir'], f'father_latent_{i}.pt')
        mother_latent_path = os.path.join(config['latent_dir'], f'mother_latent_{i}.pt')
        
        try:
            try:
                # First try with weights_only=False for PyTorch 2.6+ compatibility
                father_latent = torch.load(father_latent_path, weights_only=False)
            except TypeError:
                # Fallback for older PyTorch versions
                father_latent = torch.load(father_latent_path)
            father_latents.append(father_latent)
        except Exception as e:
            logger.error(f"Error loading father latent {father_latent_path}: {e}")
            father_latents.append(None)
            
        try:
            try:
                # First try with weights_only=False for PyTorch 2.6+ compatibility
                mother_latent = torch.load(mother_latent_path, weights_only=False)
            except TypeError:
                # Fallback for older PyTorch versions
                mother_latent = torch.load(mother_latent_path)
            mother_latents.append(mother_latent)
        except Exception as e:
            logger.error(f"Error loading mother latent {mother_latent_path}: {e}")
            mother_latents.append(None)

    # Adjust if we have more latent codes than images or vice versa
    min_length = min(len(father_latents), len(mother_latents), len(child_images))
    father_latents = father_latents[:min_length]
    mother_latents = mother_latents[:min_length]
    child_images = child_images[:min_length]

    # Filter out any families with failed processing
    valid_indices = [i for i in range(len(father_latents))
                    if father_latents[i] is not None and mother_latents[i] is not None]
    logger.info(f"Successfully loaded {len(valid_indices)} out of {len(father_latents)} latent pairs")

    return father_latents, mother_latents, valid_indices

def run_training(config=None, processor=None):
    """
    Run the full training pipeline.
    
    Args:
        config (dict, optional): Configuration dictionary
        processor (E4EProcessor, optional): Pre-initialized processor to avoid loading models if not needed
        
    Returns:
        tuple: (model, history, trainer, data)
    """
    # Use default config if not provided
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    # Prepare environment
    config = prepare_environment(config)
    
    # Load family data
    father_images, mother_images, child_images, train_indices, test_indices = load_family_data(config)
    if father_images is None:
        return None, None, None, None
    
    # Initialize processor if not provided
    if processor is None:
        processor = initialize_processor(config)
    
    # Load latents
    father_latents, mother_latents, valid_indices = load_latents(config, child_images)
    if father_latents is None:
        return None, None, None, None
    
    # Update train and test indices to only include valid families
    train_indices = [i for i in train_indices if i < len(valid_indices) and i in valid_indices]
    test_indices = [i for i in test_indices if i < len(valid_indices) and i in valid_indices]

    logger.info(f"After filtering: Training on {len(train_indices)} families, testing on {len(test_indices)} families")

    # Initialize the trainer
    trainer = LatentWeightTrainer(
        processor=processor,
        latent_shape=(18, 512),
        learning_rate=config['learning_rate'],
        save_dir=config['model_dir']
    )

    # Train the model
    model, history = trainer.train(
        father_latents=father_latents,
        mother_latents=mother_latents,
        child_images=child_images,
        train_indices=train_indices,
        test_indices=test_indices,
        num_epochs=config['num_epochs'],
        batch_size=config['batch_size'],
        embeddings_save_path=os.path.join(config['embeddings_dir'], 'child_embeddings.pt'),
        load_embeddings_from=os.path.join(config['embeddings_dir'], 'child_embeddings.pt') 
                            if os.path.exists(os.path.join(config['embeddings_dir'], 'child_embeddings.pt')) 
                            else None
    )
    
    # Collect data for visualization
    data = {
        'father_images': father_images,
        'mother_images': mother_images,
        'child_images': child_images,
        'father_latents': father_latents,
        'mother_latents': mother_latents,
        'train_indices': train_indices,
        'test_indices': test_indices
    }
    
    return model, history, trainer, data

def generate_sample(trainer, data, test_idx=0, output_dir='outputs'):
    """
    Generate a sample child image using the trained model.
    
    Args:
        trainer (LatentWeightTrainer): Trained model
        data (dict): Data dictionary with latents and images
        test_idx (int): Index of the test family to use
        output_dir (str): Directory to save outputs
    
    Returns:
        PIL.Image: Generated child image
    """
    father_latent = data['father_latents'][test_idx]
    mother_latent = data['mother_latents'][test_idx]

    # Generate child image
    generated_child_image = trainer.generate_child_image(father_latent, mother_latent)

    # Display the result
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(get_image_from_path(data['father_images'][test_idx]))
    plt.title("Father")

    plt.subplot(1, 3, 2)
    plt.imshow(get_image_from_path(data['mother_images'][test_idx]))
    plt.title("Mother")

    plt.subplot(1, 3, 3)
    plt.imshow(generated_child_image)
    plt.title("Generated Child")
    
    # Save the result
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"sample_generation_{test_idx}.png"))
    
    # Save individual images
    generated_child_image.save(os.path.join(output_dir, f"generated_child_{test_idx}.png"))
    
    return generated_child_image

class TrainingManager:
    def __init__(self, config, load_models=False):
        self.config = config
        self.setup_directories()
        
        # Only initialize processor and other components if explicitly requested
        self.processor = None
        self.trainer = None
        self.data_loader = None
        
        if load_models:
            self.initialize_processor()
            self.initialize_trainer()
            self.initialize_data_loader()
        
    def setup_directories(self):
        """Create necessary directories."""
        os.makedirs(self.config['output_dir'], exist_ok=True)
        os.makedirs(self.config['model_dir'], exist_ok=True)
        os.makedirs(self.config['embeddings_dir'], exist_ok=True)
        
    def initialize_processor(self):
        """Initialize the E4E processor."""
        if self.processor is None:
            logger.info("Initializing E4E processor...")
            self.processor = E4EProcessor(
                experiment_type='ffhq_encode',
                base_dir=None,  # Let E4EProcessor find the model files automatically
                memory_efficient=False,
                enable_mixed_precision=False,
                max_batch_size=1
            )
        return self.processor
        
    def initialize_trainer(self):
        """Initialize the LatentWeightTrainer."""
        if self.trainer is None:
            if self.processor is None:
                self.initialize_processor()
                
            logger.info("Initializing trainer...")
            self.trainer = LatentWeightTrainer(
                processor=self.processor,
                latent_shape=(18, 512),
                learning_rate=self.config['learning_rate'],
                save_dir=self.config['model_dir']
            )
            
            # Set up learning rate scheduler if specified
            if self.config.get('use_scheduler', False):
                from torch.optim.lr_scheduler import ReduceLROnPlateau
                self.trainer.scheduler = ReduceLROnPlateau(
                    self.trainer.optimizer,
                    mode='min',
                    patience=15,
                    factor=0.5,
                    min_lr=1e-7,
                    verbose=True
                )
        return self.trainer
            
    def initialize_data_loader(self):
        """Initialize the family data loader."""
        if self.data_loader is None:
            logger.info("Initializing data loader...")
            self.data_loader = FamilyDataLoader(
                base_path=self.config['base_path'],
                csv_path=self.config['csv_path']
            )
        return self.data_loader
    
    def load_data(self):
        """Load and prepare all necessary data."""
        logger.info("Loading family data...")
        
        if self.data_loader is None:
            self.initialize_data_loader()
            
        # Load family images
        self.father_images, self.mother_images, self.child_images = \
            self.data_loader.load_all_families()
            
        # Load pre-computed latents
        self.father_latents, self.mother_latents = \
            self.data_loader.load_latents(self.config['latent_dir'])
            
        # Create train-test split
        indices = np.arange(len(self.father_latents))
        train_indices, test_indices = train_test_split(
            indices, test_size=0.2, random_state=42
        )
        
        # Filter valid indices
        valid_indices = [i for i in range(len(self.father_latents))
                        if self.father_latents[i] is not None 
                        and self.mother_latents[i] is not None]
        
        self.train_indices = [i for i in train_indices if i in valid_indices]
        self.test_indices = [i for i in test_indices if i in valid_indices]
        
        logger.info(f"Training on {len(self.train_indices)} families, "
                   f"testing on {len(self.test_indices)} families")
        
        return self.train_indices, self.test_indices
    
    def train(self):
        """Main training loop."""
        # Load all data
        train_indices, test_indices = self.load_data()
        
        # Ensure trainer is initialized
        if self.trainer is None:
            self.initialize_trainer()
        
        # Train the model
        logger.info("Starting training...")
        model, history = self.trainer.train(
            father_latents=self.father_latents,
            mother_latents=self.mother_latents,
            child_images=self.child_images,
            train_indices=train_indices,
            test_indices=test_indices,
            num_epochs=self.config['num_epochs'],
            batch_size=self.config['batch_size'],
            embeddings_save_path=os.path.join(
                self.config['embeddings_dir'], 'child_embeddings.pt'
            ),
            load_embeddings_from=os.path.join(
                self.config['embeddings_dir'], 'child_embeddings.pt'
            ) if os.path.exists(os.path.join(
                self.config['embeddings_dir'], 'child_embeddings.pt'
            )) else None
        )
        
        # Plot training history
        plot_training_history(
            history['train_losses'],
            history['val_losses'],
            save_dir=self.config['output_dir']
        )
        
        return model, history
    
    def generate_sample(self, father_idx, mother_idx, child_idx=None):
        """Generate a sample child image."""
        # Ensure data is loaded
        if not hasattr(self, 'father_latents') or self.father_latents is None:
            self.load_data()
            
        # Ensure trainer is initialized
        if self.trainer is None:
            self.initialize_trainer()
            
        father_latent = self.father_latents[father_idx]
        mother_latent = self.mother_latents[mother_idx]
        
        # Generate child image
        generated_child = self.trainer.generate_child_image(
            father_latent, mother_latent
        )
        
        # Visualize the result
        actual_child = None
        if child_idx is not None:
            actual_child = self.child_images[child_idx]
            
        visualize_family_generation(
            self.father_images[father_idx],
            self.mother_images[mother_idx],
            generated_child,
            actual_child,
            save_path=os.path.join(
                self.config['output_dir'],
                f'family_generation_{father_idx}_{mother_idx}.png'
            )
        )
        
        return generated_child

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Family Generator Training Script')
    
    parser.add_argument('--train', action='store_true', help='Run the training process')
    parser.add_argument('--generate', action='store_true', help='Generate samples after training')
    parser.add_argument('--num-samples', type=int, default=3, help='Number of samples to generate')
    parser.add_argument('--epochs', type=int, default=80, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--no-scheduler', action='store_true', help='Disable learning rate scheduler')
    
    # Path arguments
    parser.add_argument('--base-path', default='./AlignedTest2', help='Base path for images')
    parser.add_argument('--csv-path', default='./CSVs/checkpoint10.csv', help='Path to family data CSV')
    parser.add_argument('--output-dir', default='./outputs', help='Output directory')
    parser.add_argument('--latent-dir', default='./latents', help='Directory for latent codes')
    parser.add_argument('--embeddings-dir', default='./embeddings', help='Directory for embeddings')
    
    return parser.parse_args()

def main():
    """Main entry point when script is run directly."""
    args = parse_arguments()
    
    # Create config from arguments
    config = {
        'output_dir': args.output_dir,
        'model_dir': 'family_models',
        'embeddings_dir': args.embeddings_dir,
        'base_path': args.base_path,
        'csv_path': args.csv_path,
        'latent_dir': args.latent_dir,
        'learning_rate': args.learning_rate,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'use_scheduler': not args.no_scheduler
    }
    
    # If neither train nor generate is specified, run both
    if not args.train and not args.generate:
        args.train = True
        args.generate = True
    
    # Create manager - only load models if needed for training or generation
    manager = TrainingManager(config, load_models=args.train or args.generate)
    
    # Train if requested
    if args.train:
        model, history = manager.train()
    
    # Generate samples if requested
    if args.generate:
        for idx in range(min(args.num_samples, 3)):
            manager.generate_sample(idx, idx)

if __name__ == "__main__":
    main()