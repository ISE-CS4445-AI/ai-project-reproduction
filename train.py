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

# Define utility functions
def get_image_from_path(image_path):
    """Load an image from the given path."""
    try:
        img = Image.open(image_path)
        return img
    except FileNotFoundError:
        print(f"File not found: {image_path}")
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

# Configure paths - update these to match your system
BASE_PATH = "/content/drive/MyDrive/Child Generator/AlignedTest2"  # Base path for images
CSV_PATH = "/content/drive/MyDrive/Child Generator/CSVs/checkpoint10.csv"  # Path to family data CSV
OUTPUT_DIR = '/content/images/outputs'  # Output directory for generated images
E4E_BASE_DIR = './e4e'  # Base directory for E4E model
LATENT_DIR = './latents'  # Directory containing pre-computed latent codes

# Check if the paths exist and provide warnings if they don't
for path in [BASE_PATH, CSV_PATH]:
    if not os.path.exists(path):
        print(f"WARNING: Path '{path}' does not exist. Please update the path.")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('family_models', exist_ok=True)

# Load all family data
print("Loading family data...")
fathers = []
mothers = []
children = []

# Try to load from the CSV path
try:
    # Attempt to load using pandas
    df = pd.read_csv(CSV_PATH)
    max_family_id = df['family_id'].max()
    
    print(f"Found {max_family_id + 1} families in the CSV")
    
    for i in range(max_family_id + 1):
        family = get_family(i, BASE_PATH, CSV_PATH)
        if family:
            father_images = family.get('father_images')
            mother_images = family.get('mother_images')
            children_images = family.get('child_images')

            for j in range(min(len(father_images), len(mother_images), len(children_images))):
                fathers.append(father_images[j])
                mothers.append(mother_images[j])
                children.append(children_images[j])
    
    print(f"Loaded {len(fathers)} family image sets")

except FileNotFoundError:
    print(f"CSV file not found at '{CSV_PATH}'")
    print("Using sample data instead...")
    
    # If CSV doesn't exist, create some sample data for testing
    # This is just placeholder data for demonstration
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
        
        print(f"Created {len(fathers)} sample family image sets")
    else:
        print("No sample directory found. Please provide valid data paths.")

# If no data could be loaded, exit
if len(fathers) == 0:
    print("No family data could be loaded. Please check your paths.")
    exit(1)

# Create train-test split (80% train, 20% test)
indices = np.arange(len(fathers))
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

print(f"Training on {len(train_indices)} families, testing on {len(test_indices)} families")

# Store data for reference
father_images = fathers
mother_images = mothers
child_images = children

# Initialize the E4E processor
print("Initializing E4E processor...")
processor = E4EProcessor(
    experiment_type='ffhq_encode',
    base_dir=E4E_BASE_DIR,
    memory_efficient=False,
    enable_mixed_precision=False,
    max_batch_size=1
)

# Load pre-computed latent codes instead of processing images
print("Loading pre-computed latent codes...")
father_latents = []
mother_latents = []

# Check if the latent directory exists
if not os.path.exists(LATENT_DIR):
    print(f"WARNING: Latent directory '{LATENT_DIR}' does not exist. Please check the path.")
    exit(1)

# Count the number of available latent pairs
num_latent_pairs = 0
while os.path.exists(os.path.join(LATENT_DIR, f'father_latent_{num_latent_pairs}.pt')) and \
      os.path.exists(os.path.join(LATENT_DIR, f'mother_latent_{num_latent_pairs}.pt')):
    num_latent_pairs += 1

print(f"Found {num_latent_pairs} latent pairs in {LATENT_DIR}")

# Load the latent codes
for i in range(num_latent_pairs):
    father_latent_path = os.path.join(LATENT_DIR, f'father_latent_{i}.pt')
    mother_latent_path = os.path.join(LATENT_DIR, f'mother_latent_{i}.pt')
    
    try:
        father_latent = torch.load(father_latent_path)
        father_latents.append(father_latent)
    except Exception as e:
        print(f"Error loading father latent {father_latent_path}: {e}")
        father_latents.append(None)
        
    try:
        mother_latent = torch.load(mother_latent_path)
        mother_latents.append(mother_latent)
    except Exception as e:
        print(f"Error loading mother latent {mother_latent_path}: {e}")
        mother_latents.append(None)

# Adjust if we have more latent codes than images or vice versa
min_length = min(len(father_latents), len(mother_latents), len(child_images))
father_latents = father_latents[:min_length]
mother_latents = mother_latents[:min_length]
child_images = child_images[:min_length]

# Filter out any families with failed processing
valid_indices = [i for i in range(len(father_latents))
                if father_latents[i] is not None and mother_latents[i] is not None]
print(f"Successfully loaded {len(valid_indices)} out of {len(father_latents)} latent pairs")

# Update train and test indices to only include valid families
train_indices = [i for i in train_indices if i < len(valid_indices) and i in valid_indices]
test_indices = [i for i in test_indices if i < len(valid_indices) and i in valid_indices]

print(f"After filtering: Training on {len(train_indices)} families, testing on {len(test_indices)} families")

# Initialize the trainer
trainer = LatentWeightTrainer(
    processor=processor,  # Your E4E processor instance
    latent_shape=(18, 512),  # Shape of StyleGAN latent codes
    learning_rate=0.0001,
    save_dir='family_models'
)

# Train the model
model, history = trainer.train(
    father_latents=father_latents,
    mother_latents=mother_latents,
    child_images=child_images,
    train_indices=train_indices,
    test_indices=test_indices,
    num_epochs=30,
    batch_size=8
)

# Use the trained model to generate a child image
# For example, using the first test family
if len(test_indices) > 0:
    test_idx = test_indices[0]
    father_latent = father_latents[test_idx]
    mother_latent = mother_latents[test_idx]

    # Generate child image
    generated_child_image = trainer.generate_child_image(father_latent, mother_latent)

    # Display the result
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(get_image_from_path(father_images[test_idx]))
    plt.title("Father")

    plt.subplot(1, 3, 2)
    plt.imshow(get_image_from_path(mother_images[test_idx]))
    plt.title("Mother")

    plt.subplot(1, 3, 3)
    plt.imshow(generated_child_image)
    plt.title("Generated Child")
    
    # Save the result
    plt.savefig(os.path.join(OUTPUT_DIR, "sample_generation.png"))
    plt.show()
    
    # Save individual images
    generated_child_image.save(os.path.join(OUTPUT_DIR, "generated_child.png"))
else:
    print("No valid test families available for visualization")