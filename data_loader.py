import os
import pandas as pd
import random
from PIL import Image
import torch
import logging

logger = logging.getLogger(__name__)

class FamilyDataLoader:
    def __init__(self, base_path, csv_path):
        """
        Initialize the family data loader.
        
        Args:
            base_path: Base path for images
            csv_path: Path to the CSV file with family data
        """
        self.base_path = base_path
        self.csv_path = csv_path
        self.df = None
        self.load_csv()
        
    def load_csv(self):
        """Load and prepare the CSV data."""
        try:
            self.df = pd.read_csv(self.csv_path)
            for list_col_name in ["mother_images", "father_images", "child_images"]:
                self.df[list_col_name] = self.df[list_col_name].map(eval)
            logger.info(f"Successfully loaded data from {self.csv_path}")
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise
            
    def get_image_from_path(self, image_path):
        """Load an image from the given path."""
        full_path = os.path.join(self.base_path, image_path)
        try:
            img = Image.open(full_path)
            return img
        except FileNotFoundError:
            logger.warning(f"File not found: {full_path}")
            return None
            
    def get_family(self, family_id):
        """
        Retrieve family information for a given family ID.
        
        Args:
            family_id: The ID of the family
            
        Returns:
            Dictionary containing family information
        """
        try:
            family_data = self.df[self.df['family_id'] == family_id].iloc[0]
            
            # Get number of images for each family member
            no_father_images = len(family_data['father_images'])
            no_mother_images = len(family_data['mother_images'])
            no_child_images = len(family_data['child_images'])
            
            # Take minimum number of images across all members
            no_of_images = min(no_father_images, no_mother_images, no_child_images)
            
            # Randomly sample the same number of images for each member
            father_images = random.sample(
                family_data['father_images'],
                k=min(no_of_images, len(family_data['father_images']))
            )
            mother_images = random.sample(
                family_data['mother_images'],
                k=min(no_of_images, len(family_data['mother_images']))
            )
            child_images = random.sample(
                family_data['child_images'],
                k=min(no_of_images, len(family_data['child_images']))
            )
            
            # Prepend base path if needed
            father_images = [
                os.path.join(self.base_path, img) if not os.path.isabs(img) else img
                for img in father_images
            ]
            mother_images = [
                os.path.join(self.base_path, img) if not os.path.isabs(img) else img
                for img in mother_images
            ]
            child_images = [
                os.path.join(self.base_path, img) if not os.path.isabs(img) else img
                for img in child_images
            ]
            
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
            logger.warning(f"Family with ID {family_id} not found.")
            return None
        except Exception as e:
            logger.error(f"Error getting family {family_id}: {e}")
            return None
            
    def load_all_families(self):
        """Load all families and their images."""
        fathers = []
        mothers = []
        children = []
        
        max_family_id = self.df['family_id'].max()
        
        for i in range(max_family_id + 1):
            family = self.get_family(i)
            if family:
                father_images = family.get('father_images')
                mother_images = family.get('mother_images')
                children_images = family.get('child_images')
                
                for j in range(min(len(father_images), len(mother_images), len(children_images))):
                    fathers.append(father_images[j])
                    mothers.append(mother_images[j])
                    children.append(children_images[j])
                    
        logger.info(f"Loaded {len(fathers)} family image sets")
        return fathers, mothers, children
        
    def load_latents(self, latent_dir):
        """Load pre-computed latent codes."""
        father_latents = []
        mother_latents = []
        
        if not os.path.exists(latent_dir):
            logger.error(f"Latent directory '{latent_dir}' does not exist")
            return [], []
            
        # Count available latent pairs
        num_latent_pairs = 0
        while os.path.exists(os.path.join(latent_dir, f'father_latent_{num_latent_pairs}.pt')) and \
              os.path.exists(os.path.join(latent_dir, f'mother_latent_{num_latent_pairs}.pt')):
            num_latent_pairs += 1
            
        logger.info(f"Found {num_latent_pairs} latent pairs in {latent_dir}")
        
        # Load the latent codes
        for i in range(num_latent_pairs):
            father_latent_path = os.path.join(latent_dir, f'father_latent_{i}.pt')
            mother_latent_path = os.path.join(latent_dir, f'mother_latent_{i}.pt')
            
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
                
        return father_latents, mother_latents 