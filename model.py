import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from differentiable_pipeline import DifferentiableFaceEncoder

class ParentLatentDataset(Dataset):
    """Dataset for parent latent codes (father, mother)"""
    
    def __init__(self, father_latents, mother_latents, indices):
        """
        Initialize the dataset.
        
        Args:
            father_latents (list): List of father latent codes
            mother_latents (list): List of mother latent codes
            indices (list): Indices to use from the latent lists
        """
        self.father_latents = [father_latents[i] for i in indices]
        self.mother_latents = [mother_latents[i] for i in indices]
        self.indices = indices  # Store original indices for reference
        
    def __len__(self):
        return len(self.father_latents)
    
    def __getitem__(self, idx):
        return {
            'father_latent': self.father_latents[idx],
            'mother_latent': self.mother_latents[idx],
            'original_idx': self.indices[idx]  # Return original index for reference
        }

class AdaptiveDimensionWeightModel(nn.Module):
    """
    Context-aware weighting model for StyleGAN latent blending.
    Predicts blending weights based on the specific features of both parents.
    """
    def __init__(self, latent_shape):
        """
        Initialize the model.
        
        Args:
            latent_shape (tuple): Shape of the latent codes (typically [18, 512] for StyleGAN)
        """
        super(AdaptiveDimensionWeightModel, self).__init__()
        
        self.latent_shape = latent_shape
        self.num_layers = latent_shape[0]
        self.latent_dim = latent_shape[1]
        
        # Total size of flattened latent
        latent_size = self.num_layers * self.latent_dim
        input_dim = 2 * latent_size  # Both parents' full latent codes
        
        # Build encoder for feature extraction
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.3),
            
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512)
        )
        
        # Attention module to highlight important features
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
        
        # Weight decoder for generating latent weights
        self.weight_decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(2048),
            
            nn.Linear(2048, latent_size),
            nn.Sigmoid()  # Output weights between 0 and 1
        )
        
    def forward(self, father_latent, mother_latent):
        """
        Forward pass to generate weights.
        
        Args:
            father_latent (torch.Tensor): Father's latent code
            mother_latent (torch.Tensor): Mother's latent code
            
        Returns:
            torch.Tensor: Weights for latent combination
        """
        # Flatten latents and concatenate
        batch_size = father_latent.size(0)
        father_flat = father_latent.view(batch_size, -1)
        mother_flat = mother_latent.view(batch_size, -1)
        combined = torch.cat([father_flat, mother_flat], dim=1)
        
        # Apply attention to highlight important features
        attention_weights = self.attention(combined)
        attended_inputs = combined * attention_weights
        
        # Generate latent weights
        encoded = self.encoder(attended_inputs)
        weights_flat = self.weight_decoder(encoded)
        
        # Reshape to original latent shape
        weights = weights_flat.view(batch_size, *self.latent_shape)
        
        return weights

class LatentWeightGenerator(nn.Module):
    """Model to predict weights for latent code combination"""
    
    def __init__(self, latent_shape):
        """
        Initialize the model.
        
        Args:
            latent_shape (tuple): Shape of the latent codes (typically [18, 512] for StyleGAN)
        """
        super(LatentWeightGenerator, self).__init__()
        
        # For compatibility with existing code, we'll use the AdaptiveDimensionWeightModel
        self.model = AdaptiveDimensionWeightModel(latent_shape)
        
    def forward(self, father_latent, mother_latent):
        """
        Forward pass to generate weights.
        
        Args:
            father_latent (torch.Tensor): Father's latent code
            mother_latent (torch.Tensor): Mother's latent code
            
        Returns:
            torch.Tensor: Weights for latent combination
        """
        return self.model(father_latent, mother_latent)

class LatentWeightTrainer:
    """Trainer for the latent weight generator model"""
    
    def __init__(self, processor, latent_shape=(18, 512), learning_rate=0.0001, 
                 device=None, save_dir='models'):
        """
        Initialize the trainer.
        
        Args:
            processor: The E4E processor with combiner for latent operations
            latent_shape (tuple): Shape of the latent codes
            learning_rate (float): Learning rate for optimizer
            device (torch.device): Device to use for training
            save_dir (str): Directory to save models
        """
        self.processor = processor
        self.latent_shape = latent_shape
        self.learning_rate = learning_rate
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir
        
        # Create model directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize model
        self.model = LatentWeightGenerator(latent_shape).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5)
        
        # Tracking metrics
        self.train_losses = []
        self.val_losses = []
        
        # Face encoder for perceptual loss
        self.face_encoder = None
        
    def _initialize_face_encoder(self):
        """Initialize face encoder for perceptual loss if not already initialized"""
        if self.face_encoder is None:
            self.face_encoder = DifferentiableFaceEncoder().to(self.device)
            # Freeze encoder weights
            for param in self.face_encoder.parameters():
                param.requires_grad = False
    
    def _combine_latents_with_weights(self, father_latent, mother_latent, weights):
        """
        Combine father and mother latents using the predicted weights.
        
        Args:
            father_latent (torch.Tensor): Father's latent code
            mother_latent (torch.Tensor): Mother's latent code
            weights (torch.Tensor): Weights for latent combination
            
        Returns:
            torch.Tensor: Combined latent code
        """
        # weights tensor represents the weight for father's contribution
        # (1 - weights) is implicitly the weight for mother's contribution
        return father_latent * weights + mother_latent * (1 - weights)
    
    def extract_face_embeddings(self, image_paths):
        """
        Extract face embeddings from a list of image paths.
        
        Args:
            image_paths (list): List of image paths
            
        Returns:
            list: List of face embeddings
        """
        self._initialize_face_encoder()
        
        face_embeddings = []
        
        for i, image_path in enumerate(tqdm(image_paths, desc="Extracting face embeddings")):
            try:
                # Load and process image
                img = Image.open(image_path).convert('RGB')
                img_tensor = self.face_encoder.preprocess(img).unsqueeze(0).to(self.device)
                
                # Extract embedding
                with torch.no_grad():
                    embedding = self.face_encoder(img_tensor)
                face_embeddings.append(embedding)
            except Exception as e:
                print(f"Error extracting embedding from {image_path}: {e}")
                face_embeddings.append(None)
                
        return face_embeddings
    
    def _compute_face_similarity_loss(self, generated_image, target_embedding):
        """
        Compute perceptual loss between generated image and target embedding.
        
        Args:
            generated_image (PIL.Image): Generated child image
            target_embedding (torch.Tensor): Target face embedding
            
        Returns:
            torch.Tensor: Perceptual loss value
        """
        self._initialize_face_encoder()
        
        # Process generated image
        img_tensor = self.face_encoder.preprocess(generated_image).unsqueeze(0).to(self.device)
        generated_embedding = self.face_encoder(img_tensor)
        
        # Compute cosine similarity (higher is better)
        similarity = F.cosine_similarity(generated_embedding, target_embedding)
        
        # Return negative similarity as loss (lower is better)
        return -similarity
    
    def prepare_dataloaders(self, father_latents, mother_latents, 
                           train_indices, test_indices, batch_size=8):
        """
        Prepare training and validation dataloaders.
        
        Args:
            father_latents (list): List of father latent codes
            mother_latents (list): List of mother latent codes
            train_indices (list): Indices for training set
            test_indices (list): Indices for test set
            batch_size (int): Batch size for dataloaders
            
        Returns:
            tuple: (train_dataloader, val_dataloader)
        """
        # Create datasets
        train_dataset = ParentLatentDataset(
            father_latents, mother_latents, train_indices
        )
        val_dataset = ParentLatentDataset(
            father_latents, mother_latents, test_indices
        )
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        return train_dataloader, val_dataloader
    
    def train_epoch(self, dataloader, child_embeddings):
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader for training data
            child_embeddings: List of pre-extracted child face embeddings
            
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        valid_batches = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            # Get data and move to device
            father_latent = batch['father_latent'].to(self.device)
            mother_latent = batch['mother_latent'].to(self.device)
            original_indices = batch['original_idx'].tolist()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass - get weights
            weights = self.model(father_latent, mother_latent)
            
            # Combine latents using weights
            predicted_child_latent = self._combine_latents_with_weights(
                father_latent, mother_latent, weights
            )
            
            # Calculate batch loss
            batch_loss = 0.0
            valid_samples = 0
            
            # Process each sample in the batch
            for i in range(father_latent.size(0)):
                # Get the original index for this sample
                original_idx = original_indices[i]
                
                # Get the target embedding
                target_embedding = child_embeddings[original_idx]
                
                if target_embedding is not None:
                    # Generate child image from predicted latent
                    sample_latent = predicted_child_latent[i]  # Keep on same device, don't use .cpu()
                    generated_image = self.processor.combiner.generate_from_latent(sample_latent)
                    
                    # Compute perceptual loss
                    sample_loss = self._compute_face_similarity_loss(generated_image, target_embedding)
                    batch_loss += sample_loss
                    valid_samples += 1
            
            # Average loss over valid samples in batch
            if valid_samples > 0:
                batch_loss = batch_loss / valid_samples
                
                # Backward pass and optimize
                batch_loss.backward()
                self.optimizer.step()
                
                total_loss += batch_loss.item()
                valid_batches += 1
            
        if valid_batches == 0:
            return 0.0
        return total_loss / valid_batches
    
    def validate(self, dataloader, child_embeddings):
        """
        Validate the model.
        
        Args:
            dataloader: DataLoader for validation data
            child_embeddings: List of pre-extracted child face embeddings
            
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        total_loss = 0
        valid_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                # Get data and move to device
                father_latent = batch['father_latent'].to(self.device)
                mother_latent = batch['mother_latent'].to(self.device)
                original_indices = batch['original_idx'].tolist()
                
                # Forward pass - get weights
                weights = self.model(father_latent, mother_latent)
                
                # Combine latents using weights
                predicted_child_latent = self._combine_latents_with_weights(
                    father_latent, mother_latent, weights
                )
                
                # Calculate batch loss
                batch_loss = 0.0
                valid_samples = 0
                
                # Process each sample in the batch
                for i in range(father_latent.size(0)):
                    # Get the original index for this sample
                    original_idx = original_indices[i]
                    
                    # Get the target embedding
                    target_embedding = child_embeddings[original_idx]
                    
                    if target_embedding is not None:
                        # Generate child image from predicted latent
                        sample_latent = predicted_child_latent[i]  # Keep on same device, don't use .cpu()
                        generated_image = self.processor.combiner.generate_from_latent(sample_latent)
                        
                        # Compute perceptual loss
                        sample_loss = self._compute_face_similarity_loss(generated_image, target_embedding)
                        batch_loss += sample_loss
                        valid_samples += 1
                
                # Average loss over valid samples in batch
                if valid_samples > 0:
                    batch_loss = batch_loss / valid_samples
                    total_loss += batch_loss.item()
                    valid_batches += 1
                
        if valid_batches == 0:
            return 0.0
        return total_loss / valid_batches
    
    def train(self, father_latents, mother_latents, child_images, 
              train_indices, test_indices, num_epochs=30, batch_size=8):
        """
        Train the model.
        
        Args:
            father_latents (list): List of father latent codes
            mother_latents (list): List of mother latent codes
            child_images (list): List of child image paths
            train_indices (list): Indices for training set
            test_indices (list): Indices for test set
            num_epochs (int): Number of epochs to train for
            batch_size (int): Batch size for training
            
        Returns:
            tuple: (trained_model, training_history)
        """
        print("Extracting face embeddings from child images...")
        child_embeddings = self.extract_face_embeddings(child_images)
        
        # Filter out families with missing data
        valid_indices = []
        for i in range(len(father_latents)):
            if (i < len(father_latents) and father_latents[i] is not None and
                i < len(mother_latents) and mother_latents[i] is not None and
                i < len(child_embeddings) and child_embeddings[i] is not None):
                valid_indices.append(i)
                
        train_indices = [i for i in train_indices if i in valid_indices]
        test_indices = [i for i in test_indices if i in valid_indices]
        
        print(f"Training on {len(train_indices)} families, validating on {len(test_indices)} families")
        
        # Prepare dataloaders
        train_dataloader, val_dataloader = self.prepare_dataloaders(
            father_latents, mother_latents,
            train_indices, test_indices, batch_size
        )
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train for one epoch
            train_loss = self.train_epoch(train_dataloader, child_embeddings)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_dataloader, child_embeddings)
            self.val_losses.append(val_loss)
            
            # Update learning rate based on validation performance
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model("best_model.pt")
                
                # Create visualizations for the best model
                self.visualize_samples(father_latents, mother_latents, child_images, 
                                       test_indices[:3], epoch)
                
            # Print progress
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, "
                  f"Time: {epoch_time:.2f}s")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_model(f"checkpoint_epoch_{epoch+1}.pt")
                
        # Plot and save training history
        self.plot_training_history()
        
        # Load best model
        self.load_model("best_model.pt")
        
        return self.model, {'train_losses': self.train_losses, 'val_losses': self.val_losses}
    
    def visualize_samples(self, father_latents, mother_latents, child_images, indices, epoch):
        """
        Visualize sample results for the current model.
        
        Args:
            father_latents: List of father latent codes
            mother_latents: List of mother latent codes
            child_images: List of child image paths
            indices: Indices of samples to visualize
            epoch: Current epoch number
        """
        # Create output directory
        viz_dir = os.path.join(self.save_dir, f'epoch_{epoch}_viz')
        os.makedirs(viz_dir, exist_ok=True)
        
        self.model.eval()
        
        with torch.no_grad():
            for i, idx in enumerate(indices):
                father_latent = father_latents[idx].to(self.device)
                mother_latent = mother_latents[idx].to(self.device)
                
                # If latents are missing batch dimension, add it
                if father_latent.dim() == 2:
                    father_latent = father_latent.unsqueeze(0)
                if mother_latent.dim() == 2:
                    mother_latent = mother_latent.unsqueeze(0)
                
                # Get weights for this specific pair
                weights = self.model(father_latent, mother_latent)
                
                # Blend latents
                combined_latent = self._combine_latents_with_weights(
                    father_latent, mother_latent, weights
                )
                
                # Generate image
                result_image = self.processor.combiner.generate_from_latent(combined_latent.squeeze(0))
                
                # Save result
                result_path = os.path.join(viz_dir, f'sample_{i}_result.jpg')
                result_image.save(result_path)
                
                # Copy target for comparison
                try:
                    import shutil
                    target_path = os.path.join(viz_dir, f'sample_{i}_target.jpg')
                    shutil.copy(child_images[idx], target_path)
                except Exception as e:
                    print(f"Error copying target image: {e}")
                
                # Visualize weight heatmap
                self.visualize_weight_heatmap(weights.squeeze(0), viz_dir, f'sample_{i}_weights')
    
    def visualize_weight_heatmap(self, weights, output_dir, name):
        """
        Create heatmap visualization of weights.
        
        Args:
            weights: Weight tensor to visualize
            output_dir: Directory to save the visualization
            name: Base filename for the visualization
        """
        plt.figure(figsize=(10, 6))
        plt.imshow(weights.cpu().numpy(), cmap='coolwarm', vmin=0, vmax=1)
        plt.colorbar(label='Weight (Father)')
        plt.title('Father Contribution by Feature')
        plt.xlabel('Latent Dimension')
        plt.ylabel('StyleGAN Layer')
        plt.savefig(os.path.join(output_dir, f'{name}.png'))
        plt.close()
    
    def save_model(self, filename):
        """
        Save the model.
        
        Args:
            filename (str): Filename to save model to
        """
        save_path = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'latent_shape': self.latent_shape
        }, save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, filename):
        """
        Load a saved model.
        
        Args:
            filename (str): Filename to load model from
            
        Returns:
            LatentWeightGenerator: Loaded model
        """
        load_path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(load_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"Model loaded from {load_path}")
        return self.model
    
    def plot_training_history(self):
        """Plot and save the training history."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.save_dir, f'training_history_{timestamp}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Training history plot saved to {plot_path}")
    
    def predict_weights(self, father_latent, mother_latent):
        """
        Predict weights for combining father and mother latents.
        
        Args:
            father_latent (torch.Tensor): Father's latent code
            mother_latent (torch.Tensor): Mother's latent code
            
        Returns:
            torch.Tensor: Predicted weights for latent combination
        """
        self.model.eval()
        
        with torch.no_grad():
            # Ensure latents are on the correct device
            father_latent = father_latent.to(self.device)
            mother_latent = mother_latent.to(self.device)
            
            # Add batch dimension if needed
            if father_latent.dim() == 2:
                father_latent = father_latent.unsqueeze(0)
            if mother_latent.dim() == 2:
                mother_latent = mother_latent.unsqueeze(0)
            
            # Predict weights
            weights = self.model(father_latent, mother_latent)
            
            # Return weights (removing batch dimension if it was added)
            if weights.size(0) == 1:
                weights = weights.squeeze(0)
                
            return weights
    
    def generate_child_latent(self, father_latent, mother_latent):
        """
        Generate a child latent code from father and mother latents.
        
        Args:
            father_latent (torch.Tensor): Father's latent code
            mother_latent (torch.Tensor): Mother's latent code
            
        Returns:
            torch.Tensor: Predicted child latent code
        """
        # Predict weights
        weights = self.predict_weights(father_latent, mother_latent)
        
        # Ensure latents are on the same device as weights
        father_latent = father_latent.to(weights.device)
        mother_latent = mother_latent.to(weights.device)
        
        # Add batch dimension if needed
        if weights.dim() == 2:
            weights = weights.unsqueeze(0)
            father_latent = father_latent.unsqueeze(0)
            mother_latent = mother_latent.unsqueeze(0)
        
        # Combine latents using weights
        child_latent = self._combine_latents_with_weights(
            father_latent, mother_latent, weights
        )
        
        # Remove batch dimension if it was added
        if child_latent.size(0) == 1:
            child_latent = child_latent.squeeze(0)
            
        return child_latent
    
    def generate_child_image(self, father_latent, mother_latent):
        """
        Generate a child image from father and mother latents.
        
        Args:
            father_latent (torch.Tensor): Father's latent code
            mother_latent (torch.Tensor): Mother's latent code
            
        Returns:
            PIL.Image: Generated child image
        """
        # Generate child latent
        child_latent = self.generate_child_latent(father_latent, mother_latent)
        
        try:
            # Try to generate image directly with latent on its current device
            result_image = self.processor.combiner.generate_from_latent(child_latent)
        except RuntimeError as e:
            # If there's a device mismatch error, try with CPU tensor as fallback
            if "must be a CUDA tensor" in str(e):
                print("Warning: Generator requires CUDA tensors. Moving latent to CUDA...")
                try:
                    # Try moving to CUDA
                    child_latent = child_latent.cuda()
                    result_image = self.processor.combiner.generate_from_latent(child_latent)
                except (RuntimeError, AttributeError):
                    # If that fails, maybe the generator expects CPU tensors?
                    print("Warning: Moving latent to CPU instead...")
                    child_latent = child_latent.cpu()
                    result_image = self.processor.combiner.generate_from_latent(child_latent)
            elif "expected CPU tensor" in str(e):
                # Generator might be on CPU instead of GPU
                print("Warning: Generator requires CPU tensors. Moving latent to CPU...")
                child_latent = child_latent.cpu()
                result_image = self.processor.combiner.generate_from_latent(child_latent)
            else:
                # Some other error
                raise e
            
        return result_image