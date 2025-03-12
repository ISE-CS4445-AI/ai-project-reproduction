import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

class FamilyLatentDataset(Dataset):
    """Dataset for family latent codes (father, mother, child)"""
    
    def __init__(self, father_latents, mother_latents, child_latents, indices):
        """
        Initialize the dataset.
        
        Args:
            father_latents (list): List of father latent codes
            mother_latents (list): List of mother latent codes
            child_latents (list): List of child latent codes
            indices (list): Indices to use from the latent lists
        """
        self.father_latents = [father_latents[i] for i in indices]
        self.mother_latents = [mother_latents[i] for i in indices]
        self.child_latents = [child_latents[i] for i in indices]
        
    def __len__(self):
        return len(self.father_latents)
    
    def __getitem__(self, idx):
        return {
            'father_latent': self.father_latents[idx],
            'mother_latent': self.mother_latents[idx],
            'child_latent': self.child_latents[idx]
        }

class LatentWeightGenerator(nn.Module):
    """Model to predict weights for latent code combination"""
    
    def __init__(self, latent_shape):
        """
        Initialize the model.
        
        Args:
            latent_shape (tuple): Shape of the latent codes (typically [18, 512] for StyleGAN)
        """
        super(LatentWeightGenerator, self).__init__()
        
        self.latent_shape = latent_shape
        latent_size = latent_shape[0] * latent_shape[1]  # Total size of flattened latent
        
        # Encoder for father and mother latents
        self.encoder = nn.Sequential(
            nn.Linear(latent_size * 2, 2048),
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
        
        # Weight decoder - generates weights for the latent combination
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
        
        # Generate weight tensor
        encoded = self.encoder(combined)
        weights_flat = self.weight_decoder(encoded)
        
        # Reshape to original latent shape
        weights = weights_flat.view(batch_size, *self.latent_shape)
        
        return weights

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
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # Tracking metrics
        self.train_losses = []
        self.val_losses = []
        
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
    
    def process_child_images(self, child_images, child_latents=None):
        """
        Process child images to extract latent codes.
        
        Args:
            child_images (list): List of child image paths
            child_latents (list): Optional pre-computed child latents
            
        Returns:
            list: List of child latent codes
        """
        if child_latents is not None:
            return child_latents
            
        child_latents = []
        for i, child_path in enumerate(tqdm(child_images, desc="Processing child images")):
            try:
                _, child_latent, _ = self.processor.process_image(child_path)
                child_latents.append(child_latent)
            except Exception as e:
                print(f"Error processing child image {child_path}: {e}")
                child_latents.append(None)
                
        return child_latents
    
    def prepare_dataloaders(self, father_latents, mother_latents, child_latents, 
                          train_indices, test_indices, batch_size=8):
        """
        Prepare training and validation dataloaders.
        
        Args:
            father_latents (list): List of father latent codes
            mother_latents (list): List of mother latent codes
            child_latents (list): List of child latent codes
            train_indices (list): Indices for training set
            test_indices (list): Indices for test set
            batch_size (int): Batch size for dataloaders
            
        Returns:
            tuple: (train_dataloader, val_dataloader)
        """
        # Create datasets
        train_dataset = FamilyLatentDataset(
            father_latents, mother_latents, child_latents, train_indices
        )
        val_dataset = FamilyLatentDataset(
            father_latents, mother_latents, child_latents, test_indices
        )
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        return train_dataloader, val_dataloader
    
    def train_epoch(self, dataloader):
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader for training data
            
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            # Get data and move to device
            father_latent = batch['father_latent'].to(self.device)
            mother_latent = batch['mother_latent'].to(self.device)
            child_latent = batch['child_latent'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass - get weights
            weights = self.model(father_latent, mother_latent)
            
            # Combine latents using weights
            predicted_child_latent = self._combine_latents_with_weights(
                father_latent, mother_latent, weights
            )
            
            # Calculate loss
            mse = self.mse_loss(predicted_child_latent, child_latent)
            l1 = self.l1_loss(predicted_child_latent, child_latent)
            loss = mse + 0.1 * l1  # Weighted combination of losses
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        """
        Validate the model.
        
        Args:
            dataloader: DataLoader for validation data
            
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                # Get data and move to device
                father_latent = batch['father_latent'].to(self.device)
                mother_latent = batch['mother_latent'].to(self.device)
                child_latent = batch['child_latent'].to(self.device)
                
                # Forward pass - get weights
                weights = self.model(father_latent, mother_latent)
                
                # Combine latents using weights
                predicted_child_latent = self._combine_latents_with_weights(
                    father_latent, mother_latent, weights
                )
                
                # Calculate loss
                mse = self.mse_loss(predicted_child_latent, child_latent)
                l1 = self.l1_loss(predicted_child_latent, child_latent)
                loss = mse + 0.1 * l1  # Weighted combination of losses
                
                total_loss += loss.item()
                
        return total_loss / len(dataloader)
    
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
        print("Processing child images...")
        child_latents = self.process_child_images(child_images)
        
        # Filter out families with missing latents
        valid_indices = []
        for i in range(len(father_latents)):
            if (father_latents[i] is not None and 
                mother_latents[i] is not None and
                child_latents[i] is not None):
                valid_indices.append(i)
                
        train_indices = [i for i in train_indices if i in valid_indices]
        test_indices = [i for i in test_indices if i in valid_indices]
        
        print(f"Training on {len(train_indices)} families, validating on {len(test_indices)} families")
        
        # Prepare dataloaders
        train_dataloader, val_dataloader = self.prepare_dataloaders(
            father_latents, mother_latents, child_latents,
            train_indices, test_indices, batch_size
        )
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train for one epoch
            train_loss = self.train_epoch(train_dataloader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_dataloader)
            self.val_losses.append(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model("best_model.pt")
                
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
        
        # Move latent to CPU for inference
        child_latent = child_latent.cpu()
        
        # Generate image from latent
        result_image = self.processor.combiner.generate_from_latent(child_latent)
        return result_image