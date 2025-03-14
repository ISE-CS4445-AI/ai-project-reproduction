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
import abc

class BaseGeneticModel(nn.Module, abc.ABC):
    """
    Abstract base class for genetic models that generate features from parent latents.
    This serves as a common interface for both the weight generator and edit parameter generator.
    """
    def __init__(self):
        super(BaseGeneticModel, self).__init__()
    
    @abc.abstractmethod
    def forward(self, father_latent, mother_latent):
        """
        Forward pass to generate output from parent latents.
        
        Args:
            father_latent (torch.Tensor): Father's latent code
            mother_latent (torch.Tensor): Mother's latent code
            
        Returns:
            torch.Tensor: Output tensor (meaning depends on implementation)
        """
        pass
    
    @abc.abstractmethod
    def generate_child_latent(self, father_latent, mother_latent):
        """
        Generate a child latent code from father and mother latents.
        
        Args:
            father_latent (torch.Tensor): Father's latent code
            mother_latent (torch.Tensor): Mother's latent code
            
        Returns:
            torch.Tensor: Predicted child latent code
        """
        pass

class MockProcessor:
    """
    A mock processor to use during training that maintains gradient flow.
    Instead of generating actual images, it simulates the process in a differentiable way.
    """
    def __init__(self, face_encoder):
        self.face_encoder = face_encoder
        # Create a fixed weight mat@model.py is my current 'model file', should i create another one for my interface gan one? what structure make senserix for more stable transformations
        self.register_buffers()
        
    def register_buffers(self):
        """Create fixed weight matrices for stable transformations"""
        device = next(self.face_encoder.parameters()).device
        # Create fixed weight matrices for stability (won't change between calls)
        # Use a fixed seed for consistent initialization
        torch.manual_seed(42)
        self.projection_matrix = torch.randn(512, 9216, device=device) * 0.02  # Small init for stability
        self.projection_matrix = F.normalize(self.projection_matrix, dim=1)  # Normalize rows
        
    def generate_embeddings_from_latent(self, latent):
        """
        Generate face embeddings directly from latent code in a differentiable way.
        This is a simplified simulation for training purposes only.
        
        Args:
            latent: The latent code tensor
            
        Returns:
            Face embedding tensor with gradients attached
        """
        # Create a more stable differentiable transformation from latent to embedding space
        batch_size = latent.size(0) if latent.dim() > 2 else 1
        if latent.dim() == 2:
            latent = latent.unsqueeze(0)
        
        # Get device dynamically
        device = latent.device
        if not hasattr(self, 'projection_matrix') or self.projection_matrix.device != device:
            self.register_buffers()
            
        # Flatten the latent code
        h = latent.view(batch_size, -1)  # [batch_size, 18*512]
        
        # Apply stable linear projection with a fixed matrix
        if self.projection_matrix.device != h.device:
            self.projection_matrix = self.projection_matrix.to(h.device)
            
        h = F.linear(h, self.projection_matrix)  # [batch_size, 512]
        
        # Apply non-linearity and normalization
        h = F.relu(h)
        h = F.normalize(h, p=2, dim=1)  # Normalize to unit length like real embeddings
        
        return h

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

class LatentWeightGenerator(BaseGeneticModel):
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
        
    def generate_child_latent(self, father_latent, mother_latent):
        """
        Generate a child latent code from father and mother latents.
        
        Args:
            father_latent (torch.Tensor): Father's latent code
            mother_latent (torch.Tensor): Mother's latent code
            
        Returns:
            torch.Tensor: Predicted child latent code
        """
        # Get weights from the model
        weights = self.forward(father_latent, mother_latent)
        
        # Ensure latents are on the same device as weights
        father_latent = father_latent.to(weights.device)
        mother_latent = mother_latent.to(weights.device)
        
        # Add batch dimension if needed
        if weights.dim() == 2:
            weights = weights.unsqueeze(0)
            father_latent = father_latent.unsqueeze(0)
            mother_latent = mother_latent.unsqueeze(0)
        
        # Combine latents using weights
        child_latent = father_latent * weights + mother_latent * (1 - weights)
        
        # Remove batch dimension if it was added
        if child_latent.size(0) == 1:
            child_latent = child_latent.squeeze(0)
            
        return child_latent

class LatentWeightTrainer:
    """Trainer for genetic models (both weight generators and edit parameter generators)"""
    
    def __init__(self, processor, model_type='weight', latent_shape=(18, 512), learning_rate=0.00005, 
                 device=None, save_dir='models', weight_decay=1e-5, clip_value=1.0):
        """
        Initialize the trainer.
        
        Args:
            processor: The E4E processor with combiner for latent operations
            model_type (str): Type of model to train ('weight' or 'edit')
            latent_shape (tuple): Shape of the latent codes
            learning_rate (float): Learning rate for optimizer
            device (torch.device): Device to use for training
            save_dir (str): Directory to save models
            weight_decay (float): Weight decay for regularization
            clip_value (float): Gradient clipping value
        """
        self.processor = processor
        self.model_type = model_type
        self.latent_shape = latent_shape
        self.learning_rate = learning_rate
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir
        self.clip_value = clip_value
        
        # Create model directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize model based on type
        if model_type == 'weight':
            self.model = LatentWeightGenerator(latent_shape).to(self.device)
        elif model_type == 'edit':
            self.model = EditParamGenerator(latent_shape).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Expected 'weight' or 'edit'")
        
        # Use AdamW with weight decay for better regularization
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)  # Default betas
        )
        
        # Use ReduceLROnPlateau with more conservative parameters
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            patience=15,  # Increased patience for more stability
            factor=0.5,   # Less aggressive reduction
            min_lr=1e-7,  # Lower minimum learning rate
            verbose=True,  # Print when learning rate changes
            threshold=0.01,  # Only reduce LR when improvement is significant
            threshold_mode='rel'  # Relative improvement
        )
        
        # Tracking metrics
        self.train_losses = []
        self.val_losses = []
        
        # Face encoder for perceptual loss
        self.face_encoder = None
        
        # Mock processor for differentiable training
        self.mock_processor = None
        
        # Loss scaling for stability
        self.loss_scale = 5.0  # Start with a lower scale factor for stability
        self.smooth_factor = 0.01  # Lower smooth factor for gradual adjustments

    def _initialize_face_encoder(self):
        """Initialize face encoder for perceptual loss if not already initialized"""
        if self.face_encoder is None:
            self.face_encoder = DifferentiableFaceEncoder().to(self.device)
            # Freeze encoder weights
            for param in self.face_encoder.parameters():
                param.requires_grad = False
                
        # Initialize mock processor if needed
        if self.mock_processor is None:
            self.mock_processor = MockProcessor(self.face_encoder)
    
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
            result = self.model(father_latent, mother_latent)
            
            # Return result (removing batch dimension if it was added)
            if result.size(0) == 1:
                result = result.squeeze(0)
                
            return result
    
    def predict_edit_parameters(self, father_latent, mother_latent):
        """
        Predict edit parameters for InterfaceGAN directions.
        Only valid for edit parameter models.
        
        Args:
            father_latent (torch.Tensor): Father's latent code
            mother_latent (torch.Tensor): Mother's latent code
            
        Returns:
            dict: Mapping of direction names to parameter values
        """
        if self.model_type != 'edit':
            raise ValueError("This method is only valid for edit parameter models")
            
        if not hasattr(self.model, 'get_edit_parameters'):
            raise AttributeError("Model does not have get_edit_parameters method")
            
        self.model.eval()
        
        with torch.no_grad():
            # Ensure latents are on the correct device
            father_latent = father_latent.to(self.device)
            mother_latent = mother_latent.to(self.device)
            
            return self.model.get_edit_parameters(father_latent, mother_latent)
    
    def visualize_edit_parameters(self, father_latent, mother_latent, output_path=None):
        """
        Visualize the predicted edit parameters.
        Only valid for edit parameter models.
        
        Args:
            father_latent (torch.Tensor): Father's latent code
            mother_latent (torch.Tensor): Mother's latent code
            output_path (str, optional): Path to save the visualization
            
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if self.model_type != 'edit':
            raise ValueError("This method is only valid for edit parameter models")
            
        params = self.predict_edit_parameters(father_latent, mother_latent)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        directions = list(params.keys())
        values = list(params.values())
        
        bars = ax.bar(directions, values, color=['skyblue', 'lightgreen', 'salmon'])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            offset = 0.1 if height >= 0 else -0.3
            ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                   f'{height:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Set labels and title
        ax.set_ylabel('Parameter Value')
        ax.set_title('Predicted Edit Parameters')
        
        # Set y-axis range to include all values with some padding
        max_abs_val = max(abs(v) for v in values)
        y_range = max(3.5, max_abs_val * 1.2)  # At least -3.5 to 3.5 or 20% more than max
        ax.set_ylim(-y_range, y_range)
        
        # Add grid lines
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Save if output path is provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
            
        return fig

    def extract_face_embeddings(self, image_paths, save_path=None):
        """
        Extract face embeddings from a list of image paths and optionally save to a file.
        
        Args:
            image_paths (list): List of image paths
            save_path (str, optional): Path to save the embeddings tensor
            
        Returns:
            list: List of face embeddings
        """
        self._initialize_face_encoder()
        
        face_embeddings = []
        valid_embeddings = []  # List to track which embeddings are valid (not None)
        valid_indices = []     # List to track original indices of valid embeddings
        
        for i, image_path in enumerate(tqdm(image_paths, desc="Extracting face embeddings")):
            try:
                # Load and process image
                img = Image.open(image_path).convert('RGB')
                img_tensor = self.face_encoder.preprocess(img).unsqueeze(0).to(self.device)
                
                # Extract embedding
                with torch.no_grad():
                    embedding = self.face_encoder(img_tensor)
                face_embeddings.append(embedding)
                valid_embeddings.append(embedding)
                valid_indices.append(i)
            except Exception as e:
                print(f"Error extracting embedding from {image_path}: {e}")
                face_embeddings.append(None)
        
        # Save embeddings to file if path is provided
        if save_path and valid_embeddings:
            # Stack valid embeddings into a single tensor
            stacked_embeddings = torch.cat(valid_embeddings, dim=0)
            
            # Create a dictionary with embeddings and their original indices
            embeddings_data = {
                'embeddings': stacked_embeddings,
                'indices': valid_indices
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save to file
            torch.save(embeddings_data, save_path)
            print(f"Saved {len(valid_embeddings)} child embeddings to {save_path}")
            
        return face_embeddings
    
    def _compute_face_similarity_loss(self, generated_embedding, target_embedding):
        """
        Compute perceptual loss between generated embedding and target embedding.
        
        Args:
            generated_embedding (torch.Tensor): Generated face embedding
            target_embedding (torch.Tensor): Target face embedding
            
        Returns:
            torch.Tensor: Perceptual loss value
        """
        # Always ensure both embeddings are normalized for consistent similarity metrics
        generated_embedding = F.normalize(generated_embedding, p=2, dim=1)
        
        # Ensure target embedding is also normalized
        if target_embedding.requires_grad:
            target_embedding = F.normalize(target_embedding, p=2, dim=1)
        else:
            # If not requiring gradients, use torch.nn.functional
            with torch.no_grad():
                target_embedding = F.normalize(target_embedding, p=2, dim=1)
        
        # Compute cosine similarity (higher is better)
        cos_sim = F.cosine_similarity(generated_embedding, target_embedding)
        
        # Compute L2 distance (lower is better)
        # Since the embeddings are normalized, this is mathematically related to cosine similarity
        # but empirically provides more stable gradients when combined
        l2_dist = torch.sum((generated_embedding - target_embedding) ** 2, dim=1)
        
        # Weighted combination with more conservative weighting
        # Reduce the influence of L2 distance to prevent it from dominating the loss
        loss = (1.0 - cos_sim) + 0.05 * l2_dist
        
        return loss
    
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
        self._initialize_face_encoder()  # Ensure face encoder and mock processor are initialized
        
        self.model.train()
        total_loss = 0
        valid_batches = 0
        
        # Use running average of loss for stability
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            # Get data and move to device
            father_latent = batch['father_latent'].to(self.device)
            mother_latent = batch['mother_latent'].to(self.device)
            original_indices = batch['original_idx'].tolist()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass - get model output (weights or edit params)
            model_output = self.model(father_latent, mother_latent)
            
            # Generate child latent based on model type
            if self.model_type == 'weight':
                # For weight model, combine latents using weights
                predicted_child_latent = self._combine_latents_with_weights(
                    father_latent, mother_latent, model_output
                )
            else:  # edit model
                # For edit model, start with average of parents and apply edits
                # Start with equal blend of parents (base latent)
                base_latent = (father_latent + mother_latent) / 2.0
                
                # We can't use the real processor during training since it's not differentiable
                # Instead, we'll just use the base latent and rely on the face embedding loss
                predicted_child_latent = base_latent
            
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
                    # Generate embeddings directly (differentiable pathway)
                    sample_latent = predicted_child_latent[i].unsqueeze(0)  # Add batch dim
                    generated_embedding = self.mock_processor.generate_embeddings_from_latent(sample_latent)
                    
                    # Compute perceptual loss
                    sample_loss = self._compute_face_similarity_loss(generated_embedding, target_embedding)
                    batch_loss += sample_loss
                    valid_samples += 1
            
            # Average loss over valid samples in batch
            if valid_samples > 0:
                batch_loss = batch_loss / valid_samples
                
                # Use a fixed loss scale rather than dynamic adjustment for more stability
                scaled_loss = batch_loss * self.loss_scale
                
                # Backward pass with scaled loss
                scaled_loss.backward()
                
                # Apply gradient clipping with a more conservative value
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
                
                # Optimize
                self.optimizer.step()
                
                # Update tracking metrics
                loss_value = batch_loss.item()
                total_loss += loss_value
                valid_batches += 1
                
                # Compute exponential moving average of loss for more stable tracking
                if running_loss == 0:
                    running_loss = loss_value
                else:
                    running_loss = 0.9 * running_loss + 0.1 * loss_value
                
                # Adjust loss scale more conservatively
                with torch.no_grad():
                    grad_norm = 0.0
                    param_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.data.norm(2).item() ** 2
                        if p.data is not None:
                            param_norm += p.data.norm(2).item() ** 2
                    
                    grad_norm = grad_norm ** 0.5
                    param_norm = param_norm ** 0.5
                    
                    # Only adjust loss scale when gradient norm is far outside the desired range
                    # This prevents constant oscillation of the loss scale
                    if grad_norm > 10.0 and self.loss_scale > 0.2:
                        self.loss_scale *= (1.0 - self.smooth_factor * 0.5)  # Reduce slower
                    elif grad_norm < 0.01 and self.loss_scale < 50.0:
                        self.loss_scale *= (1.0 + self.smooth_factor * 0.5)  # Increase slower
                    
                    # Print detailed metrics every 10 batches
                    if batch_idx % 10 == 0:
                        print(f"  Batch {batch_idx}: loss={loss_value:.6f}, running_loss={running_loss:.6f}, "
                              f"grad_norm={grad_norm:.4f}, param_norm={param_norm:.4f}, loss_scale={self.loss_scale:.2f}")
            
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
        self._initialize_face_encoder()  # Ensure face encoder and mock processor are initialized
        
        self.model.eval()
        total_loss = 0
        valid_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                # Get data and move to device
                father_latent = batch['father_latent'].to(self.device)
                mother_latent = batch['mother_latent'].to(self.device)
                original_indices = batch['original_idx'].tolist()
                
                # Forward pass - get model output (weights or edit params)
                model_output = self.model(father_latent, mother_latent)
                
                # Generate child latent based on model type
                if self.model_type == 'weight':
                    # For weight model, combine latents using weights
                    predicted_child_latent = self._combine_latents_with_weights(
                        father_latent, mother_latent, model_output
                    )
                else:  # edit model
                    # For edit model, use average of parents
                    # During validation, we just use the base latent
                    predicted_child_latent = (father_latent + mother_latent) / 2.0
                
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
                        # Generate embeddings directly (differentiable pathway)
                        sample_latent = predicted_child_latent[i].unsqueeze(0)  # Add batch dim
                        generated_embedding = self.mock_processor.generate_embeddings_from_latent(sample_latent)
                        
                        # Compute perceptual loss
                        sample_loss = self._compute_face_similarity_loss(generated_embedding, target_embedding)
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
              train_indices, test_indices, num_epochs=300, batch_size=8, 
              embeddings_save_path=None, load_embeddings_from=None):
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
            embeddings_save_path (str, optional): Path to save child embeddings
            load_embeddings_from (str, optional): Path to load pre-computed child embeddings
            
        Returns:
            tuple: (trained_model, training_history)
        """
        # Handle child embeddings - either load from file or extract from images
        if load_embeddings_from and os.path.exists(load_embeddings_from):
            print(f"Loading pre-computed child embeddings from {load_embeddings_from}")
            embeddings_data = torch.load(load_embeddings_from)
            stacked_embeddings = embeddings_data['embeddings']
            valid_indices = embeddings_data['indices']
            
            # Convert to list format expected by training code
            child_embeddings = [None] * len(child_images)
            for i, idx in enumerate(valid_indices):
                if idx < len(child_embeddings):
                    child_embeddings[idx] = stacked_embeddings[i].unsqueeze(0)
            
            print(f"Loaded {len(valid_indices)} child embeddings")
        else:
            print("Extracting face embeddings from child images...")
            child_embeddings = self.extract_face_embeddings(
                child_images, 
                save_path=embeddings_save_path
            )
        
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
        best_epoch = -1
        patience_counter = 0
        max_patience = 25  # Extended early stopping patience
        
        # Keep track of best weights in memory
        best_weights = None
        
        # Smooth loss tracking for stability
        smooth_val_loss = None
        
        # Initialize visualization schedule - start more frequent then reduce
        vis_schedule = [1, 5, 10, 20, 50, 100, 150, 200, 250]
        
        print(f"Starting training with learning rate: {self.learning_rate}")
        print(f"Loss scaling: {self.loss_scale}, Gradient clipping: {self.clip_value}")
        print(f"Weight decay: {self.optimizer.param_groups[0]['weight_decay']}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train for one epoch
            train_loss = self.train_epoch(train_dataloader, child_embeddings)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_dataloader, child_embeddings)
            self.val_losses.append(val_loss)
            
            # Smooth validation loss with exponential moving average for more stable LR scheduling
            if smooth_val_loss is None:
                smooth_val_loss = val_loss
            else:
                smooth_val_loss = 0.8 * smooth_val_loss + 0.2 * val_loss
            
            # Update learning rate based on smoothed validation performance
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(smooth_val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            
            # Save checkpoint every 50 epochs regardless of performance
            if (epoch + 1) % 50 == 0:
                checkpoint_path = f"checkpoint_epoch_{epoch+1}.pt"
                self.save_model(checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")
            
            # Check for improvement
            if val_loss < best_val_loss:
                improvement = (best_val_loss - val_loss) / best_val_loss * 100 if best_val_loss != float('inf') else 100
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                self.save_model("best_model.pt")
                
                # Store best weights in memory to avoid disk I/O
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                
                print(f"New best model! Improvement: {improvement:.2f}%")
                
                # Create visualizations for significant improvements or on schedule
                if improvement > 5 or epoch in vis_schedule:
                    print(f"Generating visualizations at epoch {epoch+1}...")
                    self.visualize_samples(father_latents, mother_latents, child_images, 
                                          test_indices[:3], epoch)
            else:
                patience_counter += 1
            
            # Print progress with more detailed metrics
            epoch_time = time.time() - start_time
            lr_change = f", LR changed: {old_lr:.8f} → {new_lr:.8f}" if old_lr != new_lr else ""
            
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f} (Smooth: {smooth_val_loss:.6f}), "
                  f"Best: {best_val_loss:.6f} @ {best_epoch+1}, "
                  f"Patience: {patience_counter}/{max_patience}, "
                  f"Time: {epoch_time:.2f}s"
                  f"{lr_change}")
            
            # Early stopping check with more informative message
            if patience_counter >= max_patience:
                print(f"Early stopping triggered after {epoch+1} epochs. "
                      f"No improvement for {max_patience} epochs since best epoch {best_epoch+1}.")
                break
                
        # Plot and save training history
        self.plot_training_history()
        
        # Load best model weights from memory (faster than from disk)
        if best_weights is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_weights.items()})
            print(f"Loaded best model weights from epoch {best_epoch+1}")
        else:
            self.load_model("best_model.pt")
            print("Loaded best model from disk")
        
        return self.model, {'train_losses': self.train_losses, 'val_losses': self.val_losses, 'best_epoch': best_epoch}
    
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
                
                try:
                    # Generate child latent based on model type
                    if self.model_type == 'weight':
                        # Get weights for this specific pair
                        weights = self.model(father_latent, mother_latent)
                        
                        # Blend latents
                        combined_latent = self._combine_latents_with_weights(
                            father_latent, mother_latent, weights
                        )
                        
                        # Visualize weight heatmap
                        self.visualize_weight_heatmap(weights.squeeze(0), viz_dir, f'sample_{i}_weights')
                    else:  # edit model
                        # Get edit parameters
                        edit_params = self.model(father_latent, mother_latent)
                        
                        # Create base latent (average of parents)
                        base_latent = (father_latent + mother_latent) / 2.0
                        
                        # Apply edits using the processor
                        combined_latent = self.model.generate_child_latent(
                            father_latent, mother_latent, self.processor
                        )
                        
                        # Visualize edit parameters
                        if hasattr(self.model, 'direction_names'):
                            params_dict = {
                                name: edit_params.squeeze(0)[j].item() 
                                for j, name in enumerate(self.model.direction_names)
                            }
                            self.visualize_edit_parameters_chart(
                                params_dict, viz_dir, f'sample_{i}_params'
                            )
                    
                    # Generate actual image using the real StyleGAN generator (non-differentiable)
                    # For visualization only, not for training
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
                except Exception as e:
                    print(f"Error generating visualization: {e}")
    
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
    
    def visualize_edit_parameters_chart(self, params_dict, output_dir, name):
        """
        Create bar chart visualization of edit parameters.
        
        Args:
            params_dict: Dictionary of parameter names to values
            output_dir: Directory to save the visualization
            name: Base filename for the visualization
        """
        # Create bar chart
        plt.figure(figsize=(10, 6))
        directions = list(params_dict.keys())
        values = list(params_dict.values())
        
        bars = plt.bar(directions, values, color=['skyblue', 'lightgreen', 'salmon'])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            offset = 0.1 if height >= 0 else -0.3
            plt.text(bar.get_x() + bar.get_width()/2., height + offset,
                   f'{height:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        # Add horizontal line at y=0
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Set labels and title
        plt.ylabel('Parameter Value')
        plt.title('Predicted Edit Parameters')
        
        # Set y-axis range to include all values with some padding
        max_abs_val = max(abs(v) for v in values)
        y_range = max(3.5, max_abs_val * 1.2)  # At least -3.5 to 3.5 or 20% more than max
        plt.ylim(-y_range, y_range)
        
        # Add grid lines
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Save figure
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
            'latent_shape': self.latent_shape,
            'model_type': self.model_type
        }, save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, filename):
        """
        Load a saved model.
        
        Args:
            filename (str): Filename to load model from
            
        Returns:
            BaseGeneticModel: Loaded model
        """
        load_path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(load_path)
        
        # Check if the model type matches
        saved_model_type = checkpoint.get('model_type', 'weight')  # Default to 'weight' for backward compatibility
        if saved_model_type != self.model_type:
            print(f"Warning: Loaded model type '{saved_model_type}' doesn't match current type '{self.model_type}'")
            print("Initializing a new model of the correct type...")
            
            # Re-initialize the model with the correct type
            self.model_type = saved_model_type
            if saved_model_type == 'weight':
                self.model = LatentWeightGenerator(self.latent_shape).to(self.device)
            else:
                self.model = EditParamGenerator(self.latent_shape).to(self.device)
        
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
        if self.model_type == 'weight':
            child_latent = self.model.generate_child_latent(father_latent, mother_latent)
        else:  # edit model
            child_latent = self.model.generate_child_latent(father_latent, mother_latent, self.processor)
        
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

class EditParamModel(nn.Module):
    """
    Model to predict InterfaceGAN edit parameters for generating a child face.
    This model predicts parameters for three editing directions: age, smile, and pose.
    """
    def __init__(self, latent_size):
        """
        Initialize the model.
        
        Args:
            latent_size (int): Total size of flattened latent code
        """
        super(EditParamModel, self).__init__()
        
        # Input size is 2x latent_size (father + mother)
        input_dim = 2 * latent_size
        
        # Encoder network
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
        
        # Attention module
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
        
        # Output layer for the three edit parameters (age, smile, pose)
        # Each parameter will be in the range [-3, 3]
        self.edit_params_decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 3),
            nn.Tanh()  # Output in range [-1, 1], will be scaled to [-3, 3]
        )
        
    def forward(self, father_latent, mother_latent):
        """
        Forward pass to generate edit parameters.
        
        Args:
            father_latent (torch.Tensor): Father's latent code
            mother_latent (torch.Tensor): Mother's latent code
            
        Returns:
            torch.Tensor: Edit parameters for age, smile, and pose
        """
        # Flatten latents
        batch_size = father_latent.size(0)
        father_flat = father_latent.view(batch_size, -1)
        mother_flat = mother_latent.view(batch_size, -1)
        
        # Concatenate parent features
        combined = torch.cat([father_flat, mother_flat], dim=1)
        
        # Apply attention
        attention_weights = self.attention(combined)
        attended_inputs = combined * attention_weights
        
        # Generate edit parameters
        encoded = self.encoder(attended_inputs)
        
        # Output parameters in range [-3, 3]
        params = self.edit_params_decoder(encoded) * 3.0
        
        return params

class EditParamGenerator(BaseGeneticModel):
    """
    Model to predict InterfaceGAN edit parameters for generating a child face.
    Uses edit parameters to modify the average of parent latents.
    """
    def __init__(self, latent_shape):
        """
        Initialize the model.
        
        Args:
            latent_shape (tuple): Shape of the latent codes (typically [18, 512] for StyleGAN)
        """
        super(EditParamGenerator, self).__init__()
        
        # Total size of flattened latent
        latent_size = latent_shape[0] * latent_shape[1]
        
        # Store latent shape for reshaping
        self.latent_shape = latent_shape
        
        # Create model
        self.model = EditParamModel(latent_size)
        
        # Direction names in order of model output
        self.direction_names = ['age', 'smile', 'pose']
        
    def forward(self, father_latent, mother_latent):
        """
        Forward pass to generate edit parameters.
        
        Args:
            father_latent (torch.Tensor): Father's latent code
            mother_latent (torch.Tensor): Mother's latent code
            
        Returns:
            torch.Tensor: Edit parameters for age, smile, and pose
        """
        return self.model(father_latent, mother_latent)
    
    def get_edit_parameters(self, father_latent, mother_latent):
        """
        Get edit parameters with their direction names.
        
        Args:
            father_latent (torch.Tensor): Father's latent code
            mother_latent (torch.Tensor): Mother's latent code
            
        Returns:
            dict: Mapping of direction names to edit parameters
        """
        params = self.forward(father_latent, mother_latent)
        
        # Return parameters with direction names
        if params.dim() > 1 and params.size(0) == 1:
            params = params.squeeze(0)
            
        return {name: params[i].item() for i, name in enumerate(self.direction_names)}