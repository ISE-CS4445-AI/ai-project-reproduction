import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os

class InterfaceGANParamModel(nn.Module):
    """
    Model to predict parameters for InterfaceGAN edits.
    Outputs parameters for 'age', 'smile', and 'pose' directions.
    """
    def __init__(self, latent_shape):
        """
        Initialize the model.
        
        Args:
            latent_shape (tuple): Shape of the latent codes (typically [18, 512] for StyleGAN)
        """
        super(InterfaceGANParamModel, self).__init__()
        
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
        
        # Output layer for InterfaceGAN parameters
        self.param_decoder = nn.Sequential(
            nn.Linear(512, 3),  # Output three parameters
            nn.Tanh()  # Output range between -1 and 1
        )
        
    def forward(self, father_latent, mother_latent):
        """
        Forward pass to generate InterfaceGAN parameters.
        
        Args:
            father_latent (torch.Tensor): Father's latent code
            mother_latent (torch.Tensor): Mother's latent code
            
        Returns:
            torch.Tensor: Parameters for 'age', 'smile', and 'pose'
        """
        # Flatten latents and concatenate
        batch_size = father_latent.size(0)
        father_flat = father_latent.view(batch_size, -1)
        mother_flat = mother_latent.view(batch_size, -1)
        combined = torch.cat([father_flat, mother_flat], dim=1)
        
        # Encode features
        encoded = self.encoder(combined)
        
        # Decode parameters
        params = self.param_decoder(encoded)
        
        # Scale parameters to range [-3, 3]
        params = params * 3
        
        return params

# Example usage
if __name__ == "__main__":
    # Example latent shape for StyleGAN
    latent_shape = (18, 512)
    
    # Initialize model
    model = InterfaceGANParamModel(latent_shape)
    
    # Example latents
    father_latent = torch.randn(1, *latent_shape)
    mother_latent = torch.randn(1, *latent_shape)
    
    # Generate parameters
    params = model(father_latent, mother_latent)
    print("Generated InterfaceGAN parameters:", params) 