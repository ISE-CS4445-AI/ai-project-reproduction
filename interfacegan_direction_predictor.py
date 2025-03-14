import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os

class InterfaceGANDirectionPredictor(nn.Module):
    """
    Predicts factors for three InterFaceGAN directions based on parent latents.
    Instead of predicting weights for blending, this model predicts edit factors.
    """
    def __init__(self, latent_shape, direction_names=None):
        super(InterfaceGANDirectionPredictor, self).__init__()
        
        self.latent_shape = latent_shape
        latent_size = latent_shape[0] * latent_shape[1]
        input_dim = 2 * latent_size  # Both parents' full latent codes
        
        # Direction names to use with apply_interfacegan
        self.direction_names = direction_names or ["age", "gender", "smile"]
        self.num_directions = len(self.direction_names)
        
        # Keep encoder similar to AdaptiveDimensionWeightModel
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
        
        # Attention module (keep for highlighting important features)
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
        
        # Direction factor decoder - outputs NUM_DIRECTIONS values
        self.factor_decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, self.num_directions),
            nn.Tanh()  # Output in range [-1, 1], will be scaled later
        )
        
        # Scaling factors for each direction (can be learned or fixed)
        self.register_buffer('factor_scales', torch.ones(self.num_directions))
        
    def forward(self, father_latent, mother_latent):
        # Flatten latents and concatenate
        batch_size = father_latent.size(0)
        father_flat = father_latent.view(batch_size, -1)
        mother_flat = mother_latent.view(batch_size, -1)
        combined = torch.cat([father_flat, mother_flat], dim=1)
        
        # Apply attention to highlight important features
        attention_weights = self.attention(combined)
        attended_inputs = combined * attention_weights
        
        # Generate direction factors
        encoded = self.encoder(attended_inputs)
        raw_factors = self.factor_decoder(encoded)
        
        # Scale factors to appropriate ranges for each direction
        # Typically range might be [-3, 3] or similar
        scaled_factors = raw_factors * self.factor_scales
        
        return scaled_factors

class InterfaceGANTrainer:
    """
    Trainer for the InterfaceGANDirectionPredictor model.
    """
    def __init__(self, model, processor, learning_rate=0.00005, device=None, save_dir='models', weight_decay=1e-5, clip_value=1.0):
        self.model = model
        self.processor = processor
        self.learning_rate = learning_rate
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir
        self.clip_value = clip_value
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            patience=15,
            factor=0.5,
            min_lr=1e-7,
            verbose=True
        )
        
    def train(self, dataloader, num_epochs=100):
        self.model.to(self.device)
        self.model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in dataloader:
                father_latent = batch['father_latent'].to(self.device)
                mother_latent = batch['mother_latent'].to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                factors = self.model(father_latent, mother_latent)
                
                # Apply edits using apply_interfacegan
                edited_latents = []
                for i, direction_name in enumerate(self.model.direction_names):
                    edited_latent = self.processor.apply_interfacegan(
                        father_latent, direction_name, factor=factors[:, i]
                    )
                    edited_latents.append(edited_latent)
                
                # Compute loss (example: perceptual loss between edited and target embeddings)
                # This part will depend on your specific loss function and target data
                # loss = ...
                
                # Backward pass
                # loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
                
                # Optimize
                self.optimizer.step()
                
                # Update total loss
                # total_loss += loss.item()
            
            # Scheduler step
            # self.scheduler.step(total_loss)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss}")
        
    def save_model(self, filename):
        save_path = os.path.join(self.save_dir, filename)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, filename):
        load_path = os.path.join(self.save_dir, filename)
        self.model.load_state_dict(torch.load(load_path))
        print(f"Model loaded from {load_path}") 