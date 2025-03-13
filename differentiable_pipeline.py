import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class DifferentiableFaceEncoder(nn.Module):
    """
    Face encoder using a pre-trained ResNet50 model to extract face embeddings.
    This class provides differentiable face feature extraction for perceptual loss.
    """
    def __init__(self, embedding_size=512):
        super(DifferentiableFaceEncoder, self).__init__()
        
        # Load pre-trained ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add embedding layer
        self.embedding = nn.Linear(2048, embedding_size)
        
        # Initialize normalization
        self.l2_norm = lambda x: x / torch.norm(x, p=2, dim=1, keepdim=True)
        
        # Define image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, x):
        """
        Extract face embedding from input image tensor
        
        Args:
            x: Input image tensor [B, 3, 224, 224]
            
        Returns:
            Normalized face embedding vectors
        """
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        embedding = self.embedding(features)
        
        # Normalize embedding to unit length
        normalized_embedding = self.l2_norm(embedding)
        return normalized_embedding
    
    def extract_from_pil(self, pil_image):
        """
        Extract embedding directly from PIL image
        
        Args:
            pil_image: PIL.Image object
            
        Returns:
            Face embedding vector
        """
        img_tensor = self.preprocess(pil_image).unsqueeze(0)
        device = next(self.parameters()).device
        img_tensor = img_tensor.to(device)
        
        with torch.no_grad():
            embedding = self(img_tensor)
        
        return embedding 