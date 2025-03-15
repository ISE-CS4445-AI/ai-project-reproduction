"""
Encoder4Editing (e4e) Library
-----------------------------
A modular library for face encoding and editing based on the encoder4editing project.

Main components:
- Setup: Repository setup and model loading
- Preprocessing: Image loading and alignment
- Inference: Running inference with the e4e model
- Combination: Methods to combine latent codes from multiple images
- Editing: Various editing methods (InterFaceGAN, GANSpace, SeFa)
"""

import os
import sys
import time
import glob
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from argparse import Namespace
from typing import Dict, List, Tuple, Union, Optional
import logging
from tqdm import tqdm


class E4ESetup:
    """Handles setup and file path management for the e4e encoder."""
    
    def __init__(self, experiment_type='ffhq_encode', base_dir=None):
        """
        Initialize the setup.
        
        Args:
            experiment_type (str): Type of experiment ('ffhq_encode', 'cars_encode', etc.)
            base_dir (str): Base directory for the e4e project. If None, uses current directory
        """
        self.experiment_type = experiment_type
        self.base_dir = base_dir or os.getcwd()
        
        # Define possible paths - will check these in order
        self.code_dir = os.path.join(self.base_dir, 'encoder4editing')
        self.models_dir = os.path.join(self.code_dir, 'pretrained_models')
        self.alignments_dir = self.base_dir
        self.editings_dir = os.path.join(self.code_dir, 'editings')
        
        # Alternative directories to check for models
        self.alt_models_dirs = [
            os.path.join(self.base_dir, 'pretrained_models'),  # ./pretrained_models/
            'pretrained_models',  # ./pretrained_models/
            os.path.join(os.getcwd(), 'pretrained_models')  # <current_dir>/pretrained_models/
        ]
        
        # Alternative directories to check for face predictor
        self.alt_face_predictor_dirs = [
            self.base_dir,  # Base directory
            'pretrained_models',  # ./pretrained_models/
            os.path.join(os.getcwd(), 'pretrained_models'),  # <current_dir>/pretrained_models/
            os.getcwd()  # Current directory
        ]
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.editings_dir, exist_ok=True)
        
        # Path mappings with alternatives
        self.MODEL_FILENAMES = {
            "ffhq_encode": "e4e_ffhq_encode.pt",
            "cars_encode": "e4e_cars_encode.pt",
            "horse_encode": "e4e_horse_encode.pt",
            "church_encode": "e4e_church_encode.pt"
        }
        
        # Configure necessary paths based on experiment type
        self._configure_paths()
    
    def _configure_paths(self):
        """Configure necessary file paths based on experiment type."""
        # Set model path
        if self.experiment_type not in self.MODEL_FILENAMES:
            raise ValueError(f"Experiment type {self.experiment_type} not supported")
        
        model_filename = self.MODEL_FILENAMES[self.experiment_type]
        
        # Initialize with the default path
        self.model_path = os.path.join(self.models_dir, model_filename)
        
        # Check if file exists at default location, if not, try alternative locations
        if not os.path.isfile(self.model_path):
            for alt_dir in self.alt_models_dirs:
                alt_path = os.path.join(alt_dir, model_filename)
                if os.path.isfile(alt_path):
                    self.model_path = alt_path
                    print(f"Found model at alternative location: {alt_path}")
                    break
        
        # Set face predictor path for face alignment
        self.face_predictor_path = os.path.join(self.alignments_dir, "shape_predictor_68_face_landmarks.dat")
        
        # Check if face predictor exists at default location, if not, try alternative locations
        if not os.path.isfile(self.face_predictor_path):
            for alt_dir in self.alt_face_predictor_dirs:
                alt_path = os.path.join(alt_dir, "shape_predictor_68_face_landmarks.dat")
                if os.path.isfile(alt_path):
                    self.face_predictor_path = alt_path
                    print(f"Found face predictor at alternative location: {alt_path}")
                    break
        
        # Set editing paths
        self.interfacegan_dir = os.path.join(self.editings_dir, "interfacegan_directions")
        self.ganspace_dir = os.path.join(self.editings_dir, "ganspace_pca")
        
        # Create directories for editing files
        os.makedirs(self.interfacegan_dir, exist_ok=True)
        os.makedirs(self.ganspace_dir, exist_ok=True)
    
    def check_model_exists(self):
        """
        Check if the required model file exists.
        
        Returns:
            bool: True if the model file exists, False otherwise
        """
        return os.path.isfile(self.model_path)
    
    def check_face_predictor_exists(self):
        """
        Check if the face predictor file exists.
        
        Returns:
            bool: True if the face predictor file exists, False otherwise
        """
        return os.path.isfile(self.face_predictor_path)
    
    def get_download_instructions(self):
        """
        Get instructions for downloading required files.
        
        Returns:
            str: Instructions for downloading required files
        """
        model_filename = self.MODEL_FILENAMES[self.experiment_type]
        
        return f"""
        To use this library, you need to manually download the following files:
        
        1. Pretrained Model: {model_filename}
           - Place it in one of these directories:
             * {self.models_dir}
             * {os.path.join(os.getcwd(), 'pretrained_models')}
             * {os.path.join(self.base_dir, 'pretrained_models')}
        
        2. Face Alignment Predictor: shape_predictor_68_face_landmarks.dat
           - Place it in one of these directories:
             * {self.alignments_dir}
             * {os.path.join(os.getcwd(), 'pretrained_models')}
             * {os.getcwd()}
           - Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
           - Extract with: bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2
        
        You can download the model using gdown:
        gdown 1cUv_reLE6k3604or78EranS7XzuVMWeO -O {os.path.join(os.getcwd(), 'pretrained_models', model_filename)}
        """


class E4EPreprocessor:
    """Handles image preprocessing for the e4e encoder."""
    
    def __init__(self, experiment_type, setup=None):
        """
        Initialize the preprocessor.
        
        Args:
            experiment_type (str): Type of experiment ('ffhq_encode', 'cars_encode', etc.)
            setup (E4ESetup, optional): Setup object with path configuration
        """
        self.experiment_type = experiment_type
        self.setup = setup
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Set up the image transformations based on experiment type."""
        if self.experiment_type == 'cars_encode':
            self.transform = transforms.Compose([
                transforms.Resize((192, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            self.resize_dims = (256, 192)
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            self.resize_dims = (256, 256)
    
    def align_face(self, image_path):
        """
        Align a face image using facial landmarks.
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            PIL.Image: Aligned face image
        """
        if self.experiment_type != "ffhq_encode":
            # No alignment needed for non-face datasets
            return Image.open(image_path)
        
        # Check if predictor file exists
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if self.setup:
            predictor_path = self.setup.face_predictor_path
        
        if not os.path.isfile(predictor_path):
            raise FileNotFoundError(
                f"Face predictor file not found at: {predictor_path}\n"
                "Please download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            )
        
        # Import dlib and alignment utils
        try:
            import dlib
            # Add the repository to the path to import the necessary modules
            sys.path.append(".")
            sys.path.append("..")
            
            # Import alignment utils from our standalone file
            from face_alignment import align_face
            
            predictor = dlib.shape_predictor(predictor_path)
            aligned_image = align_face(filepath=image_path, predictor=predictor) 
            print(f"Aligned image has shape: {aligned_image.size}")
            
            return aligned_image
        except ImportError:
            print("Warning: dlib not installed. Face alignment will be skipped.")
            print("To install dlib: pip install dlib")
            return Image.open(image_path)
        except Exception as e:
            print(f"Warning: Face alignment failed: {e}")
            print("Using original image instead.")
            return Image.open(image_path)
    
    def preprocess_image(self, image_path):
        """
        Preprocess an image for inference.
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            torch.Tensor: Preprocessed image tensor
            PIL.Image: Processed image for display
        """
        # Align image if needed
        if self.experiment_type == "ffhq_encode":
            input_image = self.align_face(image_path)
        else:
            input_image = Image.open(image_path)
        
        # Resize
        input_image = input_image.resize(self.resize_dims)
        
        # Transform
        transformed_image = self.transform(input_image)
        
        # Ensure tensor is float32 to match input requirements
        transformed_image = transformed_image.float()
        
        return transformed_image, input_image


class E4EInference:
    """Handles inference with the e4e encoder."""
    
    def __init__(self, model_path, experiment_type, memory_efficient=False, max_batch_size=1):
        """
        Initialize the inference module.
        
        Args:
            model_path (str): Path to the pretrained model
            experiment_type (str): Type of experiment ('ffhq_encode', 'cars_encode', etc.)
            memory_efficient (bool): Whether to use memory-efficient mode
            max_batch_size (int): Maximum batch size to use during inference
        """
        self.model_path = model_path
        self.experiment_type = experiment_type
        self.memory_efficient = memory_efficient
        self.max_batch_size = max_batch_size
        
        # Check if model file exists
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model file not found at: {model_path}\n"
                "Please make sure the model file is in the correct location."
            )
        
        # Load the model
        self.net = self._load_model()
        # Disable mixed precision by default as it causes dtype mismatches
        self.use_mixed_precision = False
        
        # Set environment variable to avoid memory fragmentation
        if torch.cuda.is_available():
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    def _load_model(self):
        """
        Load the pretrained model.
        
        Returns:
            torch.nn.Module: Loaded model
        """
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
        
        logger.info("Starting model loading process...")
        
        # Add the repository to the path to import the necessary modules
        logger.info("Configuring Python path...")
        sys.path.append(".")
        sys.path.append("..")
        
        # Add encoder4editing directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        encoder4editing_dir = os.path.join(current_dir, "encoder4editing")
        sys.path.append(encoder4editing_dir)
        logger.info(f"Added encoder4editing directory to path: {encoder4editing_dir}")
        
        try:
            # Import the pSp model
            logger.info("Attempting to import pSp model...")
            from models.psp import pSp
            logger.info("Successfully imported pSp model")
            
            # Load the checkpoint with progress bar
            logger.info(f"Loading checkpoint from {self.model_path}")
            start_time = time.time()
            
            with tqdm(total=1, desc="Loading checkpoint", unit="file") as pbar:
                # Load on CPU to save GPU memory
                ckpt = torch.load(self.model_path, map_location='cpu')
                pbar.update(1)
            
            load_time = time.time() - start_time
            logger.info(f"Checkpoint loaded in {load_time:.2f} seconds")
            
            # Update the training options
            logger.info("Configuring model options...")
            opts = ckpt['opts']
            opts['checkpoint_path'] = self.model_path
            opts = Namespace(**opts)
            logger.debug(f"Model options configured: {vars(opts)}")
            
            # Initialize the model with progress
            logger.info("Initializing pSp model...")
            with tqdm(total=3, desc="Model initialization", unit="step") as pbar:
                # Step 1: Create model instance
                net = pSp(opts)
                pbar.update(1)
                
                # Step 2: Set to eval mode
                net.eval()
                pbar.update(1)
                
                # Step 3: Move to GPU if available
                if torch.cuda.is_available() and not self.memory_efficient:
                    logger.info("CUDA is available. Moving model to GPU...")
                    net.cuda()
                    logger.info(f"Model moved to GPU: {next(net.parameters()).device}")
                else:
                    if self.memory_efficient:
                        logger.info("Running in memory-efficient mode. Model will remain on CPU and only required parts will be moved to GPU during inference.")
                    else:
                        logger.warning("CUDA is not available. Model will run on CPU")
                pbar.update(1)
            
            # Log model statistics
            total_params = sum(p.numel() for p in net.parameters())
            trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            logger.info(f"Model loaded successfully!")
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            
            return net
            
        except ImportError as e:
            logger.error(f"Import error occurred: {str(e)}")
            logger.error(f"Current sys.path: {sys.path}")
            raise ImportError(f"Failed to import required modules: {e}\n"
                            "Make sure you have cloned the encoder4editing repository "
                            "and have all the required dependencies installed.")
        except Exception as e:
            logger.error(f"Unexpected error occurred: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            raise RuntimeError(f"Failed to load model: {e}")
    
    def run_on_batch(self, inputs):
        """
        Run inference on a batch of images.
        
        Args:
            inputs (torch.Tensor): Batch of preprocessed images
            
        Returns:
            tuple: Tuple of (generated images, latent codes)
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_tensor = inputs.to(device).float()
        
        # If in memory-efficient mode, move the entire model to the appropriate device temporarily
        if self.memory_efficient and device == "cuda":
            # Move all model components to GPU for the forward pass
            self.net.to(device)
            was_on_cpu = True
        else:
            was_on_cpu = False
        
        # Ensure the entire model uses float32 precision
        self.net = self.net.float()
        
        # Process input using float32 precision
        images, latents = self.net(input_tensor, randomize_noise=False, return_latents=True)
        
        # Move model back to CPU if it was moved
        if was_on_cpu:
            # Move the entire model back to CPU after inference
            self.net.to('cpu')
            # Ensure that the results stay on the device
            images = images.to(device)
            latents = latents.to(device)
            # Explicit garbage collection
            torch.cuda.empty_cache()
        
        if self.experiment_type == 'cars_encode':
            # Crop the cars images as needed
            images = images[:, :, 32:224, :]
        
        return images, latents
    
    def infer(self, transformed_image):
        """
        Run inference on a single preprocessed image.
        
        Args:
            transformed_image (torch.Tensor): Preprocessed image tensor
            
        Returns:
            tuple: Tuple of (generated image, latent code)
        """
        logger = logging.getLogger(__name__)
        
        # Clear CUDA cache before inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            with torch.no_grad():
                logger.info("Running inference on input image...")
                # Ensure input image is float32 
                transformed_image = transformed_image.float()
                tic = time.time()
                images, latents = self.run_on_batch(transformed_image.unsqueeze(0))
                toc = time.time()
                logger.info(f'Inference took {toc - tic:.4f} seconds.')
            
            # Convert tensor to PIL Image
            logger.info("Converting tensor to PIL Image...")
            try:
                # First try to use the official tensor2im method
                sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "encoder4editing"))
                from utils.common import tensor2im
                logger.info("Using encoder4editing's tensor2im for conversion")
                
                # Add detailed logging for debugging
                logger.debug(f"Image tensor shape: {images[0].shape}")
                logger.debug(f"Image tensor type: {images[0].dtype}")
                logger.debug(f"Image tensor device: {images[0].device}")
                
                try:
                    result_image = tensor2im(images[0])
                    logger.info("Tensor successfully converted to PIL Image")
                except Exception as conversion_error:
                    logger.error(f"Error in tensor2im conversion: {str(conversion_error)}")
                    logger.error("Falling back to manual conversion method")
                    # Fallback conversion
                    result_tensor = images[0].detach().cpu().float().numpy()
                    result_tensor = np.transpose(result_tensor, (1, 2, 0))
                    result_tensor = (result_tensor + 1) / 2.0
                    result_tensor = np.clip(result_tensor, 0, 1)
                    result_array = (result_tensor * 255).astype(np.uint8)
                    result_image = Image.fromarray(result_array)
                    logger.info("Tensor converted to PIL Image using fallback method")
            except ImportError:
                # Fallback conversion if utils.common is not available
                logger.warning("Could not import tensor2im from utils.common, using fallback conversion method")
                result_tensor = images[0].detach().cpu().float().numpy()
                result_tensor = np.transpose(result_tensor, (1, 2, 0))
                result_tensor = (result_tensor + 1) / 2.0
                result_tensor = np.clip(result_tensor, 0, 1)
                result_array = (result_tensor * 255).astype(np.uint8)
                result_image = Image.fromarray(result_array)
                logger.info("Tensor converted to PIL Image using fallback method")
            
            return result_image, latents[0]
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            # Make one last attempt to clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        finally:
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class E4ELatentCombiner:
    """Combines latent codes from different images."""
    
    def __init__(self, decoder, experiment_type):
        """
        Initialize the latent combiner.
        
        Args:
            decoder (torch.nn.Module): StyleGAN decoder from the e4e model
            experiment_type (str): Type of experiment ('ffhq_encode', 'cars_encode', etc.)
        """
        self.decoder = decoder
        self.experiment_type = experiment_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _ensure_same_device(self, tensors):
        """
        Ensure all tensors are on the same device.
        
        Args:
            tensors (list): List of tensors to check/move
            
        Returns:
            list: List of tensors all on the same device
        """
        if not tensors:
            return tensors
            
        target_device = tensors[0].device
        return [t.to(target_device) for t in tensors]
    
    def combine_latents(self, latents_list, weights=None):
        """
        Combine multiple latent codes with optional weights.
        
        Args:
            latents_list (list): List of latent codes
            weights (list or torch.Tensor, optional): Weights for each latent. Can be:
                - None: Equal weights for all latents
                - List of scalars: One weight per latent (applied to entire latent)
                - List of tensors: Each tensor has the same shape as the latent for per-dimension weighting
                - Single tensor of shape [18, 512]: StyleGAN specific weighting tensor that will be 
                  automatically converted to appropriate weights for each latent
            
        Returns:
            torch.Tensor: Combined latent code
        """
        # Ensure all latents are on the same device
        latents_list = self._ensure_same_device(latents_list)
        
        # Get shape information from the first latent
        latent_shape = latents_list[0].shape
        num_latents = len(latents_list)
        
        # Handle the case where weights is a single tensor for StyleGAN
        if isinstance(weights, torch.Tensor) and weights.shape == latent_shape:
            # This is a special case where a single StyleGAN weight tensor is provided
            # We need to convert it to a list of weights for each latent
            
            # For the first latent, use the provided weights
            # For all other latents, distribute the remaining weight (1.0 - weights)
            stylegan_weights = []
            stylegan_weights.append(weights)
            
            if num_latents > 1:
                # Calculate remaining weight to be distributed among other latents
                remaining_weight = 1.0 - weights
                # Distribute evenly among remaining latents
                remaining_per_latent = remaining_weight / (num_latents - 1)
                
                for _ in range(num_latents - 1):
                    stylegan_weights.append(remaining_per_latent)
                    
            weights = stylegan_weights
        elif weights is None:
            # Equal weights if not specified
            weights = [1.0 / num_latents] * num_latents
        
        # Ensure we have the correct number of weights
        assert len(weights) == num_latents, "Number of weights must match number of latents"
        
        # Initialize the combined latent
        combined_latent = torch.zeros_like(latents_list[0])
        
        # Check if we're using per-dimension weights or scalar weights
        for i, latent in enumerate(latents_list):
            weight = weights[i]
            
            # Handle different weight types
            if isinstance(weight, (float, int)):
                # Traditional scalar weight (backward compatible)
                combined_latent += latent * weight
            elif isinstance(weight, torch.Tensor):
                # Per-dimension weighting
                # Ensure weight has the right shape
                if weight.shape != latent_shape:
                    raise ValueError(f"Weight tensor shape {weight.shape} doesn't match latent shape {latent_shape}")
                # Ensure weight is on the same device
                weight = weight.to(latent.device)
                combined_latent += latent * weight
            else:
                # Try to convert to tensor if possible
                try:
                    weight_tensor = torch.tensor(weight, dtype=latent.dtype, device=latent.device)
                    if weight_tensor.shape != latent_shape:
                        raise ValueError(f"Weight shape {weight_tensor.shape} doesn't match latent shape {latent_shape}")
                    combined_latent += latent * weight_tensor
                except Exception as e:
                    raise TypeError(f"Weight type not supported: {type(weight)}. Error: {str(e)}")
        
        return combined_latent
    
    def generate_from_latent(self, latent):
        """
        Generate an image from a latent code.
        
        Args:
            latent (torch.Tensor): Latent code
            
        Returns:
            PIL.Image: Generated image
        """
        # Ensure decoder is on the same device as the latent
        device = latent.device
        
        # Temporarily move decoder to the appropriate device
        original_device = next(self.decoder.parameters()).device
        self.decoder.to(device)
        
        try:
            with torch.no_grad():
                # Need to add batch dimension
                latent_code = latent.unsqueeze(0)
                
                # Generate image
                logging.info("Generating image from latent code...")
                images, _ = self.decoder([latent_code], input_is_latent=True, randomize_noise=False, return_latents=True)
                
                # Crop cars images if needed
                if self.experiment_type == 'cars_encode':
                    images = images[:, :, 32:224, :]
                
                # Convert tensor to PIL Image
                try:
                    # Try to use the official tensor2im method
                    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "encoder4editing"))
                    from utils.common import tensor2im
                    result_image = tensor2im(images[0])
                except ImportError:
                    # Fallback conversion
                    result_tensor = images[0].detach().cpu().float().numpy()
                    result_tensor = np.transpose(result_tensor, (1, 2, 0))
                    result_tensor = (result_tensor + 1) / 2.0
                    result_tensor = np.clip(result_tensor, 0, 1)
                    result_array = (result_tensor * 255).astype(np.uint8)
                    result_image = Image.fromarray(result_array)
            
            return result_image
        finally:
            # Move decoder back to original device
            self.decoder.to(original_device)
            # Clear GPU memory if needed
            if device.type == 'cuda':
                torch.cuda.empty_cache()


class E4EEditor:
    """Handles editing operations on the latent space."""
    
    def __init__(self, decoder, experiment_type, setup=None):
        """
        Initialize the editor.
        
        Args:
            decoder (torch.nn.Module): StyleGAN decoder from the e4e model
            experiment_type (str): Type of experiment ('ffhq_encode', 'cars_encode', etc.)
            setup (E4ESetup, optional): Setup object for file paths
        """
        self.decoder = decoder
        self.experiment_type = experiment_type
        self.setup = setup
        self.is_cars = experiment_type == 'cars_encode'
        self.memory_efficient = False
        
        # Add the repository to the path to import the necessary modules
        sys.path.append(".")
        sys.path.append("..")
        
        # Load directions if available
        self.interfacegan_directions = None
        self.ganspace_directions = None
        
        try:
            # Import the latent editor
            from editings import latent_editor
            self.editor = latent_editor.LatentEditor(decoder, self.is_cars)
            
            # Import tensor to image conversion function
            try:
                from utils.common import tensor2im
                self.tensor2im = tensor2im
            except ImportError:
                logging.warning("Could not import tensor2im, will use a fallback implementation")
                self.tensor2im = self._fallback_tensor2im
        except ImportError:
            logging.warning("Failed to import latent editor. Editing capabilities will be limited.")
            self.editor = None
            self.tensor2im = self._fallback_tensor2im
            
    def _fallback_tensor2im(self, tensor):
        """
        Fallback implementation of tensor2im if the original is not available.
        
        Args:
            tensor (torch.Tensor): Image tensor
            
        Returns:
            PIL.Image: PIL Image
        """
        # Ensure tensor is on CPU and detached from grad
        tensor = tensor.cpu().detach()
        
        # Normalize from [-1, 1] to [0, 1]
        tensor = (tensor + 1) / 2.0
        
        # Clamp values to [0, 1]
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to numpy and transpose
        arr = tensor.numpy().transpose(1, 2, 0)
        
        # Convert to uint8
        arr = (arr * 255).astype('uint8')
        
        # Convert to PIL Image
        img = Image.fromarray(arr)
        
        return img
    
    def set_memory_efficient(self, memory_efficient=True):
        """
        Set memory-efficient mode.
        
        Args:
            memory_efficient (bool): Whether to use memory-efficient mode
        """
        self.memory_efficient = memory_efficient
        logging.info(f"Editor memory-efficient mode set to: {memory_efficient}")
        
    def _clear_gpu_memory(self):
        """Clear GPU memory cache to free up memory."""
        if torch.cuda.is_available():
            # Force garbage collection first
            import gc
            gc.collect()
            # Then empty CUDA cache
            torch.cuda.empty_cache()
    
    def _ensure_tensor_on_device(self, tensor, device=None):
        """
        Ensure a tensor is on the specified device.
        
        Args:
            tensor (torch.Tensor): Tensor to ensure is on the right device
            device (torch.device, optional): Target device (defaults to tensor's current device)
            
        Returns:
            torch.Tensor: Tensor on the specified device
        """
        if device is None:
            device = tensor.device
        
        return tensor.to(device)
    
    def _generate_from_latent(self, latent, return_tensor=False):
        """
        Generate an image from a latent code.
        
        Args:
            latent (torch.Tensor): Latent code
            return_tensor (bool): Whether to return the raw tensor or a PIL Image
            
        Returns:
            PIL.Image or torch.Tensor: Generated image
        """
        # Get the device from the latent
        device = latent.device
        
        # Temporarily move decoder to the same device
        original_device = next(self.decoder.parameters()).device
        self.decoder.to(device)
        
        try:
            with torch.no_grad():
                # Need to add batch dimension if not present
                if latent.dim() == 2:
                    latent_code = latent.unsqueeze(0)
                else:
                    latent_code = latent
                
                # Generate image
                images, _ = self.decoder([latent_code], input_is_latent=True, randomize_noise=False, return_latents=True)
                
                # Crop cars images if needed
                if self.experiment_type == 'cars_encode':
                    images = images[:, :, 32:224, :]
                
                if return_tensor:
                    return images[0]
                
                # Convert tensor to PIL Image
                try:
                    # Try to use the official tensor2im method
                    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "encoder4editing"))
                    from utils.common import tensor2im
                    result_image = tensor2im(images[0])
                except ImportError:
                    # Fallback conversion
                    result_tensor = images[0].detach().cpu().float().numpy()
                    result_tensor = np.transpose(result_tensor, (1, 2, 0))
                    result_tensor = (result_tensor + 1) / 2.0
                    result_tensor = np.clip(result_tensor, 0, 1)
                    result_array = (result_tensor * 255).astype(np.uint8)
                    result_image = Image.fromarray(result_array)
                
                return result_image
        finally:
            # Move decoder back to original device
            self.decoder.to(original_device)
            # Clear GPU memory if needed
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    def get_interfacegan_directions(self):
        """
        Get available InterFaceGAN directions for the current experiment type.
        
        Returns:
            dict: Dictionary of available directions
        """
        if self.interfacegan_directions is not None:
            return self.interfacegan_directions
            
        if self.setup:
            base_dir = self.setup.interfacegan_dir
        else:
            base_dir = 'editings/interfacegan_directions'
        
        interfacegan_directions = {
            'ffhq_encode': {
                'age': os.path.join(base_dir, 'age.pt'),
                'smile': os.path.join(base_dir, 'smile.pt'),
                'pose': os.path.join(base_dir, 'pose.pt')
            }
        }
        
        if self.experiment_type in interfacegan_directions:
            # Verify files exist
            directions = {}
            for name, path in interfacegan_directions[self.experiment_type].items():
                if os.path.isfile(path):
                    directions[name] = path
                else:
                    logging.warning(f"InterFaceGAN direction '{name}' not found at: {path}")
            
            self.interfacegan_directions = directions
            return directions
        else:
            self.interfacegan_directions = {}
            return {}
    
    def apply_interfacegan(self, latent, direction_name, factor=1.0, return_latent=False):
        """
        Apply an InterFaceGAN direction to a latent code.
        
        Args:
            latent (torch.Tensor): Latent code
            direction_name (str): Name of the direction to apply
            factor (float): Strength of the edit
            return_latent (bool): If True, return the edited latent code instead of generating an image
            
        Returns:
            PIL.Image or torch.Tensor: Edited image or edited latent code, depending on return_latent
        """
        logger = logging.getLogger(__name__)
        
        if self.editor is None:
            raise RuntimeError("Latent editor not initialized. Cannot apply edits.")
        
        directions = self.get_interfacegan_directions()
        if not directions or direction_name not in directions:
            raise ValueError(f"Direction {direction_name} not available for {self.experiment_type}")
        
        logger.info(f"Applying {direction_name} edit with factor {factor}...")
        
        # Get original device
        original_device = latent.device
        
        # For StyleGAN2's custom CUDA ops, we must use a GPU
        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for applying edits with InterFaceGAN")
        
        # Clear any leftover GPU memory before the operation
        self._clear_gpu_memory()
        
        try:
            # Move to the preferred device
            device = torch.device('cuda')
            
            # Load direction tensor
            direction_path = directions[direction_name]
            direction_tensor = torch.load(direction_path, map_location=device)
            
            # Move latent to GPU if it's not already there
            if latent.device.type != 'cuda':
                latent = latent.to(device)
            
            # When using memory-efficient mode, process in smaller batches
            if self.memory_efficient:
                logger.info("Using memory-efficient mode for GPU editing")
                
                # Clear GPU memory first
                self._clear_gpu_memory()
                
                # Apply edit
                edit_latents = latent + direction_tensor * factor
                
                # If we only need the latent, return it directly
                if return_latent:
                    # Move result back to original device if needed
                    if edit_latents.device != original_device:
                        edit_latents = edit_latents.to(original_device)
                    return edit_latents
                
                # Make sure editor generator is entirely on the GPU
                if hasattr(self.editor, 'generator'):
                    logger.info("Moving editor generator to CUDA...")
                    # Move entire generator to CUDA using our helper
                    self.editor.generator = self._ensure_model_on_device(self.editor.generator, device)
                
                # Process the edit with the latent editor on GPU
                with torch.no_grad():
                    # Use the built-in editor for the edit, which uses custom CUDA ops
                    if hasattr(self.editor, 'apply_interfacegan'):
                        # First make sure latent & direction are on the same device as the editor
                        if hasattr(self.editor, 'generator'):
                            editor_device = next(self.editor.generator.parameters()).device
                            latent = latent.to(editor_device)
                            direction_tensor = direction_tensor.to(editor_device)
                            
                        result_image = self.editor.apply_interfacegan(latent, direction_tensor, factor=factor)
                    else:
                        # Direct approach using generator
                        if hasattr(self.editor, 'generator'):
                            # Apply edit directly
                            edit_latents = latent + direction_tensor * factor
                            
                            # Generate image
                            images, _ = self.editor.generator(
                                [edit_latents], 
                                randomize_noise=False, 
                                input_is_latent=True
                            )
                            result_image = self.tensor2im(images[0])
                        else:
                            raise RuntimeError("Editor does not have generator attribute")
                
                # Move result back to original device if needed
                if isinstance(result_image, torch.Tensor) and result_image.device != original_device:
                    result_image = result_image.to(original_device)
                
                # Clear GPU memory after processing
                self._clear_gpu_memory()
                
                return result_image
            else:
                # Standard approach
                # Apply edit directly for consistent latent handling
                edit_latents = latent + direction_tensor * factor
                
                # If we only need the latent, return it directly
                if return_latent:
                    # Move result back to original device if needed
                    if edit_latents.device != original_device:
                        edit_latents = edit_latents.to(original_device)
                    return edit_latents
                
                # Make sure editor generator is entirely on the GPU
                if hasattr(self.editor, 'generator'):
                    logger.info("Moving editor generator to CUDA...")
                    # Move entire generator to CUDA using our helper
                    self.editor.generator = self._ensure_model_on_device(self.editor.generator, device)
                
                with torch.no_grad():
                    # First make sure latent & direction are on the same device as the editor
                    if hasattr(self.editor, 'generator'):
                        editor_device = next(self.editor.generator.parameters()).device
                        latent = latent.to(editor_device)
                        direction_tensor = direction_tensor.to(editor_device)
                
                    # Use the built-in editor for the edit
                    result_image = self.editor.apply_interfacegan(latent, direction_tensor, factor=factor)
                
                # Move result back to original device if needed
                if isinstance(result_image, torch.Tensor) and result_image.device != original_device:
                    result_image = result_image.to(original_device)
                
                return result_image
                
        except RuntimeError as e:
            if 'out of memory' in str(e):
                logger.warning("Out of memory during generation. Trying with aggressive memory management...")
                
                # Clear all GPU memory
                self._clear_gpu_memory()
                
                # Only keep the essential parts of the model on GPU
                latent = latent.to('cuda')
                direction_tensor = torch.load(direction_path, map_location='cuda')
                
                # Apply the edit directly
                edit_latents = latent + direction_tensor * factor
                
                # If we only need the latent, return it directly
                if return_latent:
                    # Move result back to original device if needed
                    if edit_latents.device != original_device:
                        edit_latents = edit_latents.to(original_device)
                    return edit_latents
                
                # Run generation with minimal memory usage
                with torch.no_grad():
                    try:
                        # Use the built-in editor for the edit, but with minimal memory
                        if hasattr(self.editor, 'generator'):
                            # Move fully to GPU and ensure consistency using our helper
                            self.editor.generator = self._ensure_model_on_device(self.editor.generator, 'cuda')
                            
                            images, _ = self.editor.generator([edit_latents], randomize_noise=False, input_is_latent=True)
                            result_image = self.tensor2im(images[0])
                            
                            # Clear GPU memory
                            self._clear_gpu_memory()
                            
                            return result_image
                        else:
                            raise RuntimeError("Editor does not have generator attribute")
                    except Exception as inner_e:
                        logger.error(f"Error during memory-efficient editing: {str(inner_e)}")
                        raise
            else:
                raise
    
    def apply_interfacegan_range(self, latent, direction_name, factor_range=(-5, 5), steps=10):
        """
        Apply an InterFaceGAN direction with a range of factors.
        
        Args:
            latent (torch.Tensor): Latent code
            direction_name (str): Name of the direction to apply
            factor_range (tuple): Range of factors (min, max)
            steps (int): Number of steps in the range
            
        Returns:
            list: List of edited images as PIL Images
        """
        logger = logging.getLogger(__name__)
        
        if self.editor is None:
            raise RuntimeError("Latent editor not initialized. Cannot apply edits.")
        
        directions = self.get_interfacegan_directions()
        if not directions or direction_name not in directions:
            raise ValueError(f"Direction {direction_name} not available for {self.experiment_type}")
        
        logger.info(f"Applying {direction_name} edit with factor range {factor_range} and {steps} steps...")
        
        # Get original device
        original_device = latent.device
        
        # For StyleGAN2's custom CUDA ops, we must use a GPU
        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for applying edits with InterFaceGAN")
        
        # Clear any leftover GPU memory before the operation
        self._clear_gpu_memory()
        
        try:
            # Move to the preferred device
            device = torch.device('cuda')
            
            # Load direction tensor
            direction_path = directions[direction_name]
            direction_tensor = torch.load(direction_path, map_location=device)
            
            # Move latent to GPU if it's not already there
            if latent.device.type != 'cuda':
                latent = latent.to(device)
            
            # Generate factors
            min_factor, max_factor = factor_range
            factors = torch.linspace(min_factor, max_factor, steps)
            
            # Process each factor
            results = []
            for factor in factors:
                # Clear memory between iterations
                self._clear_gpu_memory()
                
                # Apply the edit
                try:
                    edited_image = self.apply_interfacegan(latent, direction_name, factor.item())
                    results.append(edited_image)
                except Exception as e:
                    logger.error(f"Error applying edit with factor {factor.item()}: {str(e)}")
                    # Continue with next factor
            
            # Return the results
            return results
            
        except Exception as e:
            logger.error(f"Error applying edit range: {str(e)}")
            # Clean up memory
            self._clear_gpu_memory()
            raise

    def _ensure_model_on_device(self, model, device):
        """
        Ensure all parameters of a model are on the same device.
        
        Args:
            model (torch.nn.Module): Model to check
            device (torch.device): Target device
            
        Returns:
            torch.nn.Module: Model with all parameters on the target device
        """
        logger = logging.getLogger(__name__)
        
        # Move the model to the target device
        model = model.to(device)
        
        # Check if all parameters are on the target device
        for name, param in model.named_parameters():
            if param.device != device:
                logger.warning(f"Parameter {name} is on {param.device}, moving to {device}")
                param.data = param.data.to(device)
        
        # Check if all buffers are on the target device
        for name, buffer in model.named_buffers():
            if buffer.device != device:
                logger.warning(f"Buffer {name} is on {buffer.device}, moving to {device}")
                buffer.data = buffer.data.to(device)
        
        return model


class E4EProcessor:
    """Main class that integrates all the functionality."""
    
    def __init__(self, experiment_type='ffhq_encode', base_dir=None, memory_efficient=False, enable_mixed_precision=True, max_batch_size=1):
        """
        Initialize the E4E processor.
        
        Args:
            experiment_type (str): Type of experiment ('ffhq_encode', 'cars_encode', etc.)
            base_dir (str, optional): Base directory for the e4e project
            memory_efficient (bool): Whether to use memory-efficient mode to reduce GPU memory usage
            enable_mixed_precision (bool): Whether to enable mixed precision to reduce GPU memory usage
            max_batch_size (int): Maximum batch size to use during inference
        """
        self.experiment_type = experiment_type
        self.memory_efficient = memory_efficient
        # Disable mixed precision by default to avoid dtype mismatches
        self.enable_mixed_precision = False
        self.max_batch_size = max_batch_size
        
        # Set up paths and file configuration
        self.setup = E4ESetup(experiment_type, base_dir)
        
        # Check if model exists
        if not self.setup.check_model_exists():
            raise FileNotFoundError(
                f"Model file not found at: {self.setup.model_path}\n"
                f"{self.setup.get_download_instructions()}"
            )
        
        # Check if face predictor exists for face experiments
        if experiment_type == 'ffhq_encode' and not self.setup.check_face_predictor_exists():
            print(f"Warning: Face predictor not found at: {self.setup.face_predictor_path}")
            print("Face alignment will be skipped or may fail.")
            print("See download instructions for more information.")
        
        # Log if using memory efficient mode
        if self.memory_efficient:
            logging.info("Initializing with memory-efficient mode enabled")
        
        # Log precision mode - always using float32 now
        logging.info("Using Float32 precision for all operations to ensure tensor type compatibility")
        
        # Set up the preprocessor
        self.preprocessor = E4EPreprocessor(experiment_type, self.setup)
        
        # Set up the inference module with memory optimization options
        self.inference = E4EInference(self.setup.model_path, experiment_type, 
                                      memory_efficient=memory_efficient,
                                      max_batch_size=max_batch_size)
        
        # Ensure mixed precision setting is disabled consistently
        self.inference.use_mixed_precision = False
        
        # Log that mixed precision is disabled
        logging.info("Mixed precision is disabled to avoid tensor type mismatches - model will use Float32 throughout")
        
        # Set up the latent combiner
        self.combiner = E4ELatentCombiner(self.inference.net.decoder, experiment_type)
        
        # Set up the editor
        self.editor = E4EEditor(self.inference.net.decoder, experiment_type, self.setup)
        
        # Set memory-efficient mode for editor
        self.editor.set_memory_efficient(self.memory_efficient)
    
    def _clear_gpu_memory(self):
        """Clear GPU memory cache to free up memory."""
        if torch.cuda.is_available():
            # Force garbage collection first
            import gc
            gc.collect()
            # Then empty CUDA cache
            torch.cuda.empty_cache()
            
            # Log memory usage for debugging
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
                logging.debug(f"GPU Memory: Allocated {allocated:.2f}GB (Max: {max_allocated:.2f}GB), "
                             f"Reserved {reserved:.2f}GB (Max: {max_reserved:.2f}GB)")
    
    def cleanup(self):
        """
        Clean up resources and free memory.
        Call this when you're done using the processor.
        """
        logger = logging.getLogger(__name__)
        logger.info("Cleaning up resources and freeing memory...")
        
        try:
            # Move model components to CPU
            if hasattr(self, 'inference') and hasattr(self.inference, 'net'):
                try:
                    self.inference.net.to('cpu')
                    logger.info("Moved model to CPU")
                except Exception as e:
                    logger.warning(f"Could not move model to CPU: {str(e)}")
            
            # Release combiner's decoder reference if it exists
            if hasattr(self, 'combiner'):
                try:
                    self.combiner.decoder = None
                    logger.info("Released combiner's decoder reference")
                except Exception as e:
                    logger.warning(f"Could not release combiner's decoder: {str(e)}")
            
            # Release editor's decoder reference if it exists
            if hasattr(self, 'editor'):
                try:
                    self.editor.decoder = None
                    logger.info("Released editor's decoder reference")
                except Exception as e:
                    logger.warning(f"Could not release editor's decoder: {str(e)}")
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache")
            
            # Run garbage collection
            import gc
            gc.collect()
            logger.info("Ran garbage collection")
            
            return True
        
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            # Make one final attempt to clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False
    
    def process_image(self, image_path, save_result=False, output_dir=None, filename_prefix='result', return_saved_path=False):
        """
        Process an image through the E4E pipeline.
        
        Args:
            image_path (str): Path to the image to process
            save_result (bool): Whether to save the result image
            output_dir (str): Directory to save the result in (if save_result is True)
            filename_prefix (str): Prefix for the filename (if save_result is True)
            return_saved_path (bool): Whether to return the saved path as a fourth return value
            
        Returns:
            tuple: If return_saved_path is True: (result_image, latent, processed_image, saved_path)
                  Otherwise: (result_image, latent, processed_image)
        """
        # Clear GPU memory before processing
        self._clear_gpu_memory()
        
        # Preprocess the image
        transformed_image, processed_image = self.preprocessor.preprocess_image(image_path)
        
        # Run inference
        result_image, latent = self.inference.infer(transformed_image)
        
        # Save result if requested
        saved_path = None
        if save_result:
            # Generate a timestamp for the filename
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Set default output directory if not provided
            if output_dir is None:
                output_dir = os.path.join(os.path.dirname(image_path), 'outputs')
            
            # Create the output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate the filename
            filename = f"{filename_prefix}_{timestamp}.jpg"
            saved_path = os.path.join(output_dir, filename)
            
            # Save the image
            result_image.save(saved_path)
            logging.info(f"Saved result image to: {saved_path}")
        
        # Clear GPU memory after processing
        self._clear_gpu_memory()
        
        # Return appropriate tuple based on return_saved_path parameter
        if return_saved_path:
            return result_image, latent, processed_image, saved_path
        else:
            return result_image, latent, processed_image
    
    def combine_images(self, image_path_1, image_path_2, weights=None, save_individual=True, output_dir=None):
        """
        Combine two images by blending their latent codes.
        
        Args:
            image_path_1 (str): Path to the first image
            image_path_2 (str): Path to the second image
            weights (list, optional): Weights for the combination. Can be:
                - None: Equal weights (50/50 blend)
                - List of scalars [w1, w2]: Simple weighting for each entire latent
                - List of tensors: Each tensor has the same shape as the latent for per-dimension weighting
            save_individual (bool): Whether to save individual processed images
            output_dir (str): Directory to save the results in
            
        Returns:
            tuple: Tuple of (combined image, combined latent, display image with all three)
        """
        # Clear GPU memory before processing
        self._clear_gpu_memory()
        
        # Set up output directory
        if output_dir is None:
            # Use the directory of the first image by default
            output_dir = os.path.join(os.path.dirname(image_path_1), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a timestamp for unique filenames
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Process the first image and save if requested
        logging.info(f"Processing first image: {image_path_1}")
        if save_individual:
            result_image_1, latent_1, processed_image_1, saved_path_1 = self.process_image(
                image_path_1,
                save_result=True,
                output_dir=output_dir,
                filename_prefix=f"image1_{os.path.splitext(os.path.basename(image_path_1))[0]}",
                return_saved_path=True
            )
            logging.info(f"Saved first processed image to: {saved_path_1}")
        else:
            result_image_1, latent_1, processed_image_1 = self.process_image(image_path_1)
        
        # Process the second image and save if requested
        logging.info(f"Processing second image: {image_path_2}")
        if save_individual:
            result_image_2, latent_2, processed_image_2, saved_path_2 = self.process_image(
                image_path_2,
                save_result=True,
                output_dir=output_dir,
                filename_prefix=f"image2_{os.path.splitext(os.path.basename(image_path_2))[0]}",
                return_saved_path=True
            )
            logging.info(f"Saved second processed image to: {saved_path_2}")
        else:
            result_image_2, latent_2, processed_image_2 = self.process_image(image_path_2)
        
        # Clear GPU memory after processing images
        self._clear_gpu_memory()
        
        # Ensure latents are on the same device
        if latent_1.device != latent_2.device:
            logging.info(f"Moving latents to the same device: {latent_1.device}")
            latent_2 = latent_2.to(latent_1.device)
        
        # Combine the latents with appropriate weights
        if weights is None:
            # Default 50/50 blend
            weights = [0.5, 0.5]
        
        try:
            # Log weight information
            if all(isinstance(w, (float, int)) for w in weights):
                logging.info(f"Combining latents with scalar weights: {weights}")
            else:
                logging.info(f"Combining latents with per-dimension weights")
            
            # Combine latents
            combined_latent = self.combiner.combine_latents([latent_1, latent_2], weights)
            
            # Generate image from combined latent
            result_image = self.combiner.generate_from_latent(combined_latent)
            
            # Clear GPU memory after generating result
            self._clear_gpu_memory()
            
            # Display function
            def display_alongside_source_images(result, source_1, source_2):
                resize_dims = self.preprocessor.resize_dims
                res = np.concatenate([
                    np.array(source_1.resize(resize_dims)),
                    np.array(source_2.resize(resize_dims)),
                    np.array(result.resize(resize_dims))
                ], axis=1)
                return Image.fromarray(res)
            
            # Create the display image
            display_image = display_alongside_source_images(
                result_image, processed_image_1, processed_image_2
            )
            
            return result_image, combined_latent, display_image
        
        except Exception as e:
            logging.error(f"Error combining images: {str(e)}")
            self._clear_gpu_memory()  # Make sure to clean up
            raise
    
    def edit_image(self, latent, edit_type, **kwargs):
        """
        Apply an edit to a latent code.
        
        Args:
            latent (torch.Tensor): Latent code
            edit_type (str): Type of edit ('interfacegan', 'ganspace', 'sefa')
            **kwargs: Additional arguments for the specific edit type
                return_single_image (bool): If True, return only a single image instead of concatenated images
                multi_directions (list): List of dictionaries containing direction_name and factor pairs
                    e.g., [{'direction_name': 'age', 'factor': -3.0}, {'direction_name': 'smile', 'factor': 2.0}]
            
        Returns:
            PIL.Image: Edited image
        """
        # Clear GPU memory before editing
        self._clear_gpu_memory()
        
        # Get original device
        original_device = latent.device
        
        # Use a GPU memory-efficient approach
        use_memory_efficient = kwargs.get('use_memory_efficient', self.memory_efficient)
        
        # Check if we should return a single image
        return_single_image = kwargs.get('return_single_image', False)
        
        # Set memory-efficient mode for the editor
        self.editor.set_memory_efficient(use_memory_efficient)
        
        try:
            # Apply the edit based on the edit type
            if edit_type == 'interfacegan':
                # Check if we're doing multiple direction edits
                multi_directions = kwargs.get('multi_directions', None)
                
                if multi_directions is not None and isinstance(multi_directions, list) and len(multi_directions) > 0:
                    # Apply multiple edits sequentially
                    logging.info(f"Applying multiple InterfaceGAN edits: {[d['direction_name'] for d in multi_directions]}")
                    
                    # Start with the original latent
                    edited_latent = latent.clone()
                    
                    # Apply each direction sequentially
                    for direction_info in multi_directions:
                        direction_name = direction_info.get('direction_name')
                        factor = direction_info.get('factor', 0.0)
                        
                        if not direction_name:
                            continue
                            
                        logging.info(f"Applying {direction_name} edit with factor {factor}")
                        
                        # Get the direction tensor
                        directions = self.editor.get_interfacegan_directions()
                        if direction_name not in directions:
                            logging.warning(f"Direction {direction_name} not found, skipping...")
                            continue
                            
                        direction_path = directions[direction_name]
                        direction_tensor = torch.load(direction_path, map_location=edited_latent.device)
                        
                        # Apply the edit to the latent
                        edited_latent = edited_latent + factor * direction_tensor
                    
                    # Generate image from the final edited latent
                    edited_image = self.combiner.generate_from_latent(edited_latent)
                    
                else:
                    # Original single-direction code
                    direction_name = kwargs.get('direction_name', 'age')
                    factor = kwargs.get('factor', -3.0)
                    factor_range = kwargs.get('factor_range', None)
                    
                    logging.info(f"Applying interfacegan edit: {direction_name}, factor: {factor}")
                    
                    if factor_range is not None:
                        edited_images = self.editor.apply_interfacegan_range(latent, direction_name, factor_range)
                        # If return_single_image is True, only return the first image
                        if return_single_image and isinstance(edited_images, list) and len(edited_images) > 0:
                            edited_image = edited_images[0]
                        else:
                            edited_image = edited_images
                    else:
                        # Apply the edit to get a concatenated image
                        edited_image_concat = self.editor.apply_interfacegan(latent, direction_name, factor)
                        
                        # If we need a single image, generate it directly instead of using the concatenated version
                        if return_single_image:
                            # Apply the edit to the latent
                            directions = self.editor.get_interfacegan_directions()
                            direction_path = directions[direction_name]
                            direction_tensor = torch.load(direction_path, map_location=latent.device)
                            edited_latent = latent + factor * direction_tensor
                            
                            # Generate image directly from latent
                            edited_image = self.combiner.generate_from_latent(edited_latent)
                        else:
                            edited_image = edited_image_concat
                
                # Clear GPU memory after editing
                self._clear_gpu_memory()
                
                return edited_image
            
            elif edit_type == 'ganspace':
                direction_names = kwargs.get('direction_names', ['eye_openness', 'smile'])
                logging.info(f"Applying ganspace edit with directions: {direction_names}")
                return self.editor.apply_ganspace(latent, direction_names)
            
            elif edit_type == 'sefa':
                indices = kwargs.get('indices', [2, 3, 4, 5])
                start_distance = kwargs.get('start_distance', 0.0)
                end_distance = kwargs.get('end_distance', 15.0)
                step = kwargs.get('step', 3)
                
                logging.info(f"Applying sefa edit with indices: {indices}")
                return self.editor.apply_sefa(latent, indices, start_distance, end_distance, step)
            
            else:
                raise ValueError(f"Edit type {edit_type} not supported")
        
        except Exception as e:
            logging.error(f"Error applying edit: {str(e)}")
            self._clear_gpu_memory()  # Make sure to clean up
            raise
        finally:
            # Clear GPU memory after editing
            self._clear_gpu_memory()

    def process_directory(self, input_dir, output_dir=None, file_extensions=('.jpg', '.jpeg', '.png')):
        """
        Process all images in a directory, saving each result with a unique filename.
        
        Args:
            input_dir (str): Directory containing images to process
            output_dir (str): Directory to save results in (if None, uses input_dir/outputs)
            file_extensions (tuple): File extensions to consider as images
            
        Returns:
            list: List of (input_path, output_path) tuples for all processed images
        """
        if output_dir is None:
            output_dir = os.path.join(input_dir, 'outputs')
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all images in the input directory
        image_paths = []
        for ext in file_extensions:
            image_paths.extend(glob.glob(os.path.join(input_dir, f'*{ext}')))
            image_paths.extend(glob.glob(os.path.join(input_dir, f'*{ext.upper()}')))
        
        logging.info(f"Found {len(image_paths)} images to process")
        
        # Process each image
        results = []
        for i, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
            try:
                filename = os.path.basename(image_path)
                base_name = os.path.splitext(filename)[0]
                
                # Process the image and save with a unique filename
                result_image, latent, processed_image, saved_path = self.process_image(
                    image_path,
                    save_result=True,
                    output_dir=output_dir,
                    filename_prefix=base_name,
                    return_saved_path=True  # Explicitly request the 4th return value
                )
                
                results.append((image_path, saved_path))
                logging.info(f"Processed {i+1}/{len(image_paths)}: {image_path} -> {saved_path}")
                
                # Add a small delay to ensure unique timestamps if processing is fast
                time.sleep(0.1)
                
            except Exception as e:
                logging.error(f"Error processing {image_path}: {str(e)}")
        
        return results

    def create_dimension_weights(self, latent_1, latent_2, dimension_weights=None, layer_weights=None):
        """
        Create per-dimension weights for fine-grained control over latent space blending.
        
        This method allows precise control over which dimensions from each latent code 
        contribute to the final result. Typical StyleGAN latent codes have shape [18, 512],
        where 18 represents style layers and 512 represents per-layer dimensions.
        
        Args:
            latent_1 (torch.Tensor): First latent code
            latent_2 (torch.Tensor): Second latent code
            dimension_weights (torch.Tensor or dict, optional): 
                - If tensor: Shape [512] containing weights (0-1) for each dimension across all layers
                   where 1.0 means take 100% from latent_1, 0.0 means take 100% from latent_2
                - If dict: Dictionary mapping dimension indices to weight values
            layer_weights (torch.Tensor or dict, optional): 
                - If tensor: Shape [18] containing weights (0-1) for each layer across all dimensions
                   where 1.0 means take 100% from latent_1, 0.0 means take 100% from latent_2
                - If dict: Dictionary mapping layer indices to weight values
            
        Returns:
            list: List of two weight tensors, one for each latent
        """
        # Ensure latents are on the same device
        if latent_1.device != latent_2.device:
            latent_2 = latent_2.to(latent_1.device)
        
        # Get latent shape
        latent_shape = latent_1.shape
        device = latent_1.device
        
        # Create default weights - 50% each
        weight_1 = torch.ones(latent_shape, device=device) * 0.5
        weight_2 = torch.ones(latent_shape, device=device) * 0.5
        
        # Apply dimension-specific weights if provided
        if dimension_weights is not None:
            if isinstance(dimension_weights, dict):
                # Original dictionary-based implementation
                if len(latent_shape) == 2:  # Shape is [layers, dimensions]
                    for dim, weight in dimension_weights.items():
                        # Apply weight to all layers for the specified dimension
                        if 0 <= dim < latent_shape[1]:
                            weight_1[:, dim] = weight
                            weight_2[:, dim] = 1.0 - weight
                        else:
                            logging.warning(f"Dimension {dim} is out of range (0-{latent_shape[1]-1})")
                else:
                    logging.warning(f"Dimension weights not applied: unexpected latent shape {latent_shape}")
            elif isinstance(dimension_weights, torch.Tensor):
                # New tensor-based implementation
                if len(latent_shape) == 2 and dimension_weights.shape[0] == latent_shape[1]:
                    # Expand dimension weights to match latent shape [1, dimensions] -> [layers, dimensions]
                    dim_weights_expanded = dimension_weights.unsqueeze(0).expand(latent_shape[0], -1).to(device)
                    # Apply weights
                    weight_1 = dim_weights_expanded
                    weight_2 = 1.0 - dim_weights_expanded
                else:
                    logging.warning(f"Dimension weights tensor shape {dimension_weights.shape} doesn't match latent dimensions {latent_shape[1]}")
            else:
                logging.warning(f"Dimension weights type {type(dimension_weights)} not supported")
        
        # Apply layer-specific weights if provided
        if layer_weights is not None:
            if isinstance(layer_weights, dict):
                # Original dictionary-based implementation
                if len(latent_shape) == 2:  # Shape is [layers, dimensions]
                    for layer, weight in layer_weights.items():
                        # Apply weight to all dimensions in the specified layer
                        if 0 <= layer < latent_shape[0]:
                            weight_1[layer, :] = weight
                            weight_2[layer, :] = 1.0 - weight
                        else:
                            logging.warning(f"Layer {layer} is out of range (0-{latent_shape[0]-1})")
                else:
                    logging.warning(f"Layer weights not applied: unexpected latent shape {latent_shape}")
            elif isinstance(layer_weights, torch.Tensor):
                # New tensor-based implementation
                if len(latent_shape) == 2 and layer_weights.shape[0] == latent_shape[0]:
                    # Expand layer weights to match latent shape [layers, 1] -> [layers, dimensions]
                    layer_weights_expanded = layer_weights.unsqueeze(1).expand(-1, latent_shape[1]).to(device)
                    # Apply weights
                    weight_1 = layer_weights_expanded
                    weight_2 = 1.0 - layer_weights_expanded
                else:
                    logging.warning(f"Layer weights tensor shape {layer_weights.shape} doesn't match latent layers {latent_shape[0]}")
            else:
                logging.warning(f"Layer weights type {type(layer_weights)} not supported")
        
        # Return weights for both latents
        return [weight_1, weight_2]
    
    def combine_latent_dimensions(self, latent_1, latent_2, blend_mask=None):
        """
        Combine two latent codes using a binary blend mask.
        
        Args:
            latent_1 (torch.Tensor): First latent code
            latent_2 (torch.Tensor): Second latent code
            blend_mask (torch.Tensor, optional): Binary mask with the same shape as the latent codes.
                Values of 1 take from latent_1, values of 0 take from latent_2.
                If None, a random mask will be generated.
            
        Returns:
            torch.Tensor: Combined latent code
        """
        # Ensure latents are on the same device
        if latent_1.device != latent_2.device:
            latent_2 = latent_2.to(latent_1.device)
        
        # Get latent shape and device
        latent_shape = latent_1.shape
        device = latent_1.device
        
        # Create blend mask if not provided
        if blend_mask is None:
            # Random binary mask
            blend_mask = torch.randint(0, 2, latent_shape, device=device).float()
        else:
            # Ensure mask is on the correct device and has proper shape
            blend_mask = blend_mask.to(device)
            if blend_mask.shape != latent_shape:
                raise ValueError(f"Blend mask shape {blend_mask.shape} doesn't match latent shape {latent_shape}")
        
        # Combine latents using the mask
        combined_latent = latent_1 * blend_mask + latent_2 * (1 - blend_mask)
        
        return combined_latent


# Example usage
if __name__ == "__main__":
    # Set up logging for the main script
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create images directory if it doesn't exist
    images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
    os.makedirs(images_dir, exist_ok=True)
    logger.info(f"Using images directory: {images_dir}")
    
    # Define input and output paths
    input_image = os.path.join(images_dir, 'input.jpg')
    second_image = os.path.join(images_dir, 'second.jpg')
    
    # Check if input images exist
    if not os.path.exists(input_image):
        logger.error(f"Please place an input image at: {input_image}")
        sys.exit(1)
    
    if not os.path.exists(second_image):
        logger.warning(f"Second image not found at: {second_image}")
        logger.warning("Skipping image combination example")
    
    # Initialize the processor with progress tracking
    logger.info("Initializing E4E processor...")
    with tqdm(total=1, desc="Processor initialization") as pbar:
        processor = E4EProcessor(
            experiment_type='ffhq_encode',
            memory_efficient=True  # Enable memory-efficient mode to reduce GPU memory usage
        )
        pbar.update(1)
    
    # Generate a timestamp for unique filenames
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process an image with auto-saving
    logger.info("Processing input image with auto-save...")
    output_dir = os.path.join(images_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Use return_saved_path=True to get the saved path
    result_image, latent, processed_image, saved_path = processor.process_image(
        input_image,
        save_result=True,
        output_dir=output_dir,
        filename_prefix='result',
        return_saved_path=True
    )
    logger.info(f"Processed image saved to: {saved_path}")
    
    # Example of processing multiple images (if the directory has more images)
    logger.info("Checking for multiple image processing example...")
    if len(glob.glob(os.path.join(images_dir, '*.jpg'))) > 1:
        logger.info("Found multiple images. Demonstrating batch processing...")
        results = processor.process_directory(
            input_dir=images_dir,
            output_dir=output_dir
        )
        logger.info(f"Processed {len(results)} images in batch mode")
        for input_path, output_path in results:
            logger.info(f"  {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
    
    # Try to combine images if second image exists
    if os.path.exists(second_image):
        logger.info("Combining images...")
        
        # Basic combination with global weights
        logger.info("1. Basic combination with global weights (30% first image, 70% second image)...")
        combined_image, combined_latent, display_image = processor.combine_images(
            image_path_1=input_image,
            image_path_2=second_image,
            weights=[0.3, 0.7],  # 30% from first image, 70% from second image
            save_individual=True,
            output_dir=output_dir
        )
        
        # Save combined results with timestamp-based filenames
        combined_path = os.path.join(output_dir, f'combined_global_{timestamp}.jpg')
        combined_image.save(combined_path)
        logger.info(f"Saved globally combined image to: {combined_path}")
        
        # Process both images to get their latent codes
        _, latent_1, _ = processor.process_image(input_image)
        _, latent_2, _ = processor.process_image(second_image)
        
        # Example of per-dimension weighting
        logger.info("2. Combining images with per-dimension weights...")
        try:
            # Create weights for specific dimensions (like taking eyes from first image, mouth from second)
            dimension_weights = {
                50: 0.9,   # Dimension 50 takes 90% from first image
                100: 0.8,  # Dimension 100 takes 80% from first image
                150: 0.7   # Dimension 150 takes 70% from first image
            }
            
            # Create weights for specific layers (like taking coarse features from one, fine details from another)
            layer_weights = {
                0: 0.2,   # Layer 0 (coarse) takes 20% from first image
                5: 0.5,   # Layer 5 (medium) takes 50% from first image
                10: 0.8   # Layer 10 (fine) takes 80% from first image
            }
            
            # Generate dimension-specific weights with dictionary approach
            per_dim_weights = processor.create_dimension_weights(
                latent_1, latent_2, 
                dimension_weights=dimension_weights,
                layer_weights=layer_weights
            )
            
            # Combine latents with per-dimension weights
            combined_latent_dim = processor.combiner.combine_latents([latent_1, latent_2], per_dim_weights)
            
            # Generate and save image from combined latent
            result_image_dim = processor.combiner.generate_from_latent(combined_latent_dim)
            dim_combined_path = os.path.join(output_dir, f'combined_per_dim_{timestamp}.jpg')
            result_image_dim.save(dim_combined_path)
            logger.info(f"Saved per-dimension combined image to: {dim_combined_path}")
            
            # Example using tensor-based weights
            logger.info("4. Combining images with tensor-based weights...")
            
            # Get latent dimensions
            latent_shape = latent_1.shape  # Typically [18, 512]
            
            # Create continuous tensor weights for dimensions where:
            # - Dimensions 0-200: Linear increase from 0.2 to 0.8 (more from second image to more from first)
            # - Dimensions 200-400: All 0.5 (equal blending)
            # - Dimensions 400-512: Linear decrease from 0.8 to 0.2 (more from first to more from second)
            dim_tensor = torch.zeros(latent_shape[1], device=latent_1.device)
            
            # Linear gradient for first segment
            dim_tensor[:200] = torch.linspace(0.2, 0.8, 200, device=latent_1.device)
            # Middle segment with equal weighting
            dim_tensor[200:400] = 0.5
            # Linear gradient for last segment
            dim_tensor[400:] = torch.linspace(0.8, 0.2, latent_shape[1]-400, device=latent_1.device)
            
            # Create layer weights tensor for smooth transition from coarse to fine details
            # Lower layers (coarse features) take more from second image
            # Higher layers (fine details) take more from first image
            layer_tensor = torch.linspace(0.2, 0.8, latent_shape[0], device=latent_1.device)
            
            # Generate weights using tensor approach
            tensor_weights = processor.create_dimension_weights(
                latent_1, latent_2,
                dimension_weights=dim_tensor,
                layer_weights=layer_tensor
            )
            
            # Combine latents with tensor-based weights
            combined_latent_tensor = processor.combiner.combine_latents([latent_1, latent_2], tensor_weights)
            
            # Generate and save image from tensor-based combined latent
            result_image_tensor = processor.combiner.generate_from_latent(combined_latent_tensor)
            tensor_combined_path = os.path.join(output_dir, f'combined_tensor_{timestamp}.jpg')
            result_image_tensor.save(tensor_combined_path)
            logger.info(f"Saved tensor-based combined image to: {tensor_combined_path}")
            
            # Example of binary mask blending
            logger.info("5. Combining images with binary mask blending...")
            
            # Create a random binary mask for demonstration
            blend_mask = torch.randint(0, 2, latent_1.shape, device=latent_1.device).float()
            
            # Combine latents using the mask
            mask_combined_latent = processor.combine_latent_dimensions(latent_1, latent_2, blend_mask)
            
            # Generate and save image from mask-combined latent
            mask_result_image = processor.combiner.generate_from_latent(mask_combined_latent)
            mask_combined_path = os.path.join(output_dir, f'combined_mask_{timestamp}.jpg')
            mask_result_image.save(mask_combined_path)
            logger.info(f"Saved mask-combined image to: {mask_combined_path}")
            
        except Exception as e:
            logger.error(f"Error during advanced combination examples: {str(e)}")
        
        # Apply an edit to the combined image
        logger.info("Applying age edit to combined image...")
        edited_image = processor.edit_image(
            combined_latent,
            edit_type='interfacegan',
            direction_name='age',
            factor=-3,
            use_memory_efficient=True,  # Use memory-efficient mode for editing
            return_single_image=True,    # Get just one image instead of 18 concatenated images
            multi_directions=[{'direction_name': 'age', 'factor': -3.0}]
        )
        
        edited_path = os.path.join(output_dir, f'edited_{timestamp}.jpg')
        edited_image.save(edited_path)
        logger.info(f"Saved edited image to: {edited_path}")
    
    logger.info("Processing complete!")
    logger.info(f"All results have been saved to: {output_dir}")
