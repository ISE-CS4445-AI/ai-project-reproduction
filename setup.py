import os
import subprocess
import logging
import sys
import requests
from tqdm import tqdm
import gdown  # For Google Drive folder download

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_file(url, destination, description=None):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    desc = description if description else os.path.basename(destination)
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=desc)
    
    with open(destination, 'wb') as file:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)
    progress_bar.close()

def download_gdrive_file(file_id, destination, description=None):
    """Download a single file from Google Drive."""
    logger.info(f"Downloading file from Google Drive to {destination}...")
    
    # Create destination directory
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Install gdown if not already installed
    try:
        import gdown
    except ImportError:
        logger.info("Installing gdown for Google Drive download...")
        run_command('pip install gdown', "Installing gdown")
        import gdown
    
    # Download the file
    url = f"https://drive.google.com/uc?id={file_id}"
    desc = description if description else os.path.basename(destination)
    gdown.download(url, destination, quiet=False)
    
    logger.info(f"Successfully downloaded {desc} to {destination}")

def download_gdrive_folder(folder_id, destination_dir):
    """Download an entire folder from Google Drive."""
    logger.info(f"Downloading Google Drive folder to {destination_dir}...")
    
    # Install gdown if not already installed
    try:
        import gdown
    except ImportError:
        logger.info("Installing gdown for Google Drive folder download...")
        run_command('pip install gdown', "Installing gdown")
        import gdown
    
    # Create destination directory
    os.makedirs(destination_dir, exist_ok=True)
    
    # Download the folder
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    gdown.download_folder(url=url, output=destination_dir, quiet=False)
    
    logger.info(f"Successfully downloaded folder to {destination_dir}")

def run_command(command, description=None):
    """Run a shell command with logging."""
    if description:
        logger.info(description)
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)

def setup_environment():
    """Set up the Python environment with required packages."""
    requirements = [
        'torch>=1.7.1',
        'torchvision>=0.8.2',
        'numpy>=1.19.2',
        'pandas>=1.2.0',
        'Pillow>=8.1.0',
        'matplotlib>=3.3.3',
        'tqdm>=4.56.0',
        'requests>=2.25.1',
        'scikit-learn>=0.24.1',
        'ninja',  # Required for C++ extensions
        'gdown',  # For Google Drive folder download
    ]
    
    with open('requirements.txt', 'w') as f:
        f.write('\n'.join(requirements))
    
    logger.info("Installing required packages...")
    run_command('pip install -r requirements.txt', "Installing Python packages")

def download_models():
    """Download required model files and pre-computed data."""
    # Required model files
    models = {
        'e4e_ffhq_encode': {
            'url': 'https://drive.google.com/uc?id=1cUv_reLE6k3604or78EranS7XzuVMWeO',
            'path': 'pretrained_models/e4e_ffhq_encode.pt'
        },
        'shape_predictor': {
            'url': 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2',
            'path': 'pretrained_models/shape_predictor_68_face_landmarks.dat.bz2'
        }
    }
    
    # Pre-computed data from Google Drive
    precomputed_data = {
        'latents_folder': {
            'id': '1heLmPFZUA52YEQvSCavNZM5OHG4e0SCF',
            'path': 'latents',
            'is_folder': True
        },
        'child_embeddings': {
            'id': '1b6LFeOdt58DbWY72BAR_smMU-zzQ8mb9',
            'path': 'embeddings/child_embeddings.pt',
            'is_folder': False
        }
    }
    
    # Download and extract model files
    os.makedirs('pretrained_models', exist_ok=True)
    for model_name, model_info in models.items():
        if not os.path.exists(model_info['path']):
            logger.info(f"Downloading {model_name}...")
            download_file(model_info['url'], model_info['path'], f"Downloading {model_name}")
            
            if model_info['path'].endswith('.bz2'):
                logger.info(f"Extracting {model_name}...")
                run_command(f"bzip2 -dk {model_info['path']}", f"Extracting {model_name}")
    
    # Download pre-computed data
    for data_name, data_info in precomputed_data.items():
        if data_info['is_folder']:
            logger.info(f"Downloading pre-computed {data_name}...")
            if not os.listdir(data_info['path']) if os.path.exists(data_info['path']) else True:
                download_gdrive_folder(data_info['id'], data_info['path'])
        else:
            if not os.path.exists(data_info['path']):
                logger.info(f"Downloading pre-computed {data_name}...")
                download_gdrive_file(data_info['id'], data_info['path'], f"Downloading {data_name}")

def setup_encoder4editing():
    """Clone and set up the encoder4editing repository."""
    if not os.path.exists('encoder4editing'):
        logger.info("Cloning encoder4editing repository...")
        run_command(
            'git clone https://github.com/omertov/encoder4editing.git',
            "Cloning encoder4editing"
        )

def create_directories():
    """Create necessary directories for the project."""
    directories = [
        'outputs',
        'family_models',
        'embeddings',
        'latents',
        'sample_images/fathers',
        'sample_images/mothers',
        'sample_images/children'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def main():
    logger.info("Starting setup...")
    
    # Create project structure
    create_directories()
    
    # Set up Python environment
    setup_environment()
    
    # Clone required repositories
    setup_encoder4editing()
    
    # Download model files and pre-computed data
    download_models()
    
    logger.info("""
Setup complete! The system is ready to go:
1. Pre-computed latents and embeddings have been downloaded
2. All required models have been installed
3. Directory structure has been created

To start training:
1. Place your family images in the appropriate directories (if you plan to use your own images):
   - sample_images/fathers/
   - sample_images/mothers/
   - sample_images/children/
2. Run: python train.py
""")

if __name__ == "__main__":
    main() 