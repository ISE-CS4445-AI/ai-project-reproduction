import os
import subprocess
import logging
import sys
import requests
from tqdm import tqdm
import gdown  # For Google Drive folder download
import zipfile  # For extracting zip files
import shutil  # For directory operations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_file(url, destination, description=None):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(destination) or '.', exist_ok=True)
    
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
    
    # Create destination directory if needed (handling the case where dirname is empty)
    dir_name = os.path.dirname(destination)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
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

def extract_zip(zip_path, extract_to):
    """Extract a zip file to the specified directory."""
    logger.info(f"Extracting {zip_path} to {extract_to}...")
    
    # Create extraction directory if it doesn't exist
    os.makedirs(extract_to, exist_ok=True)
    
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    logger.info(f"Successfully extracted {zip_path}")

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
        'child_embeddings': {
            'id': '1b6LFeOdt58DbWY72BAR_smMU-zzQ8mb9',
            'path': 'embeddings/child_embeddings.pt',
            'is_zip': False
        },
        'latents': {
            'id': '1x5PnIu-pqeImqTE1rhY3iSMO3LCwBzid',
            'path': './latents.zip',  # Added ./ to ensure there's a directory component
            'is_zip': True,
            'extract_to': 'latents'
        },
        'checkpoint_csv': {
            'id': '1-MLAiuDcROAkd7yzrjKQpsrChTU8_Wh9',
            'path': 'CSVs/checkpoint10.csv',
            'is_zip': False
        },
        'aligned_test': {
            'id': '1VoKZyFXG8HpTbfgMtJ24qsE9J81tZvte',
            'path': './AlignedTest2.zip',
            'is_zip': True,
            'extract_to': '.'
        }
    }
    
    # Create model directories and ensure they exist
    os.makedirs('pretrained_models', exist_ok=True)
    os.makedirs('e4e', exist_ok=True)
    
    # Download and extract model files
    for model_name, model_info in models.items():
        if not os.path.exists(model_info['path']):
            logger.info(f"Downloading {model_name}...")
            download_file(model_info['url'], model_info['path'], f"Downloading {model_name}")
            
            if model_info['path'].endswith('.bz2'):
                logger.info(f"Extracting {model_name}...")
                run_command(f"bzip2 -dk {model_info['path']}", f"Extracting {model_name}")
                
                # Also copy the extracted file to the root directory for easier access
                unbz2_path = model_info['path'][:-4]  # Remove .bz2 extension
                shutil.copy2(unbz2_path, os.path.basename(unbz2_path))
                logger.info(f"Copied {unbz2_path} to {os.path.basename(unbz2_path)}")
    
    # Also copy the e4e model to various locations to ensure it's found
    if os.path.exists('pretrained_models/e4e_ffhq_encode.pt'):
        logger.info("Copying e4e model to alternative locations...")
        # Copy to e4e directory
        os.makedirs('e4e/encoder4editing/pretrained_models', exist_ok=True)
        shutil.copy2('pretrained_models/e4e_ffhq_encode.pt', 'e4e/encoder4editing/pretrained_models/e4e_ffhq_encode.pt')
        logger.info("Copied e4e model to e4e/encoder4editing/pretrained_models/")
    
    # Create CSVs directory if it doesn't exist
    os.makedirs('CSVs', exist_ok=True)
    
    # Download pre-computed data
    for data_name, data_info in precomputed_data.items():
        if data_info['is_zip']:
            # For zip files, check if the extraction directory already has files
            extract_dir = data_info.get('extract_to', os.path.dirname(data_info['path']))
            if os.path.exists(extract_dir) and os.listdir(extract_dir):
                if extract_dir == 'latents':
                    logger.info(f"{extract_dir} directory already contains files, skipping download of {data_name}.")
                elif extract_dir == '.' and os.path.exists('AlignedTest2'):
                    logger.info(f"AlignedTest2 directory already exists, skipping download of {data_name}.")
                else:
                    # Download and extract the zip file
                    logger.info(f"Downloading pre-computed {data_name}...")
                    download_gdrive_file(data_info['id'], data_info['path'], f"Downloading {data_name}")
                    
                    # Extract the zip file
                    extract_zip(data_info['path'], extract_dir)
                    
                    # Clean up the zip file
                    os.remove(data_info['path'])
            else:
                # Download and extract the zip file
                logger.info(f"Downloading pre-computed {data_name}...")
                download_gdrive_file(data_info['id'], data_info['path'], f"Downloading {data_name}")
                
                # Extract the zip file
                extract_zip(data_info['path'], extract_dir)
                
                # Clean up the zip file
                os.remove(data_info['path'])
        else:
            # For non-zip files, simply download if they don't exist
            if not os.path.exists(data_info['path']):
                logger.info(f"Downloading pre-computed {data_name}...")
                download_gdrive_file(data_info['id'], data_info['path'], f"Downloading {data_name}")
                
    # Create symbolic links to ensure files are found in expected locations
    logger.info("Creating symbolic links for model files...")
    os.makedirs('encoder4editing/pretrained_models', exist_ok=True)
    
    # Create symlinks for e4e model
    if os.path.exists('pretrained_models/e4e_ffhq_encode.pt'):
        try:
            if not os.path.exists('encoder4editing/pretrained_models/e4e_ffhq_encode.pt'):
                os.symlink(
                    os.path.abspath('pretrained_models/e4e_ffhq_encode.pt'),
                    'encoder4editing/pretrained_models/e4e_ffhq_encode.pt'
                )
                logger.info("Created symlink for e4e model in encoder4editing/pretrained_models/")
        except OSError as e:
            logger.warning(f"Failed to create symlink: {e}")
            # If symlink fails, copy the file instead
            shutil.copy2('pretrained_models/e4e_ffhq_encode.pt', 'encoder4editing/pretrained_models/e4e_ffhq_encode.pt')
            logger.info("Copied e4e model to encoder4editing/pretrained_models/ instead")
            
    # Create symlinks for face predictor
    if os.path.exists('pretrained_models/shape_predictor_68_face_landmarks.dat'):
        try:
            if not os.path.exists('shape_predictor_68_face_landmarks.dat'):
                os.symlink(
                    os.path.abspath('pretrained_models/shape_predictor_68_face_landmarks.dat'),
                    'shape_predictor_68_face_landmarks.dat'
                )
                logger.info("Created symlink for face predictor in root directory")
        except OSError as e:
            logger.warning(f"Failed to create symlink: {e}")
            # If symlink fails, copy the file instead
            shutil.copy2('pretrained_models/shape_predictor_68_face_landmarks.dat', 'shape_predictor_68_face_landmarks.dat')
            logger.info("Copied face predictor to root directory instead")

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
        'CSVs',
        'sample_images/fathers',
        'sample_images/mothers',
        'sample_images/children'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def update_train_py():
    """Update the train.py file to use the correct paths."""
    logger.info("Updating train.py with correct paths...")
    
    train_py_path = "train.py"
    if os.path.exists(train_py_path):
        with open(train_py_path, 'r') as f:
            content = f.read()
        
        # Update the paths
        content = content.replace(
            "'/content/drive/MyDrive/Child Generator/AlignedTest2'", 
            "'./AlignedTest2'"
        )
        content = content.replace(
            "'/content/drive/MyDrive/Child Generator/CSVs/checkpoint10.csv'", 
            "'./CSVs/checkpoint10.csv'"
        )
        
        # Write the updated content back
        with open(train_py_path, 'w') as f:
            f.write(content)
        
        logger.info("Successfully updated paths in train.py")
    else:
        logger.warning("train.py not found. Please update the paths manually.")

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
    
    # Update train.py with correct paths
    update_train_py()
    
    logger.info("""
Setup complete! The system is ready to go:
1. Pre-computed latents and child embeddings have been downloaded
2. AlignedTest2 folder and checkpoint CSV have been downloaded
3. All required models have been installed
4. Directory structure has been created
5. train.py has been updated with correct paths

To start training:
1. Place your family images in the appropriate directories (if you plan to use your own images):
   - sample_images/fathers/
   - sample_images/mothers/
   - sample_images/children/
2. Run: python train.py
""")

if __name__ == "__main__":
    main() 