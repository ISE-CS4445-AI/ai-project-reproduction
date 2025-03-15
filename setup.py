import os
import subprocess
import logging
import sys
import requests
from tqdm import tqdm
import gdown  # For Google Drive folder download
import zipfile  # For extracting zip files
import shutil  # For directory operations

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_working_directory():
    """Set the working directory to the correct location."""
    # If we're in Colab, use /content
    if os.path.exists('/content'):
        target_dir = '/content'
    else:
        # Otherwise use the directory where setup.py is located
        target_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the target directory
    os.chdir(target_dir)
    logger.info(f"Working directory set to: {os.getcwd()}")
    
    # Create a marker file to help verify location
    with open(os.path.join(target_dir, '.project_root'), 'w') as f:
        f.write('This file marks the root directory of the project')

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
    """Download a single file from Google Drive with proper handling of confirmation pages."""
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
    
    # Download the file with proper handling of confirmation prompts
    url = f"https://drive.google.com/uc?id={file_id}"
    desc = description if description else os.path.basename(destination)
    
    try:
        # First attempt - if this fails with warning page, the except block will handle it
        logger.info(f"Downloading {desc} directly...")
        gdown.download(url, destination, quiet=False, fuzzy=True)  # fuzzy=True helps with warning pages
        
        # Check if file is HTML warning page (file size would be small)
        if os.path.exists(destination) and os.path.getsize(destination) < 50000:  # Less than 50KB
            with open(destination, 'r', errors='ignore') as f:
                content = f.read(200)
                if '<!DOCTYPE html>' in content or '<html>' in content:
                    logger.warning("Detected HTML warning page instead of model file. Retrying with special flags...")
                    os.remove(destination)  # Remove the HTML file
                    raise Exception("Got HTML warning page instead of file")
        
        logger.info(f"Successfully downloaded {desc} to {destination}")
    except Exception as e:
        logger.warning(f"Standard download failed: {e}. Trying with force-cookies flag...")
        # Force download using special flags to bypass warning page
        gdown.download(url, destination, quiet=False, fuzzy=True, use_cookies=False)
        
        # Check again for HTML content
        if os.path.exists(destination):
            with open(destination, 'r', errors='ignore') as f:
                content = f.read(200)
                if '<!DOCTYPE html>' in content or '<html>' in content:
                    logger.error("Still getting HTML warning page. Trying one last method...")
                    os.remove(destination)  # Remove the HTML file
                    
                    # Try one more time with a different approach
                    output = subprocess.check_output(
                        f"gdown --id {file_id} -O {destination} --fuzzy --no-cookies",
                        shell=True,
                        stderr=subprocess.STDOUT
                    )
                    logger.info(f"gdown output: {output.decode('utf-8', errors='ignore')}")
                
        logger.info(f"Successfully downloaded {desc} to {destination} (retry method)")

def extract_zip(zip_path, extract_to):
    """Extract a zip file to the specified directory and fix nested directories."""
    logger.info(f"Extracting {zip_path} to {extract_to}...")
    
    # Create extraction directory if it doesn't exist
    os.makedirs(extract_to, exist_ok=True)
    
    # Get the basename without extension to check for nested directories with same name
    zip_basename = os.path.splitext(os.path.basename(zip_path))[0]
    
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    # Check for nested directory with the same name as the target directory
    nested_dir = os.path.join(extract_to, zip_basename)
    if os.path.isdir(nested_dir) and os.path.exists(nested_dir):
        logger.info(f"Detected nested directory {nested_dir}, fixing directory structure...")
        
        # Move all files from nested directory to parent directory
        for item in os.listdir(nested_dir):
            source = os.path.join(nested_dir, item)
            dest = os.path.join(extract_to, item)
            
            # Handle existing files/directories
            if os.path.exists(dest):
                if os.path.isdir(dest):
                    # Merge directories
                    for subitem in os.listdir(source):
                        shutil.move(
                            os.path.join(source, subitem),
                            os.path.join(dest, subitem)
                        )
                else:
                    # For files, add a suffix to avoid conflicts
                    base, ext = os.path.splitext(dest)
                    counter = 1
                    while os.path.exists(dest):
                        dest = f"{base}_{counter}{ext}"
                        counter += 1
                    shutil.move(source, dest)
            else:
                # Simple move if destination doesn't exist
                shutil.move(source, dest)
                
        # Remove the now-empty nested directory
        try:
            os.rmdir(nested_dir)
            logger.info(f"Removed empty nested directory: {nested_dir}")
        except OSError:
            # Directory might not be empty due to hidden files, etc.
            logger.warning(f"Could not remove directory {nested_dir}, it may not be empty")
    
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
            'id': '1cUv_reLE6k3604or78EranS7XzuVMWeO',
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
            
            if model_name == 'e4e_ffhq_encode':
                # Use download_gdrive_file for Google Drive downloads
                download_gdrive_file(model_info['id'], model_info['path'], f"Downloading {model_name}")
            else:
                # Use regular download for standard URLs
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
            
            # Special case for AlignedTest2 - check if it exists directly
            if data_name == 'aligned_test':
                if not os.path.exists('AlignedTest2'):
                    logger.info(f"AlignedTest2 directory not found. Downloading {data_name}...")
                    download_gdrive_file(data_info['id'], data_info['path'], f"Downloading {data_name}")
                    extract_zip(data_info['path'], extract_dir)
                    os.remove(data_info['path'])  # Clean up zip after extraction
                else:
                    logger.info("AlignedTest2 directory already exists, skipping download.")
                continue  # Skip the rest of the loop for this item
                
            # Handle other zip files
            if os.path.exists(extract_dir) and os.listdir(extract_dir):
                if data_name == 'latents':
                    # Special handling for latents.zip to ensure proper directory structure
                    if not os.path.exists(extract_dir) or (os.path.exists(extract_dir) and len(os.listdir(extract_dir)) == 0):
                        # Download and extract the zip file
                        logger.info(f"Downloading pre-computed {data_name}...")
                        download_gdrive_file(data_info['id'], data_info['path'], f"Downloading {data_name}")
                        
                        # Extract the zip file with special handling for nested directories
                        extract_zip(data_info['path'], extract_dir)
                        
                        # Verify that latent files are in the correct location
                        if not any(f.startswith('father_latent_') for f in os.listdir(extract_dir)):
                            logger.warning(f"No latent files found directly in {extract_dir}. Checking for nested directory...")
                            
                            # Check if there's a nested 'latents' directory
                            nested_latents = os.path.join(extract_dir, 'latents')
                            if os.path.exists(nested_latents) and os.path.isdir(nested_latents):
                                logger.info(f"Found nested latents directory. Moving files to {extract_dir}...")
                                
                                # Move all files from nested latents to parent directory
                                for item in os.listdir(nested_latents):
                                    shutil.move(
                                        os.path.join(nested_latents, item),
                                        os.path.join(extract_dir, item)
                                    )
                                
                                # Remove the empty nested directory
                                try:
                                    os.rmdir(nested_latents)
                                except OSError:
                                    logger.warning(f"Could not remove {nested_latents}, it may not be empty")
                        
                        # Clean up the zip file
                        os.remove(data_info['path'])
                    else:
                        logger.info(f"{extract_dir} directory already contains files, skipping download of {data_name}.")
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

def verify_image_files():
    """Verify that all image files referenced in the CSV file exist in the AlignedTest2 directory."""
    csv_path = "CSVs/checkpoint10.csv"
    base_path = "AlignedTest2"
    
    if not os.path.exists(csv_path):
        logger.warning(f"CSV file {csv_path} not found. Cannot verify image files.")
        return
    
    if not os.path.exists(base_path):
        logger.error(f"AlignedTest2 directory not found. Images cannot be verified.")
        return
    
    logger.info("Verifying image files referenced in the CSV...")
    
    try:
        import pandas as pd
        
        # Load the CSV
        df = pd.read_csv(csv_path)
        missing_files = []
        
        # Parse string lists in the CSV
        for list_col_name in ["mother_images", "father_images", "child_images"]:
            if list_col_name in df.columns:
                try:
                    df[list_col_name] = df[list_col_name].map(eval)
                except Exception as e:
                    logger.error(f"Error parsing {list_col_name} column: {e}")
                    continue
        
        # Check all image paths
        total_files = 0
        missing_count = 0
        
        for _, family in df.iterrows():
            for column in ["mother_images", "father_images", "child_images"]:
                if column in family and isinstance(family[column], list):
                    for img_path in family[column]:
                        total_files += 1
                        full_path = os.path.join(base_path, img_path)
                        if not os.path.exists(full_path):
                            missing_files.append(img_path)
                            missing_count += 1
        
        # Report results
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing files out of {total_files} referenced in the CSV.")
            logger.warning(f"First few missing files: {missing_files[:5]}")
            
            # Write missing files to a log file for reference
            with open("missing_images.log", "w") as f:
                for file in missing_files:
                    f.write(f"{file}\n")
            logger.info("Full list of missing files written to missing_images.log")
        else:
            logger.info(f"All {total_files} image files referenced in the CSV exist!")
    
    except Exception as e:
        logger.error(f"Error verifying image files: {e}")

def download_aligned_test2():
    """
    Special function dedicated to downloading and extracting AlignedTest2.
    This function tries multiple approaches to ensure AlignedTest2 is available.
    """
    # Ensure we're in the correct directory
    if os.path.exists('/content'):
        base_dir = '/content'
    else:
        base_dir = os.getcwd()
        
    alignedtest_id = '1VoKZyFXG8HpTbfgMtJ24qsE9J81tZvte'
    zip_path = os.path.join(base_dir, 'AlignedTest2.zip')
    target_dir = os.path.join(base_dir, 'AlignedTest2')
    
    logger.info(f"Will download AlignedTest2 to: {target_dir}")
    
    # If AlignedTest2 exists in the wrong location, try to move it
    wrong_location = '/content/content/drive/MyDrive/Child Generator/AlignedTest2'
    if os.path.exists(wrong_location) and not os.path.exists(target_dir):
        logger.info(f"Found AlignedTest2 in wrong location: {wrong_location}")
        try:
            shutil.move(wrong_location, target_dir)
            logger.info(f"Successfully moved AlignedTest2 to: {target_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to move directory: {e}")
    
    if os.path.exists(target_dir) and os.listdir(target_dir):
        logger.info(f"AlignedTest2 directory already exists at {target_dir}, skipping download.")
        return True
    
    logger.info("=== DOWNLOADING AlignedTest2 DATASET (THIS MAY TAKE SOME TIME) ===")
    
    # Try multiple methods to download the file
    try:
        logger.info("Attempting download using gdown...")
        # First method: Standard gdown
        gdown.download(
            f"https://drive.google.com/uc?id={alignedtest_id}", 
            zip_path, 
            quiet=False, 
            fuzzy=True
        )
        
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1000000:  # Less than 1MB
            # Second method: gdown with no cookies
            logger.info("First attempt failed, trying with no cookies...")
            if os.path.exists(zip_path):
                os.remove(zip_path)
                
            gdown.download(
                f"https://drive.google.com/uc?id={alignedtest_id}", 
                zip_path, 
                quiet=False, 
                fuzzy=True, 
                use_cookies=False
            )
        
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1000000:  # Less than 1MB
            # Third method: command line gdown
            logger.info("Second attempt failed, trying command line gdown...")
            if os.path.exists(zip_path):
                os.remove(zip_path)
                
            subprocess.run(
                f"gdown --id {alignedtest_id} -O {zip_path} --fuzzy",
                shell=True,
                check=True
            )
        
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1000000:  # Less than 1MB
            logger.error("Failed to download AlignedTest2 dataset after multiple attempts.")
            os.makedirs(target_dir, exist_ok=True)
            return False
            
        # If we got here, we've successfully downloaded the zip file
        logger.info(f"Successfully downloaded AlignedTest2.zip ({os.path.getsize(zip_path)} bytes)")
        logger.info(f"Extracting AlignedTest2.zip to {base_dir}...")
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
        
        # Check if extraction created a nested directory
        if not os.path.exists(target_dir):
            logger.warning("Extraction may have created nested directory. Checking...")
            
            # Check common nested directory patterns
            for potential_dir in ['AlignedTest2', 'alignedtest2', 'aligned_test2', 'aligned-test2']:
                potential_path = os.path.join(base_dir, potential_dir)
                if os.path.exists(potential_path):
                    if potential_path != target_dir:
                        logger.info(f"Renaming {potential_path} to {target_dir}")
                        os.rename(potential_path, target_dir)
                    break
        
        # Cleanup
        if os.path.exists(zip_path):
            os.remove(zip_path)
            
        if os.path.exists(target_dir) and os.listdir(target_dir):
            logger.info(f"AlignedTest2 download and extraction completed successfully to {target_dir}")
            return True
        else:
            logger.error(f"AlignedTest2 directory not found at {target_dir} after extraction.")
            return False
            
    except Exception as e:
        logger.error(f"Error downloading or extracting AlignedTest2: {str(e)}")
        os.makedirs(target_dir, exist_ok=True)
        return False

def main():
    logger.info("Starting setup...")
    
    # Set the working directory first
    set_working_directory()
    
    # First priority: Download AlignedTest2
    logger.info("Downloading AlignedTest2 dataset (critical for training)...")
    alignedtest_success = download_aligned_test2()
    
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
    
    # Verify image files exist
    verify_image_files()
    
    # Final status message
    logger.info("""
Setup complete! The system is ready to go:
1. Pre-computed latents and child embeddings have been downloaded
2. AlignedTest2 folder and checkpoint CSV have been downloaded
3. All required models have been installed
4. Directory structure has been created
5. train.py has been updated with correct paths
6. Image file verification has been performed
""")

    # Show warning if AlignedTest2 download failed
    if not alignedtest_success:
        logger.warning("""
⚠️ WARNING: The AlignedTest2 dataset could not be downloaded properly.
You may encounter errors during training related to missing image files.
Please consider manually downloading the dataset from:
https://drive.google.com/uc?id=1VoKZyFXG8HpTbfgMtJ24qsE9J81tZvte
and extracting it to create an AlignedTest2 folder in your working directory.
""")
    
    logger.info("""
To start training:
1. Place your family images in the appropriate directories (if you plan to use your own images):
   - sample_images/fathers/
   - sample_images/mothers/
   - sample_images/children/
2. Run: python train.py
""")

if __name__ == "__main__":
    main() 