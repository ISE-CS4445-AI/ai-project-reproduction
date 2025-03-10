#!/usr/bin/env python3
"""
Example script showing how to use E4E with memory-efficient mode.
This is a simplified example for direct code usage rather than CLI.
"""

import os
import sys
import logging
import argparse
from e4e_lib import E4EProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='E4E Example with Memory-Efficient Mode')
    parser.add_argument('--no-mixed-precision', action='store_true',
                       help='Disable mixed precision inference')
    parser.add_argument('--input', type=str, default=None,
                       help='Path to input image (overrides default)')
    
    args = parser.parse_args()
    
    # Define input and output paths
    images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
    input_image = args.input if args.input else os.path.join(images_dir, 'input.jpg')
    output_dir = os.path.join(images_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if input image exists
    if not os.path.exists(input_image):
        logger.error(f"Input image not found at: {input_image}")
        return
    
    # Initialize the processor with memory-efficient mode enabled
    logger.info("Initializing E4E processor with memory-efficient mode...")
    logger.info(f"Mixed precision is {'DISABLED' if args.no_mixed_precision else 'ENABLED'}")
    
    processor = E4EProcessor(
        experiment_type='ffhq_encode',
        memory_efficient=True,  # Enable memory-efficient mode
        enable_mixed_precision=not args.no_mixed_precision  # Control mixed precision
    )
    
    try:
        # Process an image
        logger.info("Processing input image...")
        try:
            result_image, latent, processed_image = processor.process_image(input_image)
            
            # Save the result
            result_path = os.path.join(output_dir, 'result_memory_efficient.jpg')
            result_image.save(result_path)
            logger.info(f"Saved result to: {result_path}")
            
            # Apply an edit (as an example)
            logger.info("Applying age edit...")
            try:
                edited_image = processor.edit_image(
                    latent,
                    edit_type='interfacegan',
                    direction_name='age',
                    factor=-3
                )
                
                # Save the edited image
                edited_path = os.path.join(output_dir, 'edited_memory_efficient.jpg')
                edited_image.save(edited_path)
                logger.info(f"Saved edited image to: {edited_path}")
                
                logger.info("Processing complete!")
            except Exception as e:
                logger.error(f"Error during editing: {str(e)}")
                logger.error("Skipping edit step...")
        except Exception as e:
            logger.error(f"Error during image processing: {str(e)}")
            logger.error("This might be due to memory issues or invalid input.")
            
    finally:
        # Always clean up resources when done
        logger.info("Cleaning up resources...")
        try:
            cleanup_success = processor.cleanup()
            if cleanup_success:
                logger.info("Cleanup completed successfully.")
            else:
                logger.warning("Cleanup may not have been fully successful.")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    main() 