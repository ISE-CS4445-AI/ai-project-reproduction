#!/usr/bin/env python3
import os
import sys
import logging
import argparse
from tqdm import tqdm
from e4e_lib import E4EProcessor

def main():
    # Set up argparse for CLI arguments
    parser = argparse.ArgumentParser(description='E4E Image Processing CLI Tool')
    
    # Input/output options
    parser.add_argument('--input', '-i', type=str, help='Path to the input image')
    parser.add_argument('--second', '-s', type=str, help='Path to the second image for combination')
    parser.add_argument('--output-dir', '-o', type=str, default='images/outputs', 
                        help='Directory to save output images')
    
    # Processing options
    parser.add_argument('--experiment-type', '-e', type=str, default='ffhq_encode',
                        choices=['ffhq_encode', 'cars_encode'],
                        help='Type of experiment to run')
    
    # Memory optimization options
    parser.add_argument('--memory-efficient', '-m', action='store_true',
                        help='Enable memory-efficient mode to reduce GPU memory usage')
    parser.add_argument('--no-mixed-precision', action='store_true',
                        help='Disable mixed precision (FP16) inference')
    parser.add_argument('--force-mixed-precision', action='store_true',
                        help='Force enable mixed precision (overrides --no-mixed-precision)')
    parser.add_argument('--max-batch-size', type=int, default=1,
                        help='Maximum batch size for inference')
    
    # Editing options
    parser.add_argument('--edit', type=str, choices=['interfacegan', 'ganspace', 'sefa'],
                        help='Type of edit to apply')
    parser.add_argument('--direction', type=str, default='age',
                        help='Direction name for interfacegan edit')
    parser.add_argument('--factor', type=float, default=-3,
                        help='Factor value for interfacegan edit')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging for the main script
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # If no input is provided, show help and exit
    if args.input is None:
        parser.print_help()
        logger.error("No input image specified. Use --input or -i to specify an input image.")
        sys.exit(1)
        
    # Check if input image exists
    if not os.path.exists(args.input):
        logger.error(f"Input image not found at: {args.input}")
        sys.exit(1)
    
    # Check if second image exists if specified
    has_second_image = False
    if args.second:
        if not os.path.exists(args.second):
            logger.warning(f"Second image not found at: {args.second}")
            logger.warning("Skipping image combination")
        else:
            has_second_image = True
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Using output directory: {args.output_dir}")
    
    # Initialize the processor with specified options
    logger.info("Initializing E4E processor...")
    
    # Handle mixed precision settings
    use_mixed_precision = not args.no_mixed_precision
    if args.force_mixed_precision:
        use_mixed_precision = True
        logger.info("Forcing mixed precision mode")
    
    if args.memory_efficient:
        logger.info("Memory-efficient mode is ENABLED")
    
    with tqdm(total=1, desc="Processor initialization") as pbar:
        processor = E4EProcessor(
            experiment_type=args.experiment_type,
            memory_efficient=args.memory_efficient,
            enable_mixed_precision=use_mixed_precision,
            max_batch_size=args.max_batch_size
        )
        pbar.update(1)
    
    try:
        # Process the input image
        logger.info("Processing input image...")
        try:
            result_image, latent, processed_image = processor.process_image(args.input)
        except Exception as e:
            logger.error(f"Error processing input image: {str(e)}")
            logger.error("This might be due to memory issues or invalid input.")
            logger.error("Cleaning up and exiting...")
            processor.cleanup()
            sys.exit(1)
        
        # Save results
        with tqdm(total=1 + (3 if has_second_image else 0), desc="Saving results") as pbar:
            # Save the result image
            result_path = os.path.join(args.output_dir, 'result.jpg')
            result_image.save(result_path)
            logger.info(f"Saved result image to: {result_path}")
            pbar.update(1)
            
            # Process second image if available
            if has_second_image:
                logger.info("Combining images...")
                try:
                    combined_image, combined_latent, display_image = processor.combine_images(
                        args.input,
                        args.second
                    )
                    
                    # Save combined results
                    combined_path = os.path.join(args.output_dir, 'combined.jpg')
                    combined_image.save(combined_path)
                    logger.info(f"Saved combined image to: {combined_path}")
                    pbar.update(1)
                    
                    display_path = os.path.join(args.output_dir, 'display.jpg')
                    display_image.save(display_path)
                    logger.info(f"Saved display image to: {display_path}")
                    pbar.update(1)
                    
                    # Apply an edit if requested
                    if args.edit:
                        logger.info(f"Applying {args.edit} edit...")
                        
                        try:
                            if args.edit == 'interfacegan':
                                edited_image = processor.edit_image(
                                    combined_latent,
                                    edit_type=args.edit,
                                    direction_name=args.direction,
                                    factor=args.factor
                                )
                            elif args.edit == 'ganspace':
                                edited_image = processor.edit_image(
                                    combined_latent,
                                    edit_type=args.edit,
                                    direction_names=['eye_openness', 'smile']
                                )
                            elif args.edit == 'sefa':
                                edited_image = processor.edit_image(
                                    combined_latent,
                                    edit_type=args.edit,
                                    indices=[2, 3, 4, 5]
                                )
                            
                            edited_path = os.path.join(args.output_dir, 'edited.jpg')
                            edited_image.save(edited_path)
                            logger.info(f"Saved edited image to: {edited_path}")
                        except Exception as e:
                            logger.error(f"Error applying edit: {str(e)}")
                            logger.error("Skipping edit step...")
                    
                except Exception as e:
                    logger.error(f"Error combining images: {str(e)}")
                    logger.error("Skipping image combination steps...")
                    pbar.update(3)  # Skip remaining steps
        
        logger.info("Processing complete!")
        logger.info(f"All results have been saved to: {args.output_dir}")
        
    finally:
        # Always clean up resources, especially important for memory-efficient mode
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