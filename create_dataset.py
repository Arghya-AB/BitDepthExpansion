import os
from PIL import Image
import numpy as np

def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs('trainsets/trainH', exist_ok=True)
    os.makedirs('trainsets/trainL', exist_ok=True)

def make_square(img):
    """Create largest possible square from image."""
    width, height = img.size
    size = min(width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size
    return img.crop((left, top, right, bottom))

def quantize_to_3bit(img_array):
    """Convert image to 3-bit by setting last 5 bits to 0."""
    # Mask that sets last 5 bits to 0 
    return img_array & 0b11100000

def process_image(filepath):
    """Process a single image."""
    try:
        # Open and convert image to RGB
        img = Image.open(filepath).convert('RGB')
        
        # Make the image square and resize to 128x128
        img_square = make_square(img)
        img_128 = img_square.resize((128, 128), Image.Resampling.LANCZOS)
        
        # Get filename without path
        filename = os.path.basename(filepath)
        base, ext = os.path.splitext(filename)
        
        # Save high-quality version
        high_quality_path = os.path.join('trainsets/trainH', f'{base}.png')
        img_128.save(high_quality_path)
        
        # Create 3-bit version
        img_array = np.array(img_128)
        img_3bit = quantize_to_3bit(img_array)
        img_3bit_pil = Image.fromarray(img_3bit.astype('uint8'))
        
        # Save low-quality version
        low_quality_path = os.path.join('trainsets/trainL', f'{base}.png')
        img_3bit_pil.save(low_quality_path)
        
        return True
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return False

def process_directory(root_dir):
    """Recursively process all images in directory."""
    # Create output directories
    create_directories()
    
    # Supported image extensions
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    
    # Counter for processed images
    processed = 0
    failed = 0
    
    # Walk through directory
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in valid_extensions:
                filepath = os.path.join(dirpath, filename)
                if process_image(filepath):
                    processed += 1
                else:
                    failed += 1
    
    return processed, failed

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory_path>")
        sys.exit(1)
    
    root_directory = sys.argv[1]
    if not os.path.isdir(root_directory):
        print("Error: Provided path is not a directory")
        sys.exit(1)
    
    processed, failed = process_directory(root_directory)
    print(f"Processing complete!")
    print(f"Successfully processed: {processed} images")
    print(f"Failed to process: {failed} images")