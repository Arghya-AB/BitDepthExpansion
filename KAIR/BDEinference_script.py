import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests
import models.network_loader
from models.network_bit_depth_expansion_Ultra import BitDepthSwinIR as net
from utils import utils_image as util
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(description='Image Inference Script')
    parser.add_argument('--input_folder', type=str, required=True, 
                        help='Path to folder containing input images')
    parser.add_argument('--expected_folder', type=str, default="None", 
                        help='Path to folder containing expected images')
    parser.add_argument('--output_folder', type=str, default='./results', 
                        help='Path to save output images')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to pre-trained model weights')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to run inference on')
    parser.add_argument('--framework', default='torch', help='Maybe future tensorflow support')
    return parser.parse_args()

def load_model(model_path, device):
    """Load pre-trained model."""
    model = net()
    if device == 'cuda':
        model = model.cuda()

    model = torch.load(model, model_path, map_location=device)
    return model


def calculate_metrics(original_path, processed_image):
    # Read original image and convert to RGB
    original = cv2.imread(original_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original = original.astype(np.float32)
    processed = processed_image.astype(np.float32)
    mse = np.mean((original - processed) ** 2)
    psnr = peak_signal_noise_ratio(original, processed, data_range=255)
    ssim = structural_similarity(original, processed, 
                               data_range=255,
                               channel_axis=2)  # specify channel axis for RGB
    
    print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr:.2f} dB")
    print(f"Structural Similarity Index (SSIM): {ssim:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    
    return psnr, ssim, mse


def run_inference(model, image_path, output_path, expected_path=None, device="cuda"):
    """Run model inference on a single image."""
    with torch.no_grad():
        input_img = models.network_loader.preprocess_image(image_path, device)
        # print(input_img)
        # Run inference
        output = model(input_img)
        print(output)
        
        # Convert output back to numpy for saving
        output_img = output.squeeze(0).cpu().numpy()
        output_img = np.transpose(output_img, (1, 2, 0))
        output_img = np.clip(output_img * 255.0, 0, 255).astype(np.uint8)
        if expected_path is not None:
            print(calculate_metrics(expected_path, output_img))
        else:
            print(output_img)
        # Save output image
        bgr_image = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        print()
        cv2.imwrite(output_path, bgr_image)

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)
    globals()[args.framework].load = models.network_loader.inference_model
    # Load model
    model = load_model(args.model_path, args.device)
    print(f"Model is loaded on device {args.device}")
    
    # Find all image files
    image_files = glob.glob(os.path.join(args.input_folder, '*.[jp][pn]g')) + \
                  glob.glob(os.path.join(args.input_folder, '*.tif*'))
    if args.expected_folder == "None":
        args.expected_folder = None
    # Process each image
    for image_path in image_files:
        filename = os.path.basename(image_path)
        if args.expected_folder is not None:
            expected_file = os.path.join(args.expected_folder, filename)
        else:
            expected_file = None
        output_path = os.path.join(args.output_folder, filename)
        
        print(f"Processing: {filename}")
        run_inference(model, image_path, output_path, expected_file, args.device)
    
    print(f"Inference completed. Results saved in {args.output_folder}")

if __name__ == '__main__':
    main()
