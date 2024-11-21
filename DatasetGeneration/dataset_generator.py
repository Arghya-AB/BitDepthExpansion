import os
import cv2
import numpy as np
import random

def convert_to_3bit(image):
    return np.round(image / 32) * 32

def random_crop(image, crop_height, crop_width):
    height, width = image.shape[:2]
    
    if width < crop_width or height < crop_height:
        return image 
    
    x = random.randint(0, width - crop_width)
    y = random.randint(0, height - crop_height)
    
    return image[y:y+crop_height, x:x+crop_width]

def process_images(input_folder, output_folder, input_full_bit_folder, crop_size=(256, 256), num_crops=4):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(input_full_bit_folder):
        os.makedirs(input_full_bit_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Error reading {filename}")
                continue

            base_name = os.path.splitext(filename)[0]

            for i in range(num_crops):
                cropped_img = random_crop(img, *crop_size)
                full_bit_path = os.path.join(input_full_bit_folder, f"{base_name}_crop{i+1}.png")
                cv2.imwrite(full_bit_path, cropped_img)
                img_3bit = convert_to_3bit(cropped_img)
                output_path = os.path.join(output_folder, f"{base_name}_crop{i+1}.png")
                cv2.imwrite(output_path, img_3bit)

            print(f"Processed {filename} - created {num_crops} crops")

input_folder = 'Kodak/'
output_folder = 'Target/'
input_full_bit_folder = 'Input/'
process_images(input_folder, output_folder, input_full_bit_folder)