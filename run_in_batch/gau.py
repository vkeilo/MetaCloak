import os
from PIL import Image, ImageFilter

# Define the directory containing the images
directory = '/data/home/yekai/github/mypro/MetaCloak/exp_data-1730000210/gen_output/release-MetaCloak-advance_steps-2-total_trail_num-4-unroll_steps-1-interval-200-total_train_steps-1000-SD21base-robust-gauK-7/dataset-VGGFace2-clean-r-11-model-SD21base-gen_prompt-sks/0/image_clean_ref'

# Gaussian blur kernel size
blur_radius = 7

# Process each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.png'):
        # Open the image
        image_path = os.path.join(directory, filename)
        with Image.open(image_path) as img:
            # Apply Gaussian blur
            blurred_img = img.filter(ImageFilter.GaussianBlur(blur_radius))
            
            # Save the image with the same filename
            blurred_img.save(image_path)

print("All PNG images have been blurred.")