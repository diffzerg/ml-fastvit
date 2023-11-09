import os
import shutil
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Define your base directories
original_dataset_dir = './images'
base_dir = './dataset'

transform_pipeline = transforms.Compose([
    transforms.Resize((96, 64)),  # Resize the image to 96x64 pixels
    transforms.CenterCrop((96, 64)),  # Crop the center to 96x64 pixels if necessary
])

def process_and_save_images(image_files, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for img_file in image_files:
        image = Image.open(os.path.join(original_dataset_dir, img_file))
        # Convert image to 'RGB' if it's 'P' or 'RGBA'
        if image.mode == 'P' or image.mode == 'RGBA':
            image = image.convert('RGB')
        transformed_image = transform_pipeline(image)
        save_path = os.path.join(target_dir, img_file)
        # Save as JPEG
        transformed_image.save(save_path, 'JPEG')
        
# List of valid image extensions (you can add more if needed)
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

# Read all image files with valid extensions
all_image_files = [f for f in os.listdir(original_dataset_dir) if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(original_dataset_dir, f))]

# Split the data into training and validation sets
train_files, validation_files = train_test_split(all_image_files, test_size=0.2)

# Process and save training images
process_and_save_images(train_files, os.path.join(base_dir, 'train'))

# Process and save validation images
process_and_save_images(validation_files, os.path.join(base_dir, 'validation'))

print("Images have been processed and split into training and validation sets.")
