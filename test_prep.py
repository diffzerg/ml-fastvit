import os
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split

# Define the path to the original dataset and the new directory structure
original_dataset_dir = './images'
base_dir = './dataset'

# Create directories
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
if not os.path.exists(train_dir):
    os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')
if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)

# Get all image filenames
all_images = [fname for fname in os.listdir(original_dataset_dir)
              if fname.endswith('.jpg')]

# Extract class names and sort them into unique categories
class_names = sorted(list(set('_'.join(fname.split('_')[:-1]) for fname in all_images)))

# Limit to 30 classes
class_names = class_names[:30]

# Create subdirectories for class names
for class_name in class_names:
    # Create subdirectories for training data
    class_train_dir = os.path.join(train_dir, class_name)
    if not os.path.exists(class_train_dir):
        os.mkdir(class_train_dir)

    # Create subdirectories for validation data
    class_validation_dir = os.path.join(validation_dir, class_name)
    if not os.path.exists(class_validation_dir):
        os.mkdir(class_validation_dir)

# Group images by class
class_to_images = {class_name: [] for class_name in class_names}
for image in all_images:
    class_name = '_'.join(image.split('_')[:-1])
    if class_name in class_to_images:
        class_to_images[class_name].append(image)

# Select one image per class for training and one for validation
train_images = []
validation_images = []
for class_name, images in class_to_images.items():
    if len(images) >= 2:
        train_images.append(images[0])
        validation_images.append(images[1])
    else:
        # If a class doesn't have enough images, output a message and skip the class
        print(f"Not enough images for class {class_name}, needs at least 2, has {len(images)}")

# Function to resize and copy images to new structure
def resize_and_copy_images(images, source_dir, target_dir):
    for image in images:
        class_name = '_'.join(image.split('_')[:-1])
        shutil.copy(os.path.join(source_dir, image), os.path.join(target_dir, class_name, image))

        src_path = os.path.join(source_dir, image)
        dst_path = os.path.join(target_dir, class_name, image)
        
        with Image.open(src_path) as img:
            img = img.resize((64, 96), Image.Resampling.LANCZOS)  # Resize the image to 64x96 pixels using LANCZOS
            img = img.convert('RGB')  # Convert the image to RGB mode
            img.save(dst_path, 'JPEG')  # Save the resized image as a JPEG


# Copy images to the respective directories
resize_and_copy_images(train_images, original_dataset_dir, train_dir)
resize_and_copy_images(validation_images, original_dataset_dir, validation_dir)

print('Dataset successfully restructured with images resized to 96x64 pixels.')
