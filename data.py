import os
from PIL import Image

path = "dataset"

# Create a new directory for resized images
new_path = os.path.join(path, "resized_images")
if not os.path.exists(new_path):
    os.makedirs(new_path)

# Iterate through all subdirectories (each representing a category)
for subdir, dirs, files in os.walk(os.path.join(path, "train")):
    # Get the category name from the subdirectory name
    category_name = os.path.basename(subdir)

    # Create a subdirectory for the current category in the new directory
    category_path = os.path.join(new_path, category_name)
    if not os.path.exists(category_path):
        os.makedirs(category_path)

    for file in files:
        # Open the image
        img = Image.open(os.path.join(subdir, file))
        img = img.convert('RGB')
        # Resize the image
        img_resized = img.resize((224,224))

        # Save the resized image in the new directory
        new_file_path = os.path.join(category_path, file)
        img_resized.save(new_file_path)
