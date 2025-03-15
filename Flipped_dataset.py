import os
import cv2

# Define original and new dataset directories
original_dataset = "data"  # Your existing dataset
flipped_dataset = "flipped_data"  # New dataset for flipped images

# Create flipped dataset directory if not exists
if not os.path.exists(flipped_dataset):
    os.makedirs(flipped_dataset)

# Loop through each class folder (0 to 26)
for class_label in range(27):  # Since you have folders 0 to 26
    original_folder = os.path.join(original_dataset, str(class_label))
    flipped_folder = os.path.join(flipped_dataset, str(class_label))

    # Create class folder in flipped dataset
    if not os.path.exists(flipped_folder):
        os.makedirs(flipped_folder)

    # Process each image in the class folder
    for filename in os.listdir(original_folder):
        img_path = os.path.join(original_folder, filename)

        # Read image
        image = cv2.imread(img_path)
        if image is None:
            continue  # Skip if the image is not readable

        # Flip image horizontally
        flipped_image = cv2.flip(image, 1)

        # Save flipped image with new name
        flipped_img_path = os.path.join(flipped_folder, filename)
        cv2.imwrite(flipped_img_path, flipped_image)

print("Flipped dataset created successfully in 'flipped_data/'!")
    