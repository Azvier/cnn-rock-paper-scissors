import os
import shutil
import numpy as np

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
original_dataset_path = os.path.join(script_dir, '../image-dataset/rockpaperscissors/rps-cv-images')
original_dataset_path = os.path.normpath(original_dataset_path)
classes = ['paper', 'rock', 'scissors']

train_split = 0.7
validation_split = 0.15

# --- Main Logic ---
print("Starting dataset reorganization...")

for class_name in classes:
    print(f"\nProcessing class: {class_name}")

    source_dir = os.path.join(original_dataset_path, class_name)
    train_dir = os.path.join(original_dataset_path, 'train', class_name)
    validation_dir = os.path.join(original_dataset_path, 'validation', class_name)
    test_dir = os.path.join(original_dataset_path, 'test', class_name)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    images = os.listdir(source_dir)

    images = [f for f in images if os.path.isfile(os.path.join(source_dir, f))]
    
    np.random.shuffle(images)

    train_end = int(len(images) * train_split)
    validation_end = int(len(images) * (train_split + validation_split))

    train_images = images[:train_end]
    validation_images = images[train_end:validation_end]
    test_images = images[validation_end:]

    def move_files(file_list, destination_dir):
        for image in file_list:
            shutil.move(os.path.join(source_dir, image), os.path.join(destination_dir, image))


    move_files(train_images, train_dir)
    move_files(validation_images, validation_dir)
    move_files(test_images, test_dir)

    print(f"Moved {len(train_images)} to train, {len(validation_images)} to validation, {len(test_images)} to test.")

    shutil.rmtree(source_dir)

print("\nDataset reorganization complete")