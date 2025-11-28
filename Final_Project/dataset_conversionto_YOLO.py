import os
import shutil
import random

# Input folders
input_root = "data/YOLODataset"
classes = ["Cars", "Cycle", "Pedestrian"]  # Match folder names exactly

# Output YOLO folders
output_root = "YOLO/dataset"
os.makedirs(f"{output_root}/images/train", exist_ok=True)
os.makedirs(f"{output_root}/images/val", exist_ok=True)
os.makedirs(f"{output_root}/labels/train", exist_ok=True)
os.makedirs(f"{output_root}/labels/val", exist_ok=True)

train_split = 0.8  # 80% train, 20% val

for class_name in classes:
    class_folder = os.path.join(input_root, class_name)
    
    # List images only
    images = [f for f in os.listdir(class_folder)
              if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    random.shuffle(images)

    split_index = int(len(images) * train_split)
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Copy train images and labels
    for img in train_images:
        # Copy image
        shutil.copy(os.path.join(class_folder, img),
                    f"{output_root}/images/train/{class_name}_{img}")
        # Copy corresponding label
        label_file = os.path.splitext(img)[0] + ".txt"
        label_path = os.path.join(class_folder, label_file)
        if os.path.exists(label_path):
            shutil.copy(label_path, f"{output_root}/labels/train/{class_name}_{label_file}")
        else:
            print(f"Warning: Label not found for {img}")

    # Copy val images and labels
    for img in val_images:
        # Copy image
        shutil.copy(os.path.join(class_folder, img),
                    f"{output_root}/images/val/{class_name}_{img}")
        # Copy corresponding label
        label_file = os.path.splitext(img)[0] + ".txt"
        label_path = os.path.join(class_folder, label_file)
        if os.path.exists(label_path):
            shutil.copy(label_path, f"{output_root}/labels/val/{class_name}_{label_file}")
        else:
            print(f"Warning: Label not found for {img}")

print("Dataset structure prepared with images and labels for YOLO!")
