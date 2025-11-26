import os
import shutil
import random

# Input folders
input_root = "/home/ahzsyed/Documents/GitHub/Perception12/Final_Project/data/classifier/train"
classes = ["car", "cyclist", "pedestrian"]

# Output YOLO folders
output_root = "YOLO/dataset"
os.makedirs(f"{output_root}/images/train", exist_ok=True)
os.makedirs(f"{output_root}/images/val", exist_ok=True)
os.makedirs(f"{output_root}/labels/train", exist_ok=True)
os.makedirs(f"{output_root}/labels/val", exist_ok=True)

train_split = 0.8  # 80% train, 20% val

for class_name in classes:
    class_folder = os.path.join(input_root, class_name)
    images = [f for f in os.listdir(class_folder)
              if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    random.shuffle(images)

    split_index = int(len(images) * train_split)
    train_images = images[:split_index]
    val_images = images[split_index:]

    for img in train_images:
        shutil.copy(os.path.join(class_folder, img),
                    f"{output_root}/images/train/{class_name}_{img}")

    for img in val_images:
        shutil.copy(os.path.join(class_folder, img),
                    f"{output_root}/images/val/{class_name}_{img}")

print("Dataset structure prepared. Now annotate using LabelImg!")
