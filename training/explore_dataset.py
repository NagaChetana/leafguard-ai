from pathlib import Path

# Dataset location
DATASET_PATH = Path("dataset/archive/PlantVillage")

# Get all disease folders
classes = [folder for folder in DATASET_PATH.iterdir() if folder.is_dir()]

print("=" * 50)
print("Plant Disease Detection Dataset")
print("=" * 50)

print(f"\nNumber of Classes : {len(classes)}\n")

for cls in sorted(classes):
    image_count = len(list(cls.glob("*")))
    print(f"{cls.name:40} {image_count} images")