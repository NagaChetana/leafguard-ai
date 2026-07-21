from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image

# Dataset path
DATASET_PATH = Path("dataset/archive/PlantVillage")

# Get all classes
classes = [folder for folder in DATASET_PATH.iterdir() if folder.is_dir()]

# Choose a random class
selected_class = random.choice(classes)

# Get all images from that class
images = list(selected_class.glob("*"))

# Pick one random image
image_path = random.choice(images)

# Open image
image = Image.open(image_path)

# Display image
plt.imshow(image)
plt.title(selected_class.name)
plt.axis("off")

print(f"Class: {selected_class.name}")
print(f"Image Size: {image.size}")

plt.show()