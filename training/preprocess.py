from pathlib import Path
import tensorflow as tf

# Dataset path
DATASET_PATH = "dataset/archive/PlantVillage"

# Image size required by MobileNetV2
IMAGE_SIZE = (224, 224)

# Batch size
BATCH_SIZE = 32

# Load dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

print("\nDataset Loaded Successfully!\n")

print("Classes:")
print(train_dataset.class_names)

print("\nNumber of Classes:", len(train_dataset.class_names))