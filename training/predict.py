import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# ==========================================
# Configuration
# ==========================================

MODEL_PATH = "backend/models/plant_disease_model.keras"
DATASET_PATH = Path("dataset/archive/PlantVillage")
IMAGE_PATH = "test_images/tomato.jpg"

IMAGE_SIZE = (224, 224)

# ==========================================
# Load Model
# ==========================================

model = tf.keras.models.load_model(MODEL_PATH)

# ==========================================
# Load Class Names Automatically
# ==========================================

class_names = sorted(
    [folder.name for folder in DATASET_PATH.iterdir() if folder.is_dir()]
)

# ==========================================
# Load Image
# ==========================================

image = Image.open(IMAGE_PATH).convert("RGB")

print(f"\nImage : {IMAGE_PATH}")
print(f"Original Size : {image.size}")

plt.imshow(image)
plt.title("Uploaded Image")
plt.axis("off")
plt.show()

# Resize
image = image.resize(IMAGE_SIZE)

# Convert to NumPy
image = np.array(image, dtype=np.float32)

# ==========================================
# MobileNetV2 Preprocessing
# ==========================================

image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

# Add Batch Dimension
image = np.expand_dims(image, axis=0)

# ==========================================
# Predict
# ==========================================

predictions = model.predict(image, verbose=0)[0]

predicted_index = np.argmax(predictions)

confidence = predictions[predicted_index] * 100

# ==========================================
# Top 3 Predictions
# ==========================================

top3 = np.argsort(predictions)[::-1][:3]

print("\n========== TOP 3 PREDICTIONS ==========\n")

for i in top3:
    print(f"{class_names[i]:45} {predictions[i]*100:.2f}%")

print("\n=======================================")
print(f"Final Prediction : {class_names[predicted_index]}")
print(f"Confidence       : {confidence:.2f}%")