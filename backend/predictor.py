import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image

# ==========================================
# Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "plant_disease_model.keras"
DATASET_PATH = BASE_DIR.parent / "dataset" / "archive" / "PlantVillage"

IMAGE_SIZE = (224, 224)

# ==========================================
# Load Model (Loads only once)
# ==========================================

model = tf.keras.models.load_model(str(MODEL_PATH))

# ==========================================
# Load Class Names
# ==========================================

class_names = sorted(
    [folder.name for folder in DATASET_PATH.iterdir() if folder.is_dir()]
)

# ==========================================
# Prediction Function
# ==========================================

def predict_image(image: Image.Image):

    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image = np.array(image, dtype=np.float32)

    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image, verbose=0)[0]

    predicted_index = np.argmax(predictions)

    confidence = float(predictions[predicted_index] * 100)

    return {
        "disease": class_names[predicted_index],
        "confidence": round(confidence, 2)
    }