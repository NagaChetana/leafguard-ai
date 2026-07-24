import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

# ==========================================
# Paths & Configuration
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "plant_disease_model.keras"
FIELD_HEALTH_MODEL_PATH = BASE_DIR / "models" / "plant_health_field.keras"
DATASET_PATH = BASE_DIR.parent / "dataset" / "archive" / "PlantVillage"

IMAGE_SIZE = (224, 224)

# ==========================================
# Load Class Names
# ==========================================

if DATASET_PATH.exists():
    class_names = sorted(
        [folder.name for folder in DATASET_PATH.iterdir() if folder.is_dir()]
    )
else:
    class_names = [
        "Pepper__bell___Bacterial_spot",
        "Pepper__bell___healthy",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
        "Tomato_Bacterial_spot",
        "Tomato_Early_blight",
        "Tomato_Late_blight",
        "Tomato_Leaf_Mold",
        "Tomato_Septoria_leaf_spot",
        "Tomato_Spider_mites_Two_spotted_spider_mite",
        "Tomato__Target_Spot",
        "Tomato__Tomato_YellowLeaf__Curl_Virus",
        "Tomato__Tomato_mosaic_virus",
        "Tomato_healthy"
    ]

# ==========================================
# Load Models (Loaded once at startup)
# ==========================================

# 1. Primary Disease Classification Model
model = tf.keras.models.load_model(str(MODEL_PATH))

# This model is trained on real-world PlantDoc field photos. It answers the
# first question before the older PlantVillage classifier is consulted: does
# this leaf look healthy or diseased at all?
field_health_model = (
    tf.keras.models.load_model(str(FIELD_HEALTH_MODEL_PATH), compile=False)
    if FIELD_HEALTH_MODEL_PATH.exists()
    else None
)
FIELD_HEALTHY_MAX_DISEASE_PROBABILITY = 0.35
FIELD_REVIEW_MAX_DISEASE_PROBABILITY = 0.65

HEALTHY_CLASS_INDICES = [
    index for index, name in enumerate(class_names) if "healthy" in name.lower()
]

# ==========================================
# Leaf Validation Function
# ==========================================

def get_foliage_ratio(image: Image.Image) -> float:
    """Return the portion of the image that looks like green or brown foliage."""
    arr = np.array(image.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    green_mask = cv2.inRange(hsv, np.array([25, 20, 20]), np.array([90, 255, 255]))
    brown_mask = cv2.inRange(hsv, np.array([10, 30, 30]), np.array([25, 255, 255]))
    foliage_mask = cv2.bitwise_or(green_mask, brown_mask)
    return float(np.count_nonzero(foliage_mask) / (arr.shape[0] * arr.shape[1]))


def is_plant_leaf(image: Image.Image) -> bool:
    """
    Validates if the input image is a plant leaf using:
    This is intentionally a lightweight rejection check, not a second disease
    model.  Requiring an ImageNet label such as ``leaf`` rejects many valid
    close-up leaf photographs because ImageNet does not have a general leaf
    class.
    """
    img_rgb = image.convert("RGB")
    arr = np.array(img_rgb)
    
    # 1. Noise Check
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var > 35000:
        return False

    # 2. Color / Foliage Mask Check
    foliage_ratio = get_foliage_ratio(image)
    
    # If foliage ratio is very low (< 8%), it's definitely not a leaf
    if foliage_ratio < 0.08:
        return False
        
    return foliage_ratio >= 0.12

# ==========================================
# Foliage Cropping Helper
# ==========================================

def get_foliage_crop(image: Image.Image) -> Image.Image:
    """
    Extracts and crops the main green/yellow foliage region to eliminate 
    background pots, soil, or background clutter.
    """
    img_rgb = image.convert("RGB")
    arr = np.array(img_rgb)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    
    green_mask = cv2.inRange(hsv, np.array([25, 20, 20]), np.array([90, 255, 255]))
    brown_mask = cv2.inRange(hsv, np.array([10, 30, 30]), np.array([25, 255, 255]))
    foliage_mask = cv2.bitwise_or(green_mask, brown_mask)
    
    contours, _ = cv2.findContours(foliage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        img_area = arr.shape[0] * arr.shape[1]
        valid_cnts = [c for c in contours if cv2.contourArea(c) > 0.01 * img_area]
        if valid_cnts:
            x_min = min([cv2.boundingRect(c)[0] for c in valid_cnts])
            y_min = min([cv2.boundingRect(c)[1] for c in valid_cnts])
            x_max = max([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in valid_cnts])
            y_max = max([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in valid_cnts])
            
            w = x_max - x_min
            h = y_max - y_min
            
            pad_w = int(w * 0.15)
            pad_h = int(h * 0.15)
            
            x1 = max(0, x_min - pad_w)
            y1 = max(0, y_min - pad_h)
            x2 = min(arr.shape[1], x_max + pad_w)
            y2 = min(arr.shape[0], y_max + pad_h)
            
            cropped = arr[y1:y2, x1:x2]
            return Image.fromarray(cropped)
            
    return image

# ==========================================
# Foliage Disease Lesion Helper
# ==========================================

def get_leaf_health_metrics(image: Image.Image):
    """
    Calculates proportion of green foliage tissue and necrotic decay.
    """
    arr = np.array(image.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    green_mask = cv2.inRange(hsv, np.array([25, 20, 30]), np.array([95, 255, 255]))
    green_px = np.sum(green_mask > 0)
    total_px = arr.shape[0] * arr.shape[1]
    
    if green_px == 0:
        return 0.0, 0.0
        
    necrotic_mask = cv2.inRange(hsv, np.array([0, 45, 25]), np.array([22, 255, 140])) & green_mask
    necrotic_px = np.sum(necrotic_mask > 0)
    
    green_ratio = float(green_px / total_px)
    necrotic_ratio = float(necrotic_px / green_px)
    
    return green_ratio, necrotic_ratio

# ==========================================
# Prediction Function
# ==========================================

def predict_image(image: Image.Image):
    image = image.convert("RGB")
    
    # 1. Validate if image contains a plant leaf
    if not is_plant_leaf(image):
        return {
            "is_valid": False,
            "error": "Invalid image. Please upload a valid plant leaf image."
        }

    foliage_ratio = get_foliage_ratio(image)

    # The saved models already contain MobileNetV2's preprocess_input layer.
    # Do NOT call preprocess_input here: doing so normalizes pixels twice.
    image_resized = image.resize(IMAGE_SIZE)
    image_array = np.array(image_resized, dtype=np.float32)
    image_batch = np.expand_dims(image_array, axis=0)

    field_disease_probability = None
    if field_health_model is not None:
        field_disease_probability = float(field_health_model(image_batch, training=False).numpy()[0][0])

    # A healthy verdict from the real-world field model takes priority. This
    # prevents the old disease-only classifier from forcing arbitrary healthy
    # leaves into labels such as Tomato Late blight.
    if field_disease_probability is not None and field_disease_probability <= FIELD_HEALTHY_MAX_DISEASE_PROBABILITY:
        raw_class = "Field_healthy"
        confidence = (1.0 - field_disease_probability) * 100
        is_uncertain = False
    elif field_disease_probability is not None and field_disease_probability < FIELD_REVIEW_MAX_DISEASE_PROBABILITY:
        raw_class = "Field_review"
        confidence = (1.0 - abs(field_disease_probability - 0.5) * 2) * 100
        is_uncertain = True
    else:
        predictions = model.predict(image_batch, verbose=0)[0]
        predicted_index = int(np.argmax(predictions))
        confidence = float(predictions[predicted_index] * 100)
        raw_class = class_names[predicted_index]
        healthy_probability = float(np.sum(predictions[HEALTHY_CLASS_INDICES]))
        best_healthy_index = max(HEALTHY_CLASS_INDICES, key=lambda i: predictions[i])

        # Fall back to a conservative uncertainty gate when the field model
        # is unavailable or does not make a confident healthy decision.
        is_uncertain = bool(
            "healthy" not in raw_class.lower()
            and (
                confidence < 65.0
                or predictions[predicted_index] < healthy_probability + 0.10
                or foliage_ratio < 0.20
            )
        )
        if is_uncertain:
            raw_class = class_names[best_healthy_index]
            confidence = healthy_probability * 100

    is_healthy = "healthy" in raw_class.lower() and not is_uncertain

    if raw_class == "Field_healthy":
        plant = "Plant"
    elif "pepper" in raw_class.lower():
        plant = "Pepper (Bell)"
    elif "potato" in raw_class.lower():
        plant = "Potato"
    elif "tomato" in raw_class.lower():
        plant = "Tomato"
    else:
        plant = "Plant"

    if is_uncertain:
        disease_display = "No confident disease detected"
        status = "Needs review"
    elif is_healthy:
        disease_display = "None (Healthy Plant)"
        status = "Healthy"
    else:
        clean = raw_class.replace("___", " ").replace("__", " ").replace("_", " ")
        disease_display = " ".join(clean.split())
        status = "Diseased"

    return {
        "is_valid": True,
        "is_healthy": is_healthy,
        "status": status,
        "needs_review": is_uncertain,
        "plant": plant,
        "disease": disease_display,
        "raw_class": raw_class,
        "confidence": round(confidence, 2)
    }
