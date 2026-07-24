"""Train a field-photo model from the PlantDoc dataset.

Run from the repository root:
    venv/bin/python training/train_field_model.py
"""
import json
from pathlib import Path

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "dataset" / "PlantDoc" / "train"
MODEL_PATH = ROOT / "backend" / "models" / "plant_disease_field.keras"
CLASS_NAMES_PATH = ROOT / "backend" / "models" / "plant_disease_field_classes.json"
SOURCE_MODEL_PATH = ROOT / "backend" / "models" / "plant_disease_model.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE


def load_dataset(subset):
    return tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        subset=subset,
        shuffle=subset == "training",
        seed=42,
    )


train_ds = load_dataset("training")
test_ds = load_dataset("validation")
class_names = train_ds.class_names

CLASS_NAMES_PATH.write_text(json.dumps(class_names, indent=2), encoding="utf-8")

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.25),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.15),
], name="field_augmentation")

# Reuse the already verified MobileNet feature extractor from the current
# classifier. This avoids silently skipping ImageNet weights when loading the
# legacy local H5 checkpoint.
source_model = tf.keras.models.load_model(SOURCE_MODEL_PATH)
base_model = source_model.get_layer("mobilenetv2_1.00_224")
base_model.trainable = False

inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
x = augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.4)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.3),
    tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(train_ds.prefetch(AUTOTUNE), validation_data=test_ds.prefetch(AUTOTUNE), epochs=12, callbacks=callbacks)

# Fine-tune only the top MobileNet layers at a low learning rate.
base_model.trainable = True
for layer in base_model.layers[:-35]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(train_ds.prefetch(AUTOTUNE), validation_data=test_ds.prefetch(AUTOTUNE), epochs=12, callbacks=callbacks)

print(f"Saved field-photo model to {MODEL_PATH}")
