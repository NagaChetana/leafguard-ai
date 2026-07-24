"""Train a field-photo healthy-versus-diseased classifier.

This is deliberately separate from the fine-grained disease classifier. It
answers the first, safety-critical question reliably: does this real-world
leaf show disease symptoms at all?
"""
import random
from pathlib import Path

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "dataset" / "PlantDoc"
SOURCE_MODEL_PATH = ROOT / "backend" / "models" / "plant_disease_model.keras"
MODEL_PATH = ROOT / "backend" / "models" / "plant_health_field.keras"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

# PlantDoc's plain "<plant> leaf" folders are healthy; folders containing one
# of these terms contain a labelled disease or pest condition.
DISEASE_WORDS = (
    "blight", "spot", "mold", "mosaic", "virus", "scab", "rust",
    "mildew", "mite", "gray", "yellow", "bacterial", "early", "late",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def is_diseased(class_name: str) -> int:
    return int(any(word in class_name.lower() for word in DISEASE_WORDS))


def collect_examples(split: str):
    examples = []
    for folder in (DATASET_ROOT / split).iterdir():
        if not folder.is_dir():
            continue
        label = is_diseased(folder.name)
        examples.extend((str(path), label) for path in folder.iterdir() if path.suffix in IMAGE_SUFFIXES)
    return examples


def make_dataset(examples, training=False):
    paths = [path for path, _ in examples]
    labels = [label for _, label in examples]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(examples), seed=SEED, reshuffle_each_iteration=True)

    def load_image(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE)
        return image, tf.cast(label, tf.float32)

    return ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


train_examples = collect_examples("train")
test_examples = collect_examples("test")
random.Random(SEED).shuffle(train_examples)
cutoff = int(len(train_examples) * 0.85)
train_ds = make_dataset(train_examples[:cutoff], training=True)
validation_ds = make_dataset(train_examples[cutoff:])
test_ds = make_dataset(test_examples)
train_labels = [label for _, label in train_examples[:cutoff]]
class_counts = {label: train_labels.count(label) for label in (0, 1)}
class_weights = {
    label: len(train_labels) / (2 * count) for label, count in class_counts.items()
}

source_model = tf.keras.models.load_model(SOURCE_MODEL_PATH, compile=False)
base_model = source_model.get_layer("mobilenetv2_1.00_224")
base_model.trainable = False

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.18),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
], name="field_augmentation")

inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
x = augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.45)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="health_probability")(x)
model = tf.keras.Model(inputs, outputs)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.3),
]

metrics = [
    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
    tf.keras.metrics.AUC(name="auc"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
]
model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss="binary_crossentropy", metrics=metrics)
model.fit(train_ds, validation_data=validation_ds, epochs=14, callbacks=callbacks, class_weight=class_weights)

base_model.trainable = True
for layer in base_model.layers[:-35]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="binary_crossentropy", metrics=metrics)
model.fit(train_ds, validation_data=validation_ds, epochs=10, callbacks=callbacks, class_weight=class_weights)

results = model.evaluate(test_ds, return_dict=True)
print("Held-out PlantDoc results:", {name: round(float(value), 4) for name, value in results.items()})
