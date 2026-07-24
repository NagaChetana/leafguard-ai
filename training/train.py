import tensorflow as tf

# ----------------------------
# Configuration
# ----------------------------
DATASET_PATH = "dataset/archive/PlantVillage"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 15

# ----------------------------
# Load Dataset
# ----------------------------
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

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)

# ----------------------------
# Data Augmentation
# ----------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

# ----------------------------
# Base Model
# ----------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# ----------------------------
# Build Model
# ----------------------------
inputs = tf.keras.Input(shape=(224,224,3))

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

# ----------------------------
# Compile
# ----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ----------------------------
# Callbacks
# ----------------------------
callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        "backend/models/plant_disease_model.keras",
        save_best_only=True,
        monitor="val_accuracy"
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2
    )
]

# ----------------------------
# Stage 1 Training
# ----------------------------
history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=5,

    callbacks=callbacks
)

# ----------------------------
# Fine Tuning
# ----------------------------
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(

    optimizer=tf.keras.optimizers.Adam(1e-5),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)

history_fine = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=15,

    initial_epoch=history.epoch[-1] + 1,

    callbacks=callbacks
)

print("\nTraining Completed Successfully!")
