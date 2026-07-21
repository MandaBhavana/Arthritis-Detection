from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models


DATASET_PATH = Path("KneeXrayMini")
TRAIN_PATH = DATASET_PATH / "train"
VAL_PATH = DATASET_PATH / "val"
TEST_PATH = DATASET_PATH / "test"
MODEL_PATH = Path("models")

MODEL_PATH.mkdir(exist_ok=True)


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 5


train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)


validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)


test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
)

print("Classes:", train_dataset.class_names)


autotune = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=autotune)
validation_dataset = validation_dataset.prefetch(buffer_size=autotune)
test_dataset = test_dataset.prefetch(buffer_size=autotune)


data_augmentation = models.Sequential(
    [
        layers.RandomRotation(0.03),
        layers.RandomZoom(0.05),
        layers.RandomContrast(0.1),
    ]
)


model = models.Sequential(
    [
        layers.Input(shape=(224, 224, 3)),
        data_augmentation,
        layers.Rescaling(1.0 / 255),

        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH / "best_arthritis_cnn.keras",
        monitor="val_accuracy",
        save_best_only=True,
    ),
]


model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
)


test_loss, test_accuracy = model.evaluate(test_dataset)

print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)


model.save(MODEL_PATH / "arthritis_cnn.keras")

print("Model saved successfully.")