from pathlib import Path

import tensorflow as tf

# Paths
DATASET_PATH = Path("KneeXrayMini")
TRAIN_PATH = DATASET_PATH / "train"
VAL_PATH = DATASET_PATH / "val"
TEST_PATH = DATASET_PATH / "test"
MODEL_PATH = Path("models")

# Settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10

# Load datasets
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

autotune = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(autotune)
validation_dataset = validation_dataset.prefetch(autotune)
test_dataset = test_dataset.prefetch(autotune)

# Load previously trained model
model = tf.keras.models.load_model(
    MODEL_PATH / "best_densenet_arthritis.keras"
)

# Find DenseNet base model
base_model = None

for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        base_model = layer
        break

if base_model is None:
    raise ValueError("DenseNet base model was not found.")

# Unfreeze last 50 layers
base_model.trainable = True

for layer in base_model.layers[:-50]:
    layer.trainable = False

# Keep BatchNormalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# Use a very small learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=0.0000001,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH / "best_finetuned_densenet.keras",
        monitor="val_accuracy",
        save_best_only=True,
    ),
]

# Fine-tune
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# Test
test_loss, test_accuracy = model.evaluate(test_dataset)

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# Save final model
model.save(MODEL_PATH / "finetuned_arthritis_densenet.keras")

print("Fine-tuned model saved successfully.")