import tensorflow as tf
from pathlib import Path

DATA_DIR = Path("data")
SPLIT_ROOT = DATA_DIR / "Data"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10

if not DATA_DIR.exists():
    raise SystemExit("Expected a data/ folder with class subfolders.")

use_pre_split = (SPLIT_ROOT / "train").exists() and (SPLIT_ROOT / "valid").exists()
test_dir = SPLIT_ROOT / "test"

if use_pre_split:
    train_dir = SPLIT_ROOT / "train"
    val_dir = SPLIT_ROOT / "valid"
else:
    train_dir = DATA_DIR
    val_dir = DATA_DIR

if use_pre_split:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
else:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

class_names = train_ds.class_names
num_classes = len(class_names)

normalization = tf.keras.layers.Rescaling(1.0 / 255)

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
    ]
)

train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))
train_ds = train_ds.map(lambda x, y: (normalization(x), y))
val_ds = val_ds.map(lambda x, y: (normalization(x), y))

model_layers = [
    tf.keras.layers.Input(shape=(*IMG_SIZE, 3)),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.4),
]

if num_classes == 2:
    model_layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
    loss = tf.keras.losses.BinaryCrossentropy()
else:
    model_layers.append(tf.keras.layers.Dense(num_classes, activation="softmax"))
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

model = tf.keras.Sequential(model_layers)

model.compile(
    optimizer="adam",
    loss=loss,
    metrics=["accuracy"],
)

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

model_dir = Path("model")
model_dir.mkdir(exist_ok=True)
model_path = model_dir / "model.h5"
keras_path = model_dir / "model.keras"
model.save(model_path)
model.save(keras_path)

print("Saved model to", model_path)
print("Saved model to", keras_path)
print("Class labels:", class_names)

if test_dir.exists():
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_ds = test_ds.map(lambda x, y: (normalization(x), y))
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")
