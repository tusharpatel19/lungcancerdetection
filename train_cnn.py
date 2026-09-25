from pathlib import Path
import csv
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf

DATA_DIR = Path("data")
SPLIT_ROOT = DATA_DIR / "Data"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
MODEL_DIR = Path("model")
PUBLIC_STATIC_DIR = Path("public") / "static"

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


def save_accuracy_plot(history, output_path):
    train_acc = history.get("accuracy", [])
    val_acc = history.get("val_accuracy", [])
    epochs = len(train_acc)
    if epochs == 0:
        return

    width = 760
    height = 420
    margin = 64
    chart_width = width - margin * 2
    chart_height = height - margin * 2

    image = Image.new("RGB", (width, height), "#f7fafc")
    draw = ImageDraw.Draw(image)

    try:
        title_font = ImageFont.truetype("arial.ttf", 24)
        axis_font = ImageFont.truetype("arial.ttf", 14)
        label_font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        title_font = axis_font = label_font = ImageFont.load_default()

    draw.text((margin, 20), "Training Accuracy", fill="#182033", font=title_font)
    draw.text((margin, 52), "Epoch vs accuracy for training and validation.", fill="#607089", font=label_font)

    origin_x = margin
    origin_y = height - margin
    draw.line((origin_x, margin + 20, origin_x, origin_y), fill="#9aa7c7", width=2)
    draw.line((origin_x, origin_y, origin_x + chart_width, origin_y), fill="#9aa7c7", width=2)

    for i in range(5):
        y = origin_y - (chart_height / 4) * i
        value = i * 0.25
        draw.line((origin_x - 6, y, origin_x, y), fill="#d8e2ea", width=1)
        label = f"{value:.2f}"
        bbox = draw.textbbox((0, 0), label, font=axis_font)
        draw.text((origin_x - bbox[2] - 10, y - bbox[3] / 2), label, fill="#607089", font=axis_font)

    if epochs > 1:
        step_x = chart_width / (epochs - 1)
    else:
        step_x = 0

    def chart_point(index, accuracy):
        x = origin_x + step_x * index
        y = origin_y - accuracy * chart_height
        return x, y

    if val_acc:
        points = [chart_point(i, min(max(val_acc[i], 0.0), 1.0)) for i in range(min(epochs, len(val_acc)))]
        for i in range(1, len(points)):
            draw.line((points[i - 1] + points[i]), fill="#f59e0b", width=3)
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#f59e0b")

    points = [chart_point(i, min(max(train_acc[i], 0.0), 1.0)) for i in range(epochs)]
    for i in range(1, len(points)):
        draw.line((points[i - 1] + points[i]), fill="#16a34a", width=3)
    for x, y in points:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#16a34a")

    for i in range(epochs):
        x = origin_x + step_x * i
        if epochs <= 10 or i % 2 == 0 or i == epochs - 1:
            label = str(i + 1)
            bbox = draw.textbbox((0, 0), label, font=axis_font)
            draw.text((x - bbox[2] / 2, origin_y + 10), label, fill="#607089", font=axis_font)

    draw.text((origin_x + chart_width - 100, margin + 18), "Train", fill="#16a34a", font=axis_font)
    draw.text((origin_x + chart_width - 100, margin + 38), "Val", fill="#f59e0b", font=axis_font)

    image.save(output_path)

MODEL_DIR.mkdir(exist_ok=True)
PUBLIC_STATIC_DIR.mkdir(parents=True, exist_ok=True)
model_path = MODEL_DIR / "model.h5"
keras_path = MODEL_DIR / "model.keras"
labels_path = MODEL_DIR / "class_labels.txt"
model.save(model_path)
model.save(keras_path)
labels_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")

accuracy_plot_path = PUBLIC_STATIC_DIR / "accuracy_plot.png"
save_accuracy_plot(history.history, accuracy_plot_path)
print("Saved accuracy plot to", accuracy_plot_path)

print("Saved model to", model_path)
print("Saved model to", keras_path)
print("Saved class labels to", labels_path)
print("Class labels:", class_names)


def build_confusion_matrix(y_true, y_pred, total_classes):
    matrix = np.zeros((total_classes, total_classes), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        matrix[int(actual), int(predicted)] += 1
    return matrix


def save_confusion_matrix_csv(matrix, labels, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Actual \\ Predicted", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])


def save_confusion_matrix_image(matrix, labels, output_path):
    def shorten_label(label):
        text = label.replace("_", " ").replace(".", " ")
        lower = text.lower()
        if "normal" in lower:
            return "normal"
        if "adenocarcinoma" in lower:
            return "adenocarcinoma"
        if "large" in lower and "cell" in lower:
            return "large cell"
        if "squamous" in lower and "cell" in lower:
            return "squamous cell"
        parts = [part for part in text.split() if part]
        return " ".join(parts[:2]) if len(parts) > 1 else text

    short_labels = [shorten_label(label) for label in labels]
    cell = 82
    left = 170
    top = 170
    width = left + cell * len(labels) + 40
    height = top + cell * len(labels) + 95
    max_value = max(int(matrix.max()), 1)

    image = Image.new("RGB", (width, height), "#f7fafc")
    draw = ImageDraw.Draw(image)

    try:
        title_font = ImageFont.truetype("arial.ttf", 24)
        header_font = ImageFont.truetype("arial.ttf", 15)
        value_font = ImageFont.truetype("arial.ttf", 21)
        small_font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        title_font = header_font = value_font = small_font = ImageFont.load_default()

    draw.text((28, 24), "Confusion Matrix", fill="#182033", font=title_font)
    draw.text((28, 58), "Rows show actual class, columns show predicted class.", fill="#607089", font=small_font)
    draw.text((left + 20, top - 62), "Predicted", fill="#26364d", font=header_font)
    draw.text((34, top + 20), "Actual", fill="#26364d", font=header_font)

    for index, short_label in enumerate(short_labels):
        x = left + index * cell
        draw.text((x + 8, top - 44), short_label, fill="#26364d", font=small_font)
        draw.text((22, top + index * cell + 30), short_label, fill="#26364d", font=small_font)

    for row in range(len(labels)):
        for col in range(len(labels)):
            value = int(matrix[row, col])
            strength = value / max_value
            red = 229 - math.floor(196 * strength)
            green = 248 - math.floor(77 * strength)
            blue = 246 - math.floor(116 * strength)
            fill = (red, green, blue)
            x0 = left + col * cell
            y0 = top + row * cell
            x1 = x0 + cell - 5
            y1 = y0 + cell - 5
            draw.rounded_rectangle((x0, y0, x1, y1), radius=7, fill=fill, outline="#d7dee8")
            text = str(value)
            bbox = draw.textbbox((0, 0), text, font=value_font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            draw.text(
                (x0 + (cell - text_w) / 2 - 2, y0 + (cell - text_h) / 2 - 4),
                text,
                fill="#101828",
                font=value_font,
            )

    image.save(output_path)




if test_dir.exists():
    test_ds_raw = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_ds = test_ds_raw.map(lambda x, y: (normalization(x), y))
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

    y_true_batches = []
    y_pred_batches = []
    for batch_images, batch_labels in test_ds:
        batch_probs = model.predict(batch_images, verbose=0)
        if num_classes == 2:
            batch_pred = (batch_probs.reshape(-1) >= 0.5).astype(int)
        else:
            batch_pred = np.argmax(batch_probs, axis=1)
        y_true_batches.append(batch_labels.numpy())
        y_pred_batches.append(batch_pred)

    y_true = np.concatenate(y_true_batches)
    y_pred = np.concatenate(y_pred_batches)
    matrix = build_confusion_matrix(y_true, y_pred, num_classes)

    csv_path = MODEL_DIR / "confusion_matrix.csv"
    png_path = PUBLIC_STATIC_DIR / "confusion_matrix.png"
    save_confusion_matrix_csv(matrix, class_names, csv_path)
    save_confusion_matrix_image(matrix, class_names, png_path)
    print("Saved confusion matrix CSV to", csv_path)
    print("Saved confusion matrix image to", png_path)
else:
    print("No test folder found, so confusion matrix was not generated.")
