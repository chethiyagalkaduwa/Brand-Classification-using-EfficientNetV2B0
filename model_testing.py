import pandas as pd
import ast

# === Parse the labels.txt file ===
label_path = "Test_Labels.txt"
image_dir = "Test_Dataset" 

data = []
with open(label_path, "r") as file:
    for line in file:
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        filename = parts[0].strip('"')
        try:
            label = ast.literal_eval(parts[1])[0].strip().lower()
            data.append((filename, label))
        except Exception as e:
            print(f"Skipping line due to error: {line}\n{e}")

df = pd.DataFrame(data, columns=["filename", "class"])

# === Load model and prepare data generator ===
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("efficientnetv2_finetuned.h5")

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=df,
    directory=image_dir,
    x_col="filename",
    y_col="class",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# === Evaluate model on the test set ===
loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy:.4f}")

# === Predict and save detailed results ===
preds = model.predict(test_generator)
predicted_classes = np.argmax(preds, axis=1)
true_classes = test_generator.classes

class_indices = test_generator.class_indices
inv_class_indices = {v: k for k, v in class_indices.items()}

predicted_labels = [inv_class_indices[i] for i in predicted_classes]
true_labels = [inv_class_indices[i] for i in true_classes]

# Save results to CSV
results_df = pd.DataFrame({
    "Filename": test_generator.filenames,
    "True Label": true_labels,
    "Predicted Label": predicted_labels,
    "Confidence": np.max(preds, axis=1)
})
results_df.to_csv("Kaggle results/test_predictions.csv", index=False)
print("Saved predictions to test_predictions.csv")
