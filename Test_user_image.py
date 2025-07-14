from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import numpy as np

model = load_model("efficientnetv2_finetuned.h5") 

class_names = np.load("class_indices_effnet.npy", allow_pickle=True)

def preprocess_image(image_path, image_size=(224, 224)):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(image_size)
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image

image_path = "/kaggle/input/train-data/Final_Data/1501.jpg"  # Replace with user image path
image = preprocess_image(image_path)

preds = model.predict(image)
predicted_class = class_names[np.argmax(preds)]
confidence = np.max(preds)

print(f"Predicted class: {predicted_class} (confidence: {confidence:.2f})")