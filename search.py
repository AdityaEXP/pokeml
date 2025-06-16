from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import json

# Load model and class map
model = load_model("model/pokemon_classifier.h5")

with open("model/class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse to get index → name
index_to_class = {v: k for k, v in class_indices.items()}


# Image from URL
url = input("Enter the image URL: ")
response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert("RGB").resize((224, 224))

# Preprocess
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
predicted_index = np.argmax(prediction)
predicted_class = index_to_class[int(predicted_index)]

print("🎯 Predicted Pokémon:", predicted_class)
