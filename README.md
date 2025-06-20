# 🧠 Who is that PokeML? - Image classifier

> A deep learning model that classifies Pokémon images using transfers learning with Mobilenetv2 + Keras. Using its own custom-related model predicts more than 150 Pokémon from files or URLs.


## 📦 Project Overview

The project trains an image classifier that can identify 150+ Pokémon characters from images. It uses **Mobilenetv2** as a pretrand base and adds a custom head for Pokémon classification.

Trained using **tensorflow + keras**, **with data growth**, **fine-tuning**, and **Early stopping** for best performance.

**Test Dataset to train model to identify pokemon: https://www.kaggle.com/datasets/lantian773030/pokemonclassification**


## 🚀 features

- ✅ Transfer Learning (Mobilentv2 + Custom Head)
- 🎨 data growth (rotation, zoom, flips)
- 🧠 fine tuning for high accuracy
- 🔁 Erustoping + Model Czechpointing
- Predic from 🌐 **image url or local file**
- 📁 saved model + class label
- 🧪 60%+ verification accuracy on 150 classes


## 📁Directory Structure
  ```
pokemon-classifier/
├── data/                    # Pokémon images (organized by class folders)
│   ├── pikachu/
│   ├── charmander/
│   └── ...
├── model/                   # Saved model and class index map
│   ├── pokemon_classifier.h5
│   └── class_indices.json
├── main.py                  # Training script (uses transfer learning with MobileNetV2)
├── search.py                # Prediction script (predicts from image URL or file)
├── README.md                # Project documentation (this file)
  ```


## 🧪 Model Info

| Property          | Value           |
|------------------|-----------------|
| Base Model       | MobileNetV2     |
| Classes          | ~150 Pokémon    |
| Accuracy         | ~74% train, ~60% val |
| Epochs           | 30 (with EarlyStopping) |
| Save Format      | `.h5` (can convert to `.keras`) |

## 🏁 How to Run

### 🔧 1. Train the Model

Make sure your Pokémon images are inside `data/` and organized like:
```
data/
├── bulbasaur/
├── charmander/
├── pikachu/
....
```


Then run:
```python main.py```

After training:
The model is saved to model/pokemon_classifier.h5
Class mappings saved to model/class_indices.json

### 🔍 2. Predict Pokémon from Image URL
```python search.py```


## 🙌 Credits
🧩 Pretrained model: MobileNetV2
🎨 Dataset: PokeAPI sprites
🛠 Frameworks: TensorFlow, Keras, Pillow, NumPy


