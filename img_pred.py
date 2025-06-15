

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("/home/aniscorp/image_prediction/best_model.h5")

# Preprocess image
def preprocess_image(image):
    image = image.resize((32, 32))
    image = np.array(image) / 255.0  # Normalize to 0–1
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

with open('/home/aniscorp/Desktop/classe_name.txt','r') as name :
  class_names = [line.strip() for line in name]

# CIFAR-100 labels
# CLASSES = [...]  # Replace with actual CIFAR-100 class names

# UI
st.title("CIFAR-10 Image Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    model = load_model()
    input_data = preprocess_image(image)
    predictions = model.predict(input_data)
    predicted_class = class_names[np.argmax(predictions)]

    st.write(f"### Predicted Class: **{predicted_class}**")
