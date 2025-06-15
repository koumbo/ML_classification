import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("/home/aniscorp/image_prediction/best_model.h5")  # adjust path if needed
    return model

# Preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((32, 32))  # CIFAR-10 image size
    image = np.array(image) / 255.0  # Normalize to 0-1
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# CIFAR-10 class labels
# CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
#            'dog', 'frog', 'horse', 'ship', 'truck']
with open('/home/aniscorp/Desktop/classe_name.txt','r') as name :
  class_names = [line.strip() for line in name]


# Streamlit UI
st.title("CIFAR-10 Image Classifier")
st.write("Upload an image, and the model will predict its class (CIFAR-10 classes only).")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Load model
    model = load_model()

    # Preprocess and predict
    input_data = preprocess_image(image)
    predictions = model.predict(input_data)
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = np.max(predictions) * 100  # Convert to percentage

    st.write(f"### Predicted Class: **{predicted_class}**")
    st.write(f"Confidence: **{confidence:.2f}%**")

    # Warn the user if confidence is low
    if confidence < 50:
        st.warning("⚠️ The model is not very confident about this prediction. "
                   "The uploaded image may not belong to CIFAR-10 classes.")
