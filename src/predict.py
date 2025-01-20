import tensorflow as tf
import numpy as np
from PIL import Image

# Suppress TensorFlow logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the saved model
model = tf.keras.models.load_model("./src/model.keras")

# Load and preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path).convert("L").resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # Reshape to (1, 28, 28, 1)
    return img_array

image_path = "./data/image.png"
image_data = preprocess_image(image_path)

# Perform prediction
predictions = model.predict(image_data)
predicted_class = np.argmax(predictions)
print(f"Predicted Class: {predicted_class}")
