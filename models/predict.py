import tensorflow as tf
import numpy as np
import cv2
import os

# Load model
model = tf.keras.models.load_model("saved_model/apple_leaf_model.h5")

# Load image and preprocess
img_path = "test_images\sample_image_1.jpeg"  # Replace with your image
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)

# Predict
pred = model.predict(img)
class_index = np.argmax(pred)

# Assuming class names are in alphabetical order
class_names = os.listdir("datasets")
print("Predicted Class:", class_names[class_index])
