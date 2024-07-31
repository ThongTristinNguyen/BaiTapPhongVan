from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import cv2
import joblib

app = Flask(__name__)

# Load the pre-trained model weights
model = joblib.load("path_to_your_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Convert image to grayscale and resize to 28x28
        image = Image.open(file).convert('L')
        image = image.resize((28, 28))
        image = np.array(image).astype(float) / 255.0
        image = image.reshape(1, -1)
        
        # Predict using the model
        prediction = model.predict(image)
        return jsonify({"prediction": int(prediction[0])}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
