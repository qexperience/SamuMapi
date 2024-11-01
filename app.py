import os
import json
import numpy as np
from firebase_admin import credentials, firestore, initialize_app
from flask import Flask, request, render_template, jsonify
from PIL import Image, ImageEnhance

app = Flask(__name__)

# Load Firebase credentials from environment variable
firebase_credentials = json.loads(os.getenv("FIREBASE_SERVICE_ACCOUNT"))
cred = credentials.Certificate(firebase_credentials)
initialize_app(cred)
db = firestore.client()

# Function to find the closest pH value based on RGB input
def get_ph_value_from_rgb(rgb_color):
    # Query the Firestore collection
    ph_colors = db.collection('ph_colors').stream()
    closest_ph = None
    min_distance = float('inf')
    for doc in ph_colors:
        data = doc.to_dict()
        r, g, b = data['rgb']
        distance = np.linalg.norm(np.array([r, g, b]) - np.array(rgb_color))
        if distance < min_distance:
            min_distance = distance
            closest_ph = data['ph']
    return closest_ph

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['file']
    saturation = float(request.form.get('saturation', 1.0))
    brightness = float(request.form.get('brightness', 1.0))
    selected_x = int(request.form.get('x', 0))
    selected_y = int(request.form.get('y', 0))

    # Open and adjust image
    image = Image.open(file)
    enhancer_saturation = ImageEnhance.Color(image)
    image = enhancer_saturation.enhance(saturation)
    enhancer_brightness = ImageEnhance.Brightness(image)
    image = enhancer_brightness.enhance(brightness)
    
    # Extract RGB at the specified location
    img_array = np.array(image)
    selected_rgb = img_array[selected_y, selected_x]

    # Get pH value from Firestore
    ph_value = get_ph_value_from_rgb(selected_rgb)
    return jsonify({"ph_value": ph_value, "rgb": selected_rgb.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
