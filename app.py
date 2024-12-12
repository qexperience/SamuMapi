from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model

# Initialize Flask App
app = Flask(__name__)

# Load the trained model
model = load_model("final_ph_model.keras")

# Define a function to predict pH from RGB
def predict_ph_from_rgb(r, g, b):
    """
    Predict the pH value given an RGB triplet using the loaded model.
    Parameters:
        r (int): Red component (0-255)
        g (int): Green component (0-255)
        b (int): Blue component (0-255)
    Returns:
        float: Predicted pH value (clamped to range [0, 14]) with 4 decimal places
    """
    input_rgb = np.array([[r, g, b]]) / 255.0  # Normalize RGB values to [0, 1]
    predicted_ph = model.predict(input_rgb)[0][0] * 14  # Scale back to pH range [0, 14]
    return round(max(0, min(14, predicted_ph)), 4)  # Clamp and round to 4 decimal places

# Define routes
@app.route("/")
def home():
    return render_template("index.html")  # Serves the HTML front-end

@app.route("/predict", methods=["POST"])
def predict():
    # Get data from the POST request
    data = request.get_json()
    try:
        r = int(data["r"])
        g = int(data["g"])
        b = int(data["b"])
        # Validate the RGB values
        if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
            predicted_ph = predict_ph_from_rgb(r, g, b)
            return jsonify({"predicted_ph": predicted_ph})
        else:
            return jsonify({"error": "RGB values must be in the range 0-255."}), 400
    except (ValueError, KeyError):
        return jsonify({"error": "Invalid input. Please provide 'r', 'g', and 'b' as integers."}), 400

if __name__ == "__main__":
    app.run(debug=True)
