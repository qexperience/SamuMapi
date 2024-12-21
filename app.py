import os
from flask import Flask, request, jsonify, make_response
from tensorflow.keras.models import load_model
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
try:
    model = load_model("fixed_model.keras")  # Ensure the model file is named correctly
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

@app.after_request
def add_cors_headers(response):
    """
    Add CORS headers to every response to allow cross-origin requests.
    """
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

@app.route("/")
def home():
    """
    Home route to confirm the API is running.
    """
    return jsonify({"message": "API is live. Use the /predict endpoint for predictions."})

@app.route("/predict", methods=["POST"])
def predict_ph():
    """
    Predict the pH value based on RGB input provided in the POST request.
    """
    try:
        # Parse input JSON
        data = request.json
        if "r" not in data or "g" not in data or "b" not in data:
            return jsonify({"error": "Please provide 'r', 'g', and 'b' values."}), 400
        
        r, g, b = data["r"], data["g"], data["b"]
        
        # Validate input values
        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
            return jsonify({"error": "RGB values must be between 0 and 255."}), 400
        
        # Format input for the model
        input_rgb = np.array([[r, g, b]])
        
        # Predict pH value
        predicted_ph = model.predict(input_rgb)[0][0]
        predicted_ph = max(0, min(14, predicted_ph))  # Clamp the pH value to the range [0, 14]
        
        # Return the prediction
        return jsonify({"predicted_ph": round(predicted_ph, 4)})
    
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    # Get the port from the environment variable or use the default port 5000
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
