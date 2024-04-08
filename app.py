import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

# Create flask app
app = Flask(__name__)

# Load the pickle model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract features from the request form
        bedrooms = float(request.form.get('bedrooms'))
        bathrooms = float(request.form.get('bathrooms'))
        toilets = float(request.form.get('toilets'))
        total_rooms = float(request.form.get('total_rooms'))

        # Make prediction using the loaded model
        features = np.array([[bedrooms, bathrooms, toilets, total_rooms]])
        prediction = model.predict(features)

        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
