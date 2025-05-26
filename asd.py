from flask import Flask, request, jsonify, abort
from flask_cors import CORS, cross_origin
import pandas as pd
from joblib import load
import logging
from category_encoders import BinaryEncoder

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
api = Flask(__name__)
CORS(api)  # Enable CORS globally

# Global variables
encoder = None
model = None
categorical_features = [
    'Age', 'Tumor Size (cm)', 'Cost of Treatment (USD)', 
    'Economic Burden (Lost Workdays per Year)', 'Country', 'Gender', 'Tobacco Use', 'Alcohol Consumption',
    'HPV Infection', 'Betel Quid Use', 'Chronic Sun Exposure', 'Poor Oral Hygiene', 'Diet (Fruits & Vegetables Intake)',
    'Family History of Cancer', 'Compromised Immune System', 'Oral Lesions', 'Unexplained Bleeding', 
    'Difficulty Swallowing', 'White or Red Patches in Mouth', 'Treatment Type', 'Early Diagnosis'
]

# Function to load the model and encoder lazily
def load_model_and_encoder():
    global model, encoder
    if model is None:
        logging.info("Loading model...")
        model = load("decision_tree_model.joblib")
    if encoder is None:
        logging.info("Loading encoder...")
        x = pd.read_csv("dataset.csv", dtype={'Age': 'int32', 'Tumor Size (cm)': 'float32', 'Cost of Treatment (USD)': 'float32'})
        encoder = BinaryEncoder()
        encoder.fit(x[categorical_features])

# Endpoint for heart failure prediction
@api.route('/predict', methods=['POST', 'OPTIONS'])
@cross_origin(origins='*')
def predict_heart_failure():
    # Limit the request size to 10MB
    max_data_size = 10 * 1024 * 1024  # Max data size 10MB
    if request.content_length > max_data_size:
        abort(413, description="Payload too large")

    if request.method == 'OPTIONS':
        # Handle preflight request (for CORS)
        return '', 200

    try:
        # Load the model and encoder if they are not already loaded
        load_model_and_encoder()

        data = request.json['inputs']
        logging.info(f"Data received: {data}")

        # Convert input data to a DataFrame
        input_df = pd.DataFrame(data)

        # Encode categorical features
        input_encoded = encoder.transform(input_df[categorical_features])

        # Drop categorical features from the original input DataFrame
        input_df = input_df.drop(categorical_features, axis=1)

        # Add 'ID' column (you can customize this as needed)
        input_df['ID'] = 0  # Can be adjusted based on your input format

        # Reset the index to ensure proper concatenation
        input_encoded = input_encoded.reset_index(drop=True)
        input_df = input_df.reset_index(drop=True)

        # Concatenate encoded features and input_df (including 'ID')
        final_input = pd.concat([input_df, input_encoded], axis=1)

        # Make the prediction using the trained model
        prediction_probs = model.predict_proba(final_input)[0]

        # Log the probabilities
        logging.info(f"Prediction probabilities: {prediction_probs}")

        # Construct a response with just the raw probabilities
        response = {"prediction_probabilities": prediction_probs.tolist()}

        return jsonify(response)

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Run the app with more robust configurations in production
    api.run(debug=True, host='0.0.0.0', port=5000)
