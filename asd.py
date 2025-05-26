from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pandas as pd
from joblib import load
from category_encoders import BinaryEncoder

# Load the trained model
model = load("decision_tree_model.joblib")

# Load dataset (ensure this is correctly preprocessed, if needed)
x = pd.read_csv("dataset.csv")

# Define categorical features for encoding
categorical_features = [
    'Age', 'Tumor Size (cm)', 'Cost of Treatment (USD)', 
    'Economic Burden (Lost Workdays per Year)', 'Country', 'Gender', 'Tobacco Use', 'Alcohol Consumption',
    'HPV Infection', 'Betel Quid Use', 'Chronic Sun Exposure', 'Poor Oral Hygiene', 'Diet (Fruits & Vegetables Intake)',
    'Family History of Cancer', 'Compromised Immune System', 'Oral Lesions', 'Unexplained Bleeding', 
    'Difficulty Swallowing', 'White or Red Patches in Mouth', 'Treatment Type', 'Early Diagnosis'
]

# Initialize encoder and fit it on the categorical features
encoder = BinaryEncoder()
encoder.fit(x[categorical_features])

# Initialize Flask app
api = Flask(__name__)
CORS(api)  # Enable CORS globally

@api.route('/hfp_prediction', methods=['POST', 'OPTIONS'])
@cross_origin(origins='*')  # Allow requests from all origins
def predict_heart_failure():
    if request.method == 'OPTIONS':
        # Handle preflight request (for CORS)
        return '', 200

    try:
        data = request.json['inputs']
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

        # Construct a response with just the raw probabilities
        response = {"prediction_probabilities": prediction_probs.tolist()}

        return jsonify(response)

    except Exception as e:
        # Return error response in case of any failure
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    api.run(debug=True, host='0.0.0.0', port=5000)
