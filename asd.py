from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pandas as pd
from joblib import load
from category_encoders import BinaryEncoder


model = load("decision_tree_model.joblib")
x = pd.read_csv("dataset.csv")

categorical_features = [
    'Age', 'Tumor Size (cm)', 'Cost of Treatment (USD)', 
    'Economic Burden (Lost Workdays per Year)', 'Country', 'Gender', 'Tobacco Use', 'Alcohol Consumption',
    'HPV Infection', 'Betel Quid Use', 'Chronic Sun Exposure', 'Poor Oral Hygiene', 'Diet (Fruits & Vegetables Intake)',
    'Family History of Cancer', 'Compromised Immune System', 'Oral Lesions', 'Unexplained Bleeding', 
    'Difficulty Swallowing', 'White or Red Patches in Mouth', 'Treatment Type', 'Early Diagnosis'
]

encoder = BinaryEncoder()
encoder.fit_transform(x[categorical_features])

api = Flask(__name__)
CORS(api)

@api.route('/api/hfp_prediction', methods=['POST'])
def predict_heart_failure():
    data = request.json['inputs']
    input_df = pd.DataFrame(data)
    
    # Encoding categorical features
    input_encoded = encoder.transform(input_df[categorical_features])
    
    # Dropping categorical features from the original input DataFrame
    input_df = input_df.drop(categorical_features, axis=1)
    
    # Adding 'ID' as a column in the DataFrame
    input_df['ID'] = 0  # You can change this as needed (e.g., unique ID per input)

    # Resetting the index
    input_encoded = input_encoded.reset_index(drop=True)
    input_df = input_df.reset_index(drop=True)
    
    # Concatenating the ID and encoded features
    final_input = pd.concat([input_df, input_encoded], axis=1)

    # Making the prediction
    prediction = model.predict_proba(final_input)
    class_labels = model.classes_

    response = []
    for prob in prediction:
        prob_dict = {}
        for k, v in zip(class_labels, prob):
            prob_dict[str(k)] = round(float(v) * 100, 2)
        response.append(prob_dict)

    return jsonify({'prediction': response})

if __name__ == "__main__":
    api.run(debug=True, host='0.0.0.0', port=5000)
