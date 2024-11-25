import pickle
from flask import Flask, request, jsonify
import xgboost as xgb

# Load the model and DictVectorizer
model_file = 'xgboost_model0.1_6_0.1_0.8_0.8_15.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('shelter_days_prediction')

@app.route('/predict', methods=['POST'])

def predict():
    # Get the input data as JSON
    input_data = request.get_json()
    
    # Transform the input data using DictVectorizer
    X = dv.transform([input_data])
    features = list(dv.get_feature_names_out())
    
    # Create a DMatrix for prediction
    dmatrix = xgb.DMatrix(X, feature_names=features)
    
    # Predict days in shelter
    y_pred = model.predict(dmatrix)[0]
    
    result = {
        'predicted_days_in_shelter': float(y_pred)
    }
    
    # Return the result as JSON
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
