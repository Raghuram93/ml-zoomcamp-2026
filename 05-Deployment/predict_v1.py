import pickle
from flask import Flask, request, jsonify

# Load local model binary pipeline (pipeline_v1.bin)
MODEL_FILE = 'pipeline_v1.bin'

with open(MODEL_FILE, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('churn-v1')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    
    # Preprocess categorical features using DictVectorizer
    X = dv.transform([customer])
    
    # Predict probability of churn using LogisticRegression
    y_pred = model.predict_proba(X)[0, 1]
    result = {
        'probability': float(y_pred),
        'churn': bool(y_pred >= 0.5)
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
