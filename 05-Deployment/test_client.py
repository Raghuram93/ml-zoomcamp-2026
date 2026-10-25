import requests

url = 'http://127.0.0.1:9696/predict'

# Standard test customer payload for Homework 5
customer = {
    "lead_source": "Neutral",
    "number_of_courses_viewed": 3,
    "annual_income": 0.0
}

try:
    response = requests.post(url, json=customer)
    result = response.json()
    
    print("Full API Response:", result)
    if 'probability' in result:
        print(f"Predicted Probability: {result['probability']:.3f}")
except requests.exceptions.ConnectionError:
    print("Error: Could not connect to the server at http://127.0.0.1:9696. Ensure Flask or Docker is running!")
