from flask import Flask, request, jsonify
import tensorflow as tf
import json
import os
from helpers import window, model_forecast
import requests
import numpy as np

app = Flask(__name__)
PORT = 5001

model = tf.keras.models.load_model('final.h5')  

PREDICTION_ENDPOINT = os.getenv("PREDICTION_ENDPOINT", "https://webhook.site/0bc39358-0532-4bb5-912e-9d41856a6afd")

WINDOW_SIZE = 30 
BATCH_SIZE = 32
SHUFFLE = 1000


def post_prediction(data: list) -> None: 
    if data is None: 
        print("no prediction data specified")
        return None
    response = requests.post(PREDICTION_ENDPOINT, json=data)
    response.raise_for_status()
    return response.status_code

def calculate_prediction_anomalies(input, prediction: list) -> dict: 
    """Here we're taking the raw input, the prediction 
    and making calculations to determine the anomalies in our input data"""

    # we're using a z-score approach here to calculate the error.
    errors = np.abs(input.flatten()[:-WINDOW_SIZE + 1] - prediction)
    threshold = np.mean(errors) + 3 * np.std(errors)

    # check for the % of anomalies that are above our z-score threshold
    anomalies = errors > threshold

    # convert those to a % valu
    anomaly_percentage = np.mean(anomalies) * 100


    return { 
        "anomalies": anomalies.tolist().count(True),
        "anomaly_percentage": float(anomaly_percentage) 
        }


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['reading'])
    if features is not None: 
        prediction = model_forecast(model, features, window_size=WINDOW_SIZE, batch_size=BATCH_SIZE)
        prediction = prediction.flatten()
        payload = calculate_prediction_anomalies(features, prediction)
        status = post_prediction(payload)
        return jsonify({"status": status})


@app.route('/health', methods=['GET', 'POST'])
def health(): 
    return jsonify({'status': 200})

if __name__ == '__main__':
    print(f"server running on port {PORT}")
    app.run(host='0.0.0.0', port=PORT)
