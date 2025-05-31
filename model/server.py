from flask import Flask, request, jsonify
import tensorflow as tf
import json
import os
from helpers import window, model_forecast
import requests

app = Flask(__name__)
PORT = 5001

model = tf.keras.models.load_model('final.h5')  

PREDICTION_ENDPOINT = os.getenv("PREDICTION_ENDPOINT", "https://webhook.site/b425fbac-0ca2-4ebf-a205-209f40193be9")

WINDOW_SIZE = 30 
BATCH_SIZE = 32
SHUFFLE = 1000


def post_prediction(data: list) -> None: 
    if data is None: 
        print("no prediction data specified")
        return None
    payload = json.dumps({"prediction": data})
    response = requests.post(PREDICTION_ENDPOINT, data=payload)
    response.raise_for_status()
    return response.status_code

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = data['reading'] 
    if features is not None: 
        extracted_features = json.loads(features)
        print("windowing data...")
        # window the data the same way that we windowed the data in our model (using the same functions)
        windowed_data = window(extracted_features, window_size=WINDOW_SIZE, batch_size=BATCH_SIZE, shuffle_buffer=SHUFFLE)
        print(windowed_data)
        print("forecasting data...")
        # perform our forecasts using the same function we used in our notebook
        prediction = model_forecast(model, windowed_data, window_size=WINDOW_SIZE, batch_size=BATCH_SIZE)
        status = post_prediction(prediction.tolist())
        return jsonify({"status": status})


@app.route('/health', methods=['GET'])
def health(): 
    return jsonify({'status': 200})

if __name__ == '__main__':
    print(f"server running on port {PORT}")
    app.run(host='0.0.0.0', port=PORT)
