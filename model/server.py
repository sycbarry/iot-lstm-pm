from flask import Flask, request, jsonify
import tensorflow as tf
import json
from helpers import window, model_forecast

app = Flask(__name__)
PORT = 5001

model = tf.keras.models.load_model('final.h5')  


WINDOW_SIZE = 30 
BATCH_SIZE = 32
SHUFFLE = 1000

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = data['reading'] 
    if features is not None: 
        extracted_features = json.loads(features)
        print("windowing data...")
        windowed_data = window(extracted_features, window_size=WINDOW_SIZE, batch_size=BATCH_SIZE, shuffle_buffer=SHUFFLE)
        print(windowed_data)
        print("forecasting data...")
        prediction = model_forecast(model, windowed_data, window_size=WINDOW_SIZE, batch_size=BATCH_SIZE)
        return jsonify({'prediction': prediction.tolist()})

@app.route('/health', methods=['GET'])
def health(): 
    return jsonify({'status': 200})

if __name__ == '__main__':
    print(f"server running on port {PORT}")
    app.run(host='0.0.0.0', port=PORT)
