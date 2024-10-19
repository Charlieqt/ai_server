from flask import Flask, request, jsonify
import openai
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import requests

app = Flask(__name__)

# API klucze i konfiguracje
openai.api_key = os.getenv('OPENAI_API_KEY')
polygon_api_key = os.getenv('POLYGON_API_KEY')
model_path = os.getenv('MODEL_PATH', 'atfnet_model.h5')

# Załaduj model ATFNet
model = load_model(model_path)

# GPT-4 analiza
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    query = data.get('query', 'Analyze market conditions for EUR/USD')
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=query,
        max_tokens=200
    )
    return jsonify({'response': response.choices[0].text})

# Prognozowanie na podstawie ATFNet
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    ohlc_data = data.get('ohlc', [])
    prediction = model.predict([ohlc_data])
    return jsonify({'prediction': prediction.tolist()})

# Pobieranie danych OHLC z Polygon.io
@app.route('/fetch_ohlc', methods=['POST'])
def fetch_ohlc():
    data = request.json
    date = data.get('date')
    pair = 'C:EURUSD'
    url = f'https://api.polygon.io/v2/aggs/grouped/locale/global/market/fx/2023-01-09?adjusted=true&apiKey=zr0XDpp2AwbzMnEC8LJS_PMANWxrVLRO'
    response = requests.get(url)
    if response.status_code == 200:
        json_data = response.json()
        return jsonify(json_data['results'][0])
    else:
        return jsonify({'error': 'Failed to fetch data from Polygon.io'}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 4000)) 
    app.run(host='0.0.0.0', port=port)
