from flask import Blueprint, jsonify
import json

api_bp = Blueprint("api", __name__, url_prefix='/api')

@api_bp.route('/predictions', methods=['GET'])
def get_predictions():
    # Read the predictions from the JSON file
    with open('predictions.json', 'r') as f:
        predictions = json.load(f)
    return jsonify(predictions)

@api_bp.route('/prediction/<date>', methods=['GET'])
def get_prediction_by_date(date):
    try:
        # Read the predictions from the JSON file
        with open('predictions.json', 'r') as f:
            predictions = json.load(f)
        
        # Filter for the requested date
        prediction = next((item for item in predictions if item['date'] == date), None)
        
        if prediction:
            response = {
                'predicted_close': prediction['predicted_close'],
                'real_close': prediction['real_close']
            }
            return jsonify(response)
        else:
            return jsonify({'error': 'No data found for the given date'}), 404
    
    except FileNotFoundError:
        return jsonify({'error': 'Prediction data not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@api_bp.route('/forecasts', methods=['GET'])  # Route to get all forecasts
def get_forecasts():
    try:
        # Read the forecasts from the JSON file
        with open('forecasts.json', 'r') as f:
            forecasts = json.load(f)
        
        return jsonify(forecasts)
    
    except FileNotFoundError:
        return jsonify({'error': 'Forecasting data not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/forecast/<date>', methods=['GET'])  # Include <date> in your route
def get_forecast_by_date(date):
    try:
        # Read the forecasts from the JSON file
        with open('forecasts.json', 'r') as f:
            forecasts = json.load(f)
        
        # Find the forecast for the requested date
        forecast = next((item for item in forecasts if item['date'] == date), None)
        
        if forecast:
            return jsonify(forecast)
        else:
            return jsonify({'error': 'No forecast found for the given date'}), 404
    
    except FileNotFoundError:
        return jsonify({'error': 'Forecasting data not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


