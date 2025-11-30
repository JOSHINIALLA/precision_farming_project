"""
Precision Farming ML - Flask Backend API
RESTful API for farming resource optimization
"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from predictor import FarmingPredictor

# Initialize Flask app
app = Flask(__name__, 
            template_folder='../frontend/templates',
            static_folder='../frontend/static')
CORS(app)

# Initialize predictor
predictor = FarmingPredictor(models_dir='../models')

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        # Get input data
        data = request.json

        # Validate required fields
        required_fields = ['soil_moisture_%', 'soil_pH', 'temperature_C', 'rainfall_mm',
                          'humidity_%', 'sunlight_hours', 'total_days', 'NDVI_index',
                          'region', 'crop_type', 'irrigation_type', 'fertilizer_type', 
                          'crop_disease_status']

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400

        # Get recommendations
        recommendations = predictor.get_recommendations(data)

        return jsonify({
            'success': True,
            'predictions': recommendations,
            'input_data': data
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Precision Farming ML API',
        'version': '1.0.0'
    })

@app.route('/api/models/info', methods=['GET'])
def models_info():
    """Get information about loaded models"""
    import json
    with open('../models/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    return jsonify(metadata)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting Precision Farming ML API Server")
    print("="*60)
    print("\n📡 API Endpoints:")
    print("   GET  /                  - Web Interface")
    print("   POST /api/predict       - Get farming recommendations")
    print("   GET  /api/health        - Health check")
    print("   GET  /api/models/info   - Model information")
    print("\n🌐 Server running at: http://localhost:5000")
    print("="*60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
