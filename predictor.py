"""Precision Farming ML - Prediction Module
Handles all ML predictions for resource optimization.

Path handling note:
When running `backend/app.py` directly, the project root (where `_loss.py` shim lives)
is not on `sys.path`. We ensure parent directory is added so that the compatibility
shim for scikit-learn pickled models resolves correctly.
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ensure project root is on sys.path for `_loss.py` shim to be discoverable
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

class FarmingPredictor:
    """Main predictor class for farming resource optimization"""

    def __init__(self, models_dir='models'):
        """Initialize predictor with trained models"""
        self.models_dir = Path(models_dir)
        self._load_models()

    def _load_models(self):
        """Load all trained models and encoders"""
        # Load models
        with open(self.models_dir / 'water_model.pkl', 'rb') as f:
            self.water_model = pickle.load(f)

        with open(self.models_dir / 'water_scaler.pkl', 'rb') as f:
            self.water_scaler = pickle.load(f)

        with open(self.models_dir / 'fertilizer_model.pkl', 'rb') as f:
            self.fertilizer_model = pickle.load(f)

        with open(self.models_dir / 'fertilizer_scaler.pkl', 'rb') as f:
            self.fertilizer_scaler = pickle.load(f)

        with open(self.models_dir / 'yield_model.pkl', 'rb') as f:
            self.yield_model = pickle.load(f)

        with open(self.models_dir / 'yield_scaler.pkl', 'rb') as f:
            self.yield_scaler = pickle.load(f)

        with open(self.models_dir / 'label_encoders.pkl', 'rb') as f:
            self.label_encoders = pickle.load(f)

        with open(self.models_dir / 'feature_columns.pkl', 'rb') as f:
            self.feature_columns = pickle.load(f)

        print("✓ All models loaded successfully")

    def preprocess_input(self, input_data):
        """Preprocess input data for prediction"""
        df = pd.DataFrame([input_data]) if isinstance(input_data, dict) else input_data.copy()

        # Extract sowing month if sowing_date provided
        if 'sowing_date' in df.columns:
            df['sowing_date'] = pd.to_datetime(df['sowing_date'])
            df['sowing_month'] = df['sowing_date'].dt.month
        elif 'sowing_month' not in df.columns:
            df['sowing_month'] = 3  # Default to March

        # Create sowing season
        df['sowing_season'] = df['sowing_month'].apply(
            lambda x: 0 if x in [12, 1, 2] else 1 if x in [3, 4, 5] else 
                      2 if x in [6, 7, 8] else 3
        )

        # Calculate derived features
        df['moisture_temp_ratio'] = df['soil_moisture_%'] / (df['temperature_C'] + 1)
        df['water_availability'] = df['rainfall_mm'] + (df['soil_moisture_%'] * 10)
        df['growth_index'] = df['NDVI_index'] * df['sunlight_hours']

        # Encode categorical variables with graceful handling of unseen labels
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                values = df[col].astype(str)
                encoded = []
                for v in values:
                    if v in encoder.classes_:
                        encoded.append(encoder.transform([v])[0])
                    else:
                        # Unseen category: map to 0 (could be replaced with more robust strategy)
                        encoded.append(0)
                df[col + '_encoded'] = encoded
            else:
                df[col + '_encoded'] = 0  # Default encoding

        # Select only required features in correct order
        X = df[self.feature_columns]

        return X

    def predict_water_requirement(self, input_data):
        """Predict water requirement in mm per day"""
        X = self.preprocess_input(input_data)
        X_scaled = self.water_scaler.transform(X)
        prediction = self.water_model.predict(X_scaled)
        return max(0, prediction[0])

    def predict_fertilizer_requirement(self, input_data):
        """Predict fertilizer requirement in kg per hectare per week"""
        X = self.preprocess_input(input_data)
        X_scaled = self.fertilizer_scaler.transform(X)
        prediction = self.fertilizer_model.predict(X_scaled)
        return max(5, prediction[0])

    def predict_yield(self, input_data):
        """Predict crop yield in kg per hectare"""
        X = self.preprocess_input(input_data)
        X_scaled = self.yield_scaler.transform(X)
        prediction = self.yield_model.predict(X_scaled)
        return max(0, prediction[0])

    def get_recommendations(self, input_data):
        """Get comprehensive farming recommendations"""
        water_req = self.predict_water_requirement(input_data)
        fertilizer_req = self.predict_fertilizer_requirement(input_data)
        expected_yield = self.predict_yield(input_data)

        # Generate recommendations
        recommendations = {
            'water_requirement_mm_per_day': round(water_req, 2),
            'fertilizer_requirement_kg_per_week': round(fertilizer_req, 2),
            'expected_yield_kg_per_hectare': round(expected_yield, 2),
            'irrigation_recommendation': self._get_irrigation_recommendation(water_req, input_data),
            'fertilizer_recommendation': self._get_fertilizer_recommendation(fertilizer_req),
            'yield_optimization_tips': self._get_yield_tips(input_data, expected_yield)
        }

        return recommendations

    def _get_irrigation_recommendation(self, water_req, input_data):
        """Generate irrigation recommendation"""
        rainfall = input_data.get('rainfall_mm', 0)
        soil_moisture = input_data.get('soil_moisture_%', 0)

        if rainfall > 10:
            return f"Light irrigation recommended ({water_req:.1f}mm). Recent rainfall detected ({rainfall:.1f}mm)."
        elif soil_moisture < 20:
            return f"Immediate irrigation required ({water_req:.1f}mm). Soil moisture critically low ({soil_moisture:.1f}%)."
        elif water_req > 12:
            return f"Heavy irrigation needed ({water_req:.1f}mm). High water demand conditions."
        elif water_req > 8:
            return f"Moderate irrigation required ({water_req:.1f}mm). Standard water application."
        else:
            return f"Light irrigation sufficient ({water_req:.1f}mm). Soil moisture adequate ({soil_moisture:.1f}%)."

    def _get_fertilizer_recommendation(self, fertilizer_req):
        """Generate fertilizer recommendation"""
        if fertilizer_req > 45:
            return f"High fertilizer application: {fertilizer_req:.1f} kg/ha this week. Split into 2 doses."
        elif fertilizer_req > 30:
            return f"Moderate fertilizer application: {fertilizer_req:.1f} kg/ha this week."
        else:
            return f"Low fertilizer application: {fertilizer_req:.1f} kg/ha this week. Soil nutrients adequate."

    def _get_yield_tips(self, input_data, expected_yield):
        """Generate yield optimization tips"""
        tips = []

        ndvi = input_data.get('NDVI_index', 0.5)
        if ndvi < 0.5:
            tips.append("⚠️ Low vegetation health (NDVI). Consider foliar nutrition.")

        soil_ph = input_data.get('soil_pH', 6.5)
        if soil_ph < 6.0 or soil_ph > 7.5:
            tips.append(f"⚠️ Soil pH ({soil_ph:.1f}) outside optimal range. Consider pH correction.")

        temp = input_data.get('temperature_C', 25)
        if temp > 32:
            tips.append("🌡️ High temperature stress. Increase irrigation frequency.")

        tips.append(f"📈 Expected yield: {expected_yield:.0f} kg/ha")

        return tips

if __name__ == "__main__":
    # Test the predictor
    predictor = FarmingPredictor()

    test_input = {
        'soil_moisture_%': 25.5,
        'soil_pH': 6.5,
        'temperature_C': 28.0,
        'rainfall_mm': 150.0,
        'humidity_%': 65.0,
        'sunlight_hours': 7.5,
        'total_days': 120,
        'NDVI_index': 0.65,
        'region': 'North India',
        'crop_type': 'Wheat',
        'irrigation_type': 'Drip',
        'fertilizer_type': 'Organic',
        'crop_disease_status': 'Healthy'
    }

    print("\n" + "="*60)
    print("Testing Predictor with Sample Input")
    print("="*60)

    recommendations = predictor.get_recommendations(test_input)

    print(f"\n💧 Water Requirement: {recommendations['water_requirement_mm_per_day']} mm/day")
    print(f"🧪 Fertilizer Requirement: {recommendations['fertilizer_requirement_kg_per_week']} kg/ha/week")
    print(f"🌾 Expected Yield: {recommendations['expected_yield_kg_per_hectare']} kg/ha")
    print(f"\n📋 Irrigation: {recommendations['irrigation_recommendation']}")
    print(f"📋 Fertilizer: {recommendations['fertilizer_recommendation']}")
    print(f"\n💡 Optimization Tips:")
    for tip in recommendations['yield_optimization_tips']:
        print(f"   {tip}")
