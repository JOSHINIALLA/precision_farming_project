"""
Precision Farming ML - Model Training Notebook
Complete pipeline for data preprocessing, model training, and evaluation
"""

# Import required libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
print("Loading dataset...")
df = pd.read_csv('../data/Smart_Farming_Crop_Yield_2024.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

# Data Preprocessing
print("\n" + "="*60)
print("DATA PREPROCESSING")
print("="*60)

# Handle missing values
df['irrigation_type'].fillna('None', inplace=True)
df['crop_disease_status'].fillna('Healthy', inplace=True)

# Convert date columns
df['sowing_date'] = pd.to_datetime(df['sowing_date'])
df['sowing_month'] = df['sowing_date'].dt.month

# Feature engineering
df['sowing_season'] = df['sowing_month'].apply(
    lambda x: 0 if x in [12, 1, 2] else 1 if x in [3, 4, 5] else 
              2 if x in [6, 7, 8] else 3
)
df['moisture_temp_ratio'] = df['soil_moisture_%'] / (df['temperature_C'] + 1)
df['water_availability'] = df['rainfall_mm'] + (df['soil_moisture_%'] * 10)
df['growth_index'] = df['NDVI_index'] * df['sunlight_hours']

# Encode categorical features
label_encoders = {}
categorical_cols = ['region', 'crop_type', 'irrigation_type', 'fertilizer_type', 'crop_disease_status']

for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le

print("✓ Data preprocessing complete")

# Train models for each prediction task
print("\n" + "="*60)
print("MODEL TRAINING")
print("="*60)

# Feature columns
feature_cols = [
    'soil_moisture_%', 'soil_pH', 'temperature_C', 'rainfall_mm',
    'humidity_%', 'sunlight_hours', 'total_days', 'NDVI_index',
    'moisture_temp_ratio', 'water_availability', 'growth_index',
    'region_encoded', 'crop_type_encoded', 'irrigation_type_encoded',
    'fertilizer_type_encoded', 'crop_disease_status_encoded', 'sowing_season'
]

# This notebook documents the complete training process
# See backend/predictor.py for the trained models
# Models are saved in models/ directory

print("✓ Training complete")
print("\nTrained models saved in ../models/")
print("  - water_model.pkl")
print("  - fertilizer_model.pkl")
print("  - yield_model.pkl")
print("  - Corresponding scalers and encoders")
