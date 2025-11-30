"""
Test script to verify the ML system works correctly
"""
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from predictor import FarmingPredictor

def test_predictor():
    """Test the predictor with sample data"""
    print("="*60)
    print("TESTING PRECISION FARMING ML SYSTEM")
    print("="*60)

    # Initialize predictor
    print("\n1. Loading models...")
    predictor = FarmingPredictor(models_dir='models')
    print("✓ Models loaded successfully")

    # Test data
    test_inputs = [
        {
            'name': 'Test 1: Wheat Farm - North India',
            'data': {
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
        },
        {
            'name': 'Test 2: Rice Farm - South India',
            'data': {
                'soil_moisture_%': 35.0,
                'soil_pH': 6.0,
                'temperature_C': 30.0,
                'rainfall_mm': 200.0,
                'humidity_%': 75.0,
                'sunlight_hours': 6.5,
                'total_days': 130,
                'NDVI_index': 0.70,
                'region': 'South India',
                'crop_type': 'Rice',
                'irrigation_type': 'Sprinkler',
                'fertilizer_type': 'Urea',
                'crop_disease_status': 'Mild'
            }
        },
        {
            'name': 'Test 3: Maize Farm - Central USA',
            'data': {
                'soil_moisture_%': 20.0,
                'soil_pH': 7.0,
                'temperature_C': 25.0,
                'rainfall_mm': 100.0,
                'humidity_%': 55.0,
                'sunlight_hours': 8.5,
                'total_days': 110,
                'NDVI_index': 0.60,
                'region': 'Central USA',
                'crop_type': 'Maize',
                'irrigation_type': 'Drip',
                'fertilizer_type': 'NPK',
                'crop_disease_status': 'Healthy'
            }
        }
    ]

    print("\n2. Running predictions...\n")

    all_tests_passed = True

    for test in test_inputs:
        print("-" * 60)
        print(f"📋 {test['name']}")
        print("-" * 60)

        try:
            recommendations = predictor.get_recommendations(test['data'])

            print(f"💧 Water Requirement: {recommendations['water_requirement_mm_per_day']:.2f} mm/day")
            print(f"🧪 Fertilizer Requirement: {recommendations['fertilizer_requirement_kg_per_week']:.2f} kg/ha/week")
            print(f"🌾 Expected Yield: {recommendations['expected_yield_kg_per_hectare']:.2f} kg/ha")
            print(f"\n📋 {recommendations['irrigation_recommendation']}")
            print(f"📋 {recommendations['fertilizer_recommendation']}")

            print(f"\n💡 Tips:")
            for tip in recommendations['yield_optimization_tips']:
                print(f"   {tip}")

            print("\n✅ Test passed")

        except Exception as e:
            print(f"\n❌ Test failed: {str(e)}")
            all_tests_passed = False

        print()

    print("="*60)
    if all_tests_passed:
        print("✅ ALL TESTS PASSED - System is working correctly!")
    else:
        print("❌ SOME TESTS FAILED - Please check the errors above")
    print("="*60)

    return all_tests_passed

if __name__ == "__main__":
    test_predictor()
