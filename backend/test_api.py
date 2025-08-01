import requests
import json

def test_api():
    """Test the API endpoint"""
    try:
        # Test data with realistic lab values
        test_data = {
            'HAEMATOCRIT': 41.5,    # Normal range: 36-46%
            'HAEMOGLOBINS': 13.8,   # Normal range: 12-16 g/dL
            'ERYTHROCYTE': 4.6,     # Normal range: 4.0-5.5 × 10¹²/L
            'LEUCOCYTE': 8.2,       # Normal range: 4.0-11.0 × 10⁹/L
            'THROMBOCYTE': 320,     # Normal range: 150-450 × 10⁹/L
            'MCH': 30.2,            # Normal range: 27-33 pg
            'MCHC': 33.5,           # Normal range: 32-36 g/dL
            'MCV': 92.1,            # Normal range: 80-100 fL
            'AGE': 38,              # Reasonable age
            'SEX': 1                # Male
        }
        
        print("🔄 Testing API endpoint...")
        response = requests.post('http://localhost:5000/predict', json=test_data)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API Test Successful!")
            print(f"Prediction: {result.get('prediction')}")
            print(f"Confidence: {result.get('probability'):.3f}")
            
            # Print the extracted values being sent to dashboard
            print("\n📊 Dashboard Data (extracted_values):")
            extracted_values = result.get('extracted_values', {})
            for key, value in extracted_values.items():
                print(f"  {key}: {value}")
            
            # Calculate what the health score should be
            print("\n🏥 Expected Health Score Calculation:")
            parameters = [
                ('LEUCOCYTE', extracted_values.get('LEUCOCYTE', 0), 4.5, 11.0),
                ('THROMBOCYTE', extracted_values.get('THROMBOCYTE', 0), 150, 450),
                ('MCH', extracted_values.get('MCH', 0), 27, 32),
                ('MCHC', extracted_values.get('MCHC', 0), 32, 36),
                ('MCV', extracted_values.get('MCV', 0), 80, 96),
                ('HAEMATOCRIT', extracted_values.get('HAEMATOCRIT', 0), 37, 47),
                ('HAEMOGLOBINS', extracted_values.get('HAEMOGLOBINS', 0), 12, 16),
                ('ERYTHROCYTE', extracted_values.get('ERYTHROCYTE', 0), 4.2, 5.4)
            ]
            
            score = 0
            total = len(parameters)
            for param_name, value, min_val, max_val in parameters:
                status = "NORMAL" if min_val <= value <= max_val else "ABNORMAL"
                print(f"  {param_name}: {value} ({status}) - Range: {min_val}-{max_val}")
                if min_val <= value <= max_val:
                    score += 1
            
            expected_score = round((score / total) * 100)
            print(f"\n🎯 Expected Health Score: {expected_score}% ({score}/{total} parameters normal)")
            
        else:
            print(f"❌ API Test Failed! Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Could not connect to the API. Is the backend server running?")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    test_api() 