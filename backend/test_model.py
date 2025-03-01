import requests
import json

def test_normal_values():
    # Test with normal values
    data = {
        'HAEMATOCRIT': 45.0,
        'HAEMOGLOBINS': 14.0,
        'ERYTHROCYTE': 5.0,
        'LEUCOCYTE': 7.0,
        'THROMBOCYTE': 250.0,
        'MCH': 29.0,
        'MCHC': 34.0,
        'MCV': 90.0,
        'AGE': 35,
        'SEX_ENCODED': 1
    }
    
    response = requests.post('http://localhost:5000/predict', json=data)
    print("\nTest with normal values:")
    print(json.dumps(response.json(), indent=2))

def test_abnormal_values():
    # Test with abnormal values (severe anemia + infection)
    data = {
        'HAEMATOCRIT': 25.0,  # Very low
        'HAEMOGLOBINS': 10.0, # Low
        'ERYTHROCYTE': 3.5,   # Low
        'LEUCOCYTE': 12.0,    # High (infection)
        'THROMBOCYTE': 500.0, # High
        'MCH': 25.0,          # Low
        'MCHC': 30.0,         # Low
        'MCV': 75.0,          # Low
        'AGE': 65,            # Elderly
        'SEX_ENCODED': 1
    }
    
    response = requests.post('http://localhost:5000/predict', json=data)
    print("\nTest with abnormal values:")
    print(json.dumps(response.json(), indent=2))

if __name__ == '__main__':
    print("Testing model predictions...")
    test_normal_values()
    test_abnormal_values() 