import os
import pandas as pd
import joblib
import sys

# Add the current directory to Python path so the functions are available
sys.path.append(os.path.dirname(__file__))

# Import the functions from app.py
from app import feature_engineering, prepare_input_data

def test_model_loading():
    """Test if the model can be loaded successfully"""
    try:
        MODEL_PATH = os.path.join(os.path.dirname(__file__), 'final_model_pipeline.pkl')
        
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model not found at {MODEL_PATH}")
            return False
            
        print("🔄 Loading model...")
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully! Type: {type(model)}")
        
        # Test prediction
        test_data = {
            'HAEMATOCRIT': 45.0,
            'HAEMOGLOBINS': 14.0,
            'ERYTHROCYTE': 5.0,
            'LEUCOCYTE': 7.0,
            'THROMBOCYTE': 250.0,
            'MCH': 29.0,
            'MCHC': 34.0,
            'MCV': 90.0,
            'AGE': 35.0,
            'SEX_ENCODED': 1
        }
        
        print("🔄 Testing prediction...")
        input_df = prepare_input_data(test_data)
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        result = "Inpatient" if prediction[0] == 1 else "Outpatient"
        confidence = float(prediction_proba[0][1])
        
        print(f"✅ Prediction successful!")
        print(f"   Result: {result}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Raw prediction: {prediction[0]}")
        print(f"   Probabilities: {prediction_proba[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n🎉 Model test completed successfully!")
    else:
        print("\n💥 Model test failed!")
        sys.exit(1) 