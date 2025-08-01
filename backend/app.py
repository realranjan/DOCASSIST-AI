import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Session storage class
class SessionData:
    def __init__(self):
        self.reports = []
        self.total_reports = 0
        self.outpatient_count = 0
        self.inpatient_count = 0
    
    def add_report(self, data, prediction):
        report = {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'patientId': f'P{self.total_reports + 1:03d}',
            'age': int(data.get('AGE', 0)),
            'sex': 'M' if int(data.get('SEX_ENCODED', 1)) == 1 else 'F',
            'prediction': 'Inpatient' if prediction == 1 else 'Outpatient',
            'status': 'Completed',
            'parameters': {
                'HAEMATOCRIT': float(data.get('HAEMATOCRIT', 0)),
                'HAEMOGLOBINS': float(data.get('HAEMOGLOBINS', 0)),
                'ERYTHROCYTE': float(data.get('ERYTHROCYTE', 0)),
                'LEUCOCYTE': float(data.get('LEUCOCYTE', 0)),
                'THROMBOCYTE': float(data.get('THROMBOCYTE', 0)),
                'MCH': float(data.get('MCH', 0)),
                'MCHC': float(data.get('MCHC', 0)),
                'MCV': float(data.get('MCV', 0))
            }
        }
        self.reports.insert(0, report)
        self.total_reports += 1
        
        if prediction == 1:
            self.inpatient_count += 1
        else:
            self.outpatient_count += 1
        
        # Keep only last 100 reports
        if len(self.reports) > 100:
            self.reports.pop()
            
        return report

# Initialize session data
session_data = SessionData()

# Feature engineering function
def feature_engineering(df):
    """Apply the same feature engineering as in training"""
    df_copy = df.copy()
    df_copy['THROMBOCYTE_LEUCOCYTE_RATIO'] = df_copy['THROMBOCYTE'] / (df_copy['LEUCOCYTE'] + 1e-6)
    df_copy['ERYTHROCYTE_LEUCOCYTE'] = df_copy['ERYTHROCYTE'] * df_copy['LEUCOCYTE']
    return df_copy

def prepare_input_data(data):
    """Prepare input data with correct feature names and order"""
    df = pd.DataFrame([data])
    df = feature_engineering(df)
    return df

# Model loading
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'final_model_pipeline.pkl')
model = None

def load_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print("✅ Model loaded successfully")
            return True
        else:
            print(f"❌ Model not found at {MODEL_PATH}")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

# Load model on startup
if not load_model():
    print("⚠️ Warning: Failed to load model. Some endpoints may not work.")

@app.route('/')
def home():
    return jsonify({
        'status': 'success',
        'message': 'DocAssist AI Backend is running',
        'version': '1.0.0',
        'features': ['ML Prediction', 'JSON Input', 'Reports Dashboard'],
        'model_loaded': model is not None
    })

@app.route('/health')
def health_check():
    try:
        if model is None:
            return jsonify({
                'status': 'unhealthy',
                'error': 'Model not loaded',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        return jsonify({
            'status': 'healthy',
            'service': 'DocAssist AI',
            'model_loaded': True,
            'reports_count': session_data.total_reports,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 503

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not available'
            }), 503
            
        data = request.json
        
        # Create input data with validation
        input_data = {
            'HAEMATOCRIT': float(data.get('HAEMATOCRIT', 45)),
            'HAEMOGLOBINS': float(data.get('HAEMOGLOBINS', 14)),
            'ERYTHROCYTE': float(data.get('ERYTHROCYTE', 5)),
            'LEUCOCYTE': float(data.get('LEUCOCYTE', 7)),
            'THROMBOCYTE': float(data.get('THROMBOCYTE', 250)),
            'MCH': float(data.get('MCH', 29)),
            'MCHC': float(data.get('MCHC', 34)),
            'MCV': float(data.get('MCV', 90)),
            'AGE': float(data.get('AGE', 35)),
            'SEX_ENCODED': int(data.get('SEX', 1))
        }
        
        print(f"📊 Processing prediction for: {input_data}")
        
        # Prepare input for model
        input_df = prepare_input_data(input_data)
        
        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        # Add to session data
        report = session_data.add_report(input_data, prediction[0])
        
        result = "Inpatient" if prediction[0] == 1 else "Outpatient"
        confidence = float(prediction_proba[0][1])
        
        print(f"🎯 Prediction: {result} (confidence: {confidence:.3f})")
        
        return jsonify({
            'status': 'success',
            'prediction': result,
            'prediction_code': int(prediction[0]),
            'probability': confidence,
            'confidence_level': 'High' if confidence > 0.8 or confidence < 0.2 else 'Medium',
            'report_id': report['patientId'],
            'timestamp': report['date']
        })
        
    except ValueError as ve:
        return jsonify({
            'status': 'error',
            'message': f'Invalid input data: {str(ve)}'
        }), 400
    except Exception as e:
        print(f"❌ Error in prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/reports', methods=['GET'])
def get_reports():
    try:
        return jsonify({
            'status': 'success',
            'reports': session_data.reports,
            'summary': {
                'total': session_data.total_reports,
                'inpatient': session_data.inpatient_count,
                'outpatient': session_data.outpatient_count,
                'inpatient_percentage': round((session_data.inpatient_count / max(session_data.total_reports, 1)) * 100, 1),
                'outpatient_percentage': round((session_data.outpatient_count / max(session_data.total_reports, 1)) * 100, 1)
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/report/<report_id>')
def get_single_report(report_id):
    try:
        report = next((r for r in session_data.reports if r['patientId'] == report_id), None)
        if not report:
            return jsonify({
                'status': 'error',
                'message': 'Report not found'
            }), 404
            
        return jsonify({
            'status': 'success',
            'report': report
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    
    print(f"🚀 Starting DocAssist AI on port {port}")
    print(f"🔧 Debug mode: {debug_mode}")
    print(f"🤖 Model loaded: {model is not None}")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
