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

# Feature engineering function - MUST match the one used during training
def feature_engineering(df):
    """Apply the same feature engineering as in training"""
    df_copy = df.copy()
    df_copy['THROMBOCYTE_LEUCOCYTE_RATIO'] = df_copy['THROMBOCYTE'] / (df_copy['LEUCOCYTE'] + 1e-6)
    df_copy['ERYTHROCYTE_LEUCOCYTE'] = df_copy['ERYTHROCYTE'] * df_copy['LEUCOCYTE']
    return df_copy

def prepare_input_data(data):
    """Prepare input data with correct feature names and order"""
    # Create DataFrame with explicit column names
    df = pd.DataFrame([data])
    
    # Apply feature engineering
    df = feature_engineering(df)
    
    # Ensure we have all expected features in the correct order
    expected_features = [
        'HAEMATOCRIT', 'HAEMOGLOBINS', 'ERYTHROCYTE', 'LEUCOCYTE', 
        'THROMBOCYTE', 'MCH', 'MCHC', 'MCV', 'AGE', 'SEX_ENCODED',
        'THROMBOCYTE_LEUCOCYTE_RATIO', 'ERYTHROCYTE_LEUCOCYTE'
    ]
    
    # Reorder columns to match expected feature order
    df = df.reindex(columns=expected_features)
    
    return df

def generate_medical_report(data, prediction, confidence):
    """Generate a detailed medical report based on the prediction"""
    age = data.get('AGE', 35)
    sex = 'Male' if data.get('SEX_ENCODED', 1) == 1 else 'Female'
    
    # Validate and format lab values with normal ranges
    def format_lab_value(value, unit, normal_range, decimal_places=1):
        if value < 0 or value > 1000:  # Unrealistic values
            return f"<span style='color: #ef4444;'>Invalid ({value:.1f} {unit})</span>"
        elif value < normal_range[0] or value > normal_range[1]:
            return f"<span style='color: #f59e0b;'>{value:.1f} {unit} (Abnormal)</span>"
        else:
            return f"{value:.1f} {unit} (Normal)"
    
    # Normal ranges for lab values
    lab_values = {
        'HAEMATOCRIT': (data.get('HAEMATOCRIT', 0), '%', (36.0, 46.0)),
        'HAEMOGLOBINS': (data.get('HAEMOGLOBINS', 0), 'g/dL', (12.0, 16.0)),
        'ERYTHROCYTE': (data.get('ERYTHROCYTE', 0), '× 10¹²/L', (4.0, 5.5)),
        'LEUCOCYTE': (data.get('LEUCOCYTE', 0), '× 10⁹/L', (4.0, 11.0)),
        'THROMBOCYTE': (data.get('THROMBOCYTE', 0), '× 10⁹/L', (150.0, 450.0)),
        'MCH': (data.get('MCH', 0), 'pg', (27.0, 33.0)),
        'MCHC': (data.get('MCHC', 0), 'g/dL', (32.0, 36.0)),
        'MCV': (data.get('MCV', 0), 'fL', (80.0, 100.0))
    }
    
    report = f"""
    <div class="medical-report-content">
        <h3>Medical Analysis Report</h3>
        <div class="patient-info">
            <p><strong>Patient Demographics:</strong> {age}-year-old {sex}</p>
            <p><strong>Analysis Confidence:</strong> {confidence:.1%}</p>
        </div>
        
        <div class="lab-results">
            <h4>Laboratory Parameters:</h4>
            <ul>
                <li><strong>Hematocrit:</strong> {format_lab_value(*lab_values['HAEMATOCRIT'])}</li>
                <li><strong>Hemoglobin:</strong> {format_lab_value(*lab_values['HAEMOGLOBINS'])}</li>
                <li><strong>Erythrocyte Count:</strong> {format_lab_value(lab_values['ERYTHROCYTE'][0], lab_values['ERYTHROCYTE'][1], lab_values['ERYTHROCYTE'][2], 2)}</li>
                <li><strong>Leukocyte Count:</strong> {format_lab_value(*lab_values['LEUCOCYTE'])}</li>
                <li><strong>Thrombocyte Count:</strong> {format_lab_value(lab_values['THROMBOCYTE'][0], lab_values['THROMBOCYTE'][1], lab_values['THROMBOCYTE'][2], 0)}</li>
                <li><strong>MCH:</strong> {format_lab_value(*lab_values['MCH'])}</li>
                <li><strong>MCHC:</strong> {format_lab_value(*lab_values['MCHC'])}</li>
                <li><strong>MCV:</strong> {format_lab_value(*lab_values['MCV'])}</li>
            </ul>
        </div>
        
        <div class="analysis-summary">
            <h4>Analysis Summary:</h4>
            <p>Based on the comprehensive analysis of blood parameters and patient demographics, 
            the AI-powered medical decision support system has determined that this patient 
            requires <strong>{prediction.lower()} care</strong>.</p>
            
            <div class="confidence-note">
                <p><em>Note: This analysis is based on machine learning algorithms and should be 
                reviewed by qualified healthcare professionals. The confidence level indicates 
                the reliability of this prediction.</em></p>
            </div>
        </div>
    </div>
    """
    return report

def generate_recommendations(prediction, confidence, data):
    """Generate treatment recommendations based on the prediction"""
    if prediction == "Inpatient":
        recommendations = f"""
        <div class="recommendations-content">
            <h4>Immediate Actions Required:</h4>
            <ul>
                <li><strong>Hospital Admission:</strong> Immediate inpatient care is recommended</li>
                <li><strong>Continuous Monitoring:</strong> Vital signs and lab parameters require close observation</li>
                <li><strong>Specialist Consultation:</strong> Refer to appropriate medical specialist</li>
                <li><strong>Treatment Protocol:</strong> Initiate standard inpatient treatment protocols</li>
            </ul>
            
            <h4>Monitoring Requirements:</h4>
            <ul>
                <li>Hourly vital sign monitoring</li>
                <li>Daily complete blood count (CBC)</li>
                <li>Continuous cardiac monitoring if indicated</li>
                <li>Regular assessment of treatment response</li>
            </ul>
        </div>
        """
    else:
        recommendations = f"""
        <div class="recommendations-content">
            <h4>Outpatient Care Recommendations:</h4>
            <ul>
                <li><strong>Follow-up Schedule:</strong> Schedule follow-up appointment within 1-2 weeks</li>
                <li><strong>Home Monitoring:</strong> Monitor symptoms and report any changes</li>
                <li><strong>Lifestyle Modifications:</strong> Implement recommended lifestyle changes</li>
                <li><strong>Medication Management:</strong> Continue prescribed medications as directed</li>
            </ul>
            
            <h4>Preventive Measures:</h4>
            <ul>
                <li>Regular health check-ups</li>
                <li>Maintain healthy diet and exercise routine</li>
                <li>Avoid risk factors that may worsen condition</li>
                <li>Keep emergency contact information readily available</li>
            </ul>
        </div>
        """
    
    return recommendations

# Model loading
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'final_model_pipeline.pkl')
model = None

def load_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            # Try loading with joblib first
            try:
                model = joblib.load(MODEL_PATH)
                print("✅ Model loaded successfully with joblib")
                return True
            except Exception as joblib_error:
                print(f"⚠️ Joblib loading failed: {str(joblib_error)}")
                
                # Try alternative loading method for LightGBM
                try:
                    import pickle
                    with open(MODEL_PATH, 'rb') as f:
                        model = pickle.load(f)
                    print("✅ Model loaded successfully with pickle")
                    return True
                except Exception as pickle_error:
                    print(f"⚠️ Pickle loading failed: {str(pickle_error)}")
                    
                    # Try with specific LightGBM handling
                    try:
                        import lightgbm as lgb
                        with open(MODEL_PATH, 'rb') as f:
                            model = pickle.load(f)
                        # If it's a LightGBM model, ensure it's properly initialized
                        if hasattr(model, 'booster_'):
                            print("✅ LightGBM model loaded successfully")
                        return True
                    except Exception as lgb_error:
                        print(f"❌ LightGBM loading failed: {str(lgb_error)}")
                        return False
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
        
        # Create input data with validation and realistic defaults
        original_data = {
            'HAEMATOCRIT': float(data.get('HAEMATOCRIT', 42)),
            'HAEMOGLOBINS': float(data.get('HAEMOGLOBINS', 14)),
            'ERYTHROCYTE': float(data.get('ERYTHROCYTE', 4.8)),
            'LEUCOCYTE': float(data.get('LEUCOCYTE', 7.5)),
            'THROMBOCYTE': float(data.get('THROMBOCYTE', 280)),
            'MCH': float(data.get('MCH', 29)),
            'MCHC': float(data.get('MCHC', 34)),
            'MCV': float(data.get('MCV', 88)),
            'AGE': float(data.get('AGE', 45)),
            'SEX_ENCODED': int(data.get('SEX', 1))
        }
        
        # Create validated input data for prediction
        input_data = {
            'HAEMATOCRIT': min(max(original_data['HAEMATOCRIT'], 20), 60),  # Normal range: 36-46%
            'HAEMOGLOBINS': min(max(original_data['HAEMOGLOBINS'], 8), 20),  # Normal range: 12-16 g/dL
            'ERYTHROCYTE': min(max(original_data['ERYTHROCYTE'], 3), 7),    # Normal range: 4.0-5.5 × 10¹²/L
            'LEUCOCYTE': min(max(original_data['LEUCOCYTE'], 2), 20),       # Normal range: 4.0-11.0 × 10⁹/L
            'THROMBOCYTE': min(max(original_data['THROMBOCYTE'], 100), 600), # Normal range: 150-450 × 10⁹/L
            'MCH': min(max(original_data['MCH'], 20), 40),                   # Normal range: 27-33 pg
            'MCHC': min(max(original_data['MCHC'], 30), 40),                 # Normal range: 32-36 g/dL
            'MCV': min(max(original_data['MCV'], 70), 110),                  # Normal range: 80-100 fL
            'AGE': min(max(original_data['AGE'], 18), 100),                  # Reasonable age range
            'SEX_ENCODED': original_data['SEX_ENCODED']
        }
        
        print(f"📊 Processing prediction for: {input_data}")
        
        # Prepare input for model
        input_df = prepare_input_data(input_data)
        
        # Make prediction with error handling
        try:
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)
        except Exception as pred_error:
            print(f"❌ Prediction error: {str(pred_error)}")
            # Try alternative prediction method for LightGBM
            try:
                if hasattr(model, 'booster_'):
                    # For LightGBM models, try using the booster directly
                    prediction = model.booster_.predict(input_df)
                    # Convert to binary prediction
                    prediction = [1 if p > 0.5 else 0 for p in prediction]
                    prediction_proba = [[1-p, p] for p in model.booster_.predict(input_df)]
                else:
                    raise pred_error
            except Exception as alt_error:
                print(f"❌ Alternative prediction failed: {str(alt_error)}")
                return jsonify({
                    'status': 'error',
                    'message': f'Prediction failed: {str(pred_error)}'
                }), 500
        
        # Add to session data
        report = session_data.add_report(input_data, prediction[0])
        
        result = "Inpatient" if prediction[0] == 1 else "Outpatient"
        confidence = float(prediction_proba[0][1])
        
        print(f"🎯 Prediction: {result} (confidence: {confidence:.3f})")
        
        # Generate medical report and recommendations based on prediction
        medical_report = generate_medical_report(input_data, result, confidence)
        recommendations = generate_recommendations(result, confidence, input_data)
        
        return jsonify({
            'status': 'success',
            'prediction': result,
            'prediction_code': int(prediction[0]),
            'probability': confidence,
            'confidence_level': 'High' if confidence > 0.8 or confidence < 0.2 else 'Medium',
            'report_id': report['patientId'],
            'timestamp': report['date'],
            'medical_report': medical_report,
            'recommendations': recommendations,
            'extracted_values': original_data  # Send original data for dashboard
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
