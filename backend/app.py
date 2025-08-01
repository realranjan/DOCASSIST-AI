from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import FunctionTransformer
import PyPDF2
import re
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import pdfplumber
import pdfkit
import tempfile

# Configure wkhtmltopdf path
WKHTMLTOPDF_PATH = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
config = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_PATH)

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
            'date': datetime.now().strftime('%Y-%m-%d'),
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

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Blood ranges (your existing constants)
BLOOD_RANGES = {
    'HAEMATOCRIT': {
        'low': {'value': 29, 'unit': '%'},
        'high': {'value': 66, 'unit': '%'},
        'conditions': {
            'low': 'Possible anemia',
            'high': 'Possible polycythemia or dehydration'
        }
    },
    # ... (include all your other blood ranges)
}

# Disease patterns (your existing constants)
DISEASE_PATTERNS = {
    'IRON_DEFICIENCY_ANEMIA': {
        'conditions': {
            'HAEMOGLOBINS': {'condition': 'low', 'importance': 'primary'},
            'MCV': {'condition': 'low', 'importance': 'primary'},
            'MCH': {'condition': 'low', 'importance': 'secondary'},
            'HAEMATOCRIT': {'condition': 'low', 'importance': 'secondary'}
        },
        'treatments': [
            "Primary Treatment: Ferrous sulfate 325mg oral tablet twice daily",
            "Precautions: Take iron on empty stomach, avoid antacids",
            "Monitoring: CBC every 2-3 weeks until hemoglobin normalizes"
        ]
    }
    # ... (include all your other disease patterns)
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_medical_values(text):
    """Extract medical values from text using regex patterns"""
    patterns = {
        'HAEMATOCRIT': r'(?i)h[ae]matocrit.*?(\d+\.?\d*)',
        'HAEMOGLOBINS': r'(?i)h[ae]moglobin.*?(\d+\.?\d*)',
        'ERYTHROCYTE': r'(?i)erythrocyte.*?(\d+\.?\d*)',
        'LEUCOCYTE': r'(?i)leucocyte.*?(\d+\.?\d*)',
        'THROMBOCYTE': r'(?i)thrombocyte.*?(\d+\.?\d*)',
        'MCH': r'(?i)MCH.*?(\d+\.?\d*)',
        'MCHC': r'(?i)MCHC.*?(\d+\.?\d*)',
        'MCV': r'(?i)MCV.*?(\d+\.?\d*)',
        'AGE': r'(?i)age.*?(\d+)',
        'SEX': r'(?i)(?:sex|gender).*?(male|female|m|f)'
    }
    
    results = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            if key == 'SEX':
                value = 1 if value.lower() in ['male', 'm'] else 0
            else:
                value = float(value)
            results[key] = value
    return results

# SIMPLIFIED Feature engineering function
def feature_engineering(df):
    """Apply the same feature engineering as in training"""
    df_copy = df.copy()
    
    # Create engineered features
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
            print("Model loaded successfully")
            return True
        else:
            print(f"Model not found at {MODEL_PATH}")
            return False
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

# Load model on startup
if not load_model():
    print("Warning: Failed to load model. Some endpoints may not work.")

@app.route('/')
def home():
    return jsonify({
        'status': 'success',
        'message': 'DocAssist AI Backend is running'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Create input data
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
        
        print("Input Data:", input_data)
        
        # Prepare input for model
        input_df = prepare_input_data(input_data)
        
        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        # Add to session data
        report = session_data.add_report(input_data, prediction[0])
        
        result = "Inpatient" if prediction[0] == 1 else "Outpatient"
        
        return jsonify({
            'status': 'success',
            'prediction': result,
            'prediction_code': int(prediction[0]),
            'probability': float(prediction_proba[0][1]),
            'report_id': report['patientId']
        })
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/predict/file', methods=['POST'])
def predict_from_file():
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No file uploaded'
        }), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({
            'status': 'error',
            'message': 'Invalid file type. Only PDF files are allowed.'
        }), 400
    
    try:
        # Save and process file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text from PDF
        text = ""
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        
        # Extract medical values
        extracted_data = extract_medical_values(text)
        
        # Check for required fields
        required_fields = ['HAEMATOCRIT', 'HAEMOGLOBINS', 'ERYTHROCYTE', 'LEUCOCYTE', 
                          'THROMBOCYTE', 'MCH', 'MCHC', 'MCV', 'AGE', 'SEX']
        missing_fields = [field for field in required_fields if field not in extracted_data]
        
        if missing_fields:
            os.remove(filepath)
            return jsonify({
                'status': 'error',
                'message': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Prepare input data
        input_data = {
            'HAEMATOCRIT': extracted_data['HAEMATOCRIT'],
            'HAEMOGLOBINS': extracted_data['HAEMOGLOBINS'],
            'ERYTHROCYTE': extracted_data['ERYTHROCYTE'],
            'LEUCOCYTE': extracted_data['LEUCOCYTE'],
            'THROMBOCYTE': extracted_data['THROMBOCYTE'],
            'MCH': extracted_data['MCH'],
            'MCHC': extracted_data['MCHC'],
            'MCV': extracted_data['MCV'],
            'AGE': extracted_data['AGE'],
            'SEX_ENCODED': extracted_data['SEX']
        }
        
        # Make prediction
        input_df = prepare_input_data(input_data)
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        # Add to session data
        report = session_data.add_report(input_data, prediction[0])
        
        result = "Inpatient" if prediction[0] == 1 else "Outpatient"
        
        # Clean up file
        os.remove(filepath)
        
        return jsonify({
            'status': 'success',
            'prediction': result,
            'prediction_code': int(prediction[0]),
            'probability': float(prediction_proba[0][1]),
            'extracted_values': extracted_data,
            'report_id': report['patientId']
        })
        
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        print(f"Error in file prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/reports', methods=['GET'])
def get_reports():
    return jsonify({
        'status': 'success',
        'reports': session_data.reports,
        'summary': {
            'total': session_data.total_reports,
            'inpatient': session_data.inpatient_count,
            'outpatient': session_data.outpatient_count
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
