from flask import Flask, request, jsonify
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

app = Flask(__name__)
CORS(app)

# Add session storage
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
            'sex': 'M' if int(data.get('SEX', 1)) == 1 else 'F',
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
        
        self.reports.insert(0, report)  # Add to start of list
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

@app.route('/')
def home():
    return jsonify({
        'status': 'success',
        'message': 'DocAssist AI Backend is running'
    })

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Add these constants at the top of the file
BLOOD_RANGES = {
    'HAEMATOCRIT': {
        'low': {'value': 29, 'unit': '%'},
        'high': {'value': 66, 'unit': '%'},
        'conditions': {
            'low': 'Possible anemia',
            'high': 'Possible polycythemia or dehydration'
        }
    },
    'HAEMOGLOBINS': {
        'low': {'value': 12, 'unit': 'g/dL'},
        'high': {'value': 18, 'unit': 'g/dL'},
        'conditions': {
            'low': 'Anemia or blood loss',
            'high': 'Polycythemia or dehydration'
        }
    },
    'ERYTHROCYTE': {
        'low': {'value': 4.0, 'unit': 'M/µL'},
        'high': {'value': 6.2, 'unit': 'M/µL'},
        'conditions': {
            'low': 'Decreased red blood cell production or increased destruction',
            'high': 'Polycythemia or bone marrow disorder'
        }
    },
    'LEUCOCYTE': {
        'low': {'value': 4.0, 'unit': 'K/µL'},
        'high': {'value': 11.0, 'unit': 'K/µL'},
        'conditions': {
            'low': 'Weakened immune system or bone marrow problems',
            'high': 'Infection, inflammation, or leukemia'
        }
    },
    'THROMBOCYTE': {
        'low': {'value': 150, 'unit': 'K/µL'},
        'high': {'value': 450, 'unit': 'K/µL'},
        'conditions': {
            'low': 'Increased bleeding risk',
            'high': 'Increased clotting risk'
        }
    },
    'MCH': {
        'low': {'value': 27, 'unit': 'pg'},
        'high': {'value': 32, 'unit': 'pg'},
        'conditions': {
            'low': 'Iron deficiency',
            'high': 'Possible vitamin B12 deficiency'
        }
    },
    'MCHC': {
        'low': {'value': 32, 'unit': 'g/dL'},
        'high': {'value': 36, 'unit': 'g/dL'},
        'conditions': {
            'low': 'Iron deficiency anemia',
            'high': 'Hereditary spherocytosis'
        }
    },
    'MCV': {
        'low': {'value': 80, 'unit': 'fL'},
        'high': {'value': 100, 'unit': 'fL'},
        'conditions': {
            'low': 'Microcytic anemia',
            'high': 'Macrocytic anemia'
        }
    }
}

DISEASE_PATTERNS = {
    'IRON DEFICIENCY ANEMIA': {
        'conditions': {
            'HAEMOGLOBINS': {'condition': 'low', 'importance': 'primary'},
            'MCV': {'condition': 'low', 'importance': 'primary'},
            'MCH': {'condition': 'low', 'importance': 'secondary'},
            'HAEMATOCRIT': {'condition': 'low', 'importance': 'secondary'}
        },
        'treatments': [
            "Primary Treatment:\n" +
            "- Ferrous sulfate 325mg oral tablet twice daily\n" +
            "- Vitamin C 500mg with iron tablets to enhance absorption\n" +
            "- Folic acid 1mg daily",
            
            "Precautions:\n" +
            "- Take iron on empty stomach\n" +
            "- Avoid antacids, calcium supplements\n" +
            "- May cause dark stools (normal side effect)",
            
            "Monitoring:\n" +
            "- CBC every 2-3 weeks until hemoglobin normalizes\n" +
            "- Ferritin levels monthly\n" +
            "- Iron studies in 3 months"
        ]
    },
    
    'MEGALOBLASTIC ANEMIA': {
        'conditions': {
            'HAEMOGLOBINS': {'condition': 'low', 'importance': 'primary'},
            'MCV': {'condition': 'high', 'importance': 'primary'},
            'MCH': {'condition': 'high', 'importance': 'secondary'}
        },
        'treatments': [
            "Primary Treatment:\n" +
            "- Vitamin B12 1000mcg IM injection weekly for 4 weeks\n" +
            "- Folic acid 5mg daily\n" +
            "- Then monthly B12 injections for maintenance",
            
            "Precautions:\n" +
            "- Monitor neurological symptoms\n" +
            "- Avoid nitrous oxide anesthesia\n" +
            "- Report any numbness/tingling",
            
            "Monitoring:\n" +
            "- CBC weekly for first month\n" +
            "- B12 and folate levels monthly\n" +
            "- Neurological assessment every visit"
        ]
    },
    
    'ACUTE INFECTION': {
        'conditions': {
            'LEUCOCYTE': {'condition': 'high', 'importance': 'primary'},
            'THROMBOCYTE': {'condition': 'high', 'importance': 'secondary'}
        },
        'treatments': [
            "Primary Treatment:\n" +
            "- Broad-spectrum antibiotics based on likely source:\n" +
            "- Amoxicillin-clavulanate 875/125mg twice daily\n" +
            "- Alternative: Azithromycin 500mg day 1, then 250mg days 2-5",
            
            "Precautions:\n" +
            "- Complete full course of antibiotics\n" +
            "- Monitor for allergic reactions\n" +
            "- Report fever > 101°F or worsening symptoms",
            
            "Monitoring:\n" +
            "- Daily temperature checks\n" +
            "- CBC with differential in 3-5 days\n" +
            "- CRP and ESR to track inflammation"
        ]
    },
    
    'SEVERE THROMBOCYTOPENIA': {
        'conditions': {
            'THROMBOCYTE': {'condition': 'low', 'importance': 'primary'}
        },
        'treatments': [
            "Primary Treatment:\n" +
            "- If autoimmune: Prednisone 1mg/kg/day\n" +
            "- Platelet transfusion if count < 10,000 or bleeding\n" +
            "- IVIG 1g/kg if severe autoimmune cause",
            
            "Precautions:\n" +
            "- Avoid aspirin and NSAIDs\n" +
            "- No contact sports or activities with bleeding risk\n" +
            "- Use soft toothbrush, electric razor only\n" +
            "- Report any unusual bruising or bleeding",
            
            "Monitoring:\n" +
            "- Daily platelet counts until stable\n" +
            "- Bleeding time and coagulation studies\n" +
            "- Regular blood pressure checks"
        ]
    },
    
    'POLYCYTHEMIA': {
        'conditions': {
            'HAEMATOCRIT': {'condition': 'high', 'importance': 'primary'},
            'HAEMOGLOBINS': {'condition': 'high', 'importance': 'primary'},
            'ERYTHROCYTE': {'condition': 'high', 'importance': 'secondary'}
        },
        'treatments': [
            "Primary Treatment:\n" +
            "- Therapeutic phlebotomy 500mL weekly\n" +
            "- Hydroxyurea 500mg twice daily if indicated\n" +
            "- Low-dose aspirin 81mg daily for clot prevention",
            
            "Precautions:\n" +
            "- Maintain adequate hydration\n" +
            "- Avoid smoking and alcohol\n" +
            "- Report headaches or visual changes\n" +
            "- Avoid high altitudes",
            
            "Monitoring:\n" +
            "- CBC weekly until stable\n" +
            "- Iron studies monthly\n" +
            "- JAK2 mutation testing\n" +
            "- Regular blood pressure monitoring"
        ]
    },
    
    'PANCYTOPENIA': {
        'conditions': {
            'HAEMOGLOBINS': {'condition': 'low', 'importance': 'primary'},
            'LEUCOCYTE': {'condition': 'low', 'importance': 'primary'},
            'THROMBOCYTE': {'condition': 'low', 'importance': 'primary'}
        },
        'treatments': [
            "Primary Treatment:\n" +
            "- Immediate hematology consultation\n" +
            "- Blood product support as needed\n" +
            "- G-CSF if neutropenic\n" +
            "- Bone marrow evaluation required",
            
            "Precautions:\n" +
            "- Strict infection precautions\n" +
            "- Avoid crowds and sick contacts\n" +
            "- No invasive procedures without coverage\n" +
            "- Bleeding precautions as for thrombocytopenia",
            
            "Monitoring:\n" +
            "- Daily CBC with differential\n" +
            "- Fever monitoring every 4 hours\n" +
            "- Weekly bone marrow recovery assessment\n" +
            "- Regular blood product support assessment"
        ]
    }
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
                # Convert sex to binary (0 for female, 1 for male)
                value = 1 if value.lower() in ['male', 'm'] else 0
            else:
                value = float(value)
            results[key] = value
    
    return results

# Feature engineering function exactly as used in training
def feature_engineering(df):
    # Ensure column order matches training
    columns = ['HAEMATOCRIT', 'HAEMOGLOBINS', 'ERYTHROCYTE', 'LEUCOCYTE',
               'THROMBOCYTE', 'MCH', 'MCHC', 'MCV', 'AGE', 'SEX_ENCODED']
    
    # Normalize borderline values more aggressively
    for param in ['HAEMATOCRIT', 'HAEMOGLOBINS', 'ERYTHROCYTE', 'LEUCOCYTE',
                  'THROMBOCYTE', 'MCH', 'MCHC', 'MCV']:
        if param in BLOOD_RANGES:
            ranges = BLOOD_RANGES[param]
            margin_low = ranges['low']['value'] * 0.90  # Increase margin to 10%
            margin_high = ranges['high']['value'] * 1.10  # Increase margin to 10%
            
            # If value is within 10% of normal range, normalize it to be well within range
            df.loc[df[param] >= margin_low, param] = df[param].apply(
                lambda x: max(x, ranges['low']['value'] + (ranges['high']['value'] - ranges['low']['value']) * 0.25) 
                if x < ranges['low']['value'] + (ranges['high']['value'] - ranges['low']['value']) * 0.1 else x
            )
            df.loc[df[param] <= margin_high, param] = df[param].apply(
                lambda x: min(x, ranges['high']['value'] - (ranges['high']['value'] - ranges['low']['value']) * 0.25)
                if x > ranges['high']['value'] - (ranges['high']['value'] - ranges['low']['value']) * 0.1 else x
            )
    
    # Create engineered features
    df['THROMBOCYTE_LEUCOCYTE_RATIO'] = df['THROMBOCYTE'] / (df['LEUCOCYTE'] + 1e-6)
    df['ERYTHROCYTE_LEUCOCYTE'] = df['ERYTHROCYTE'] * df['LEUCOCYTE']
    
    # Add engineered features to columns
    columns.extend(['THROMBOCYTE_LEUCOCYTE_RATIO', 'ERYTHROCYTE_LEUCOCYTE'])
    
    # Ensure all columns are present and in correct order
    return df[columns]

def prepare_input_data(data):
    """Prepare input data with correct feature names and order"""
    # Create DataFrame with proper column names
    df = pd.DataFrame([data])
    
    # Apply feature engineering
    df = feature_engineering(df)
    
    return df

# Define model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'final_model_pipeline.pkl')

# Global variable for model
model = None

def load_model():
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"Model not found at {MODEL_PATH}")
            # Try alternate path
            alternate_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'final_model_pipeline.pkl')
            if os.path.exists(alternate_path):
                print(f"Found model at alternate path: {alternate_path}")
                model = joblib.load(alternate_path)
            else:
                raise FileNotFoundError(f"Model not found at {MODEL_PATH} or {alternate_path}")
        else:
            print(f"Attempting to load model from: {MODEL_PATH}")
            model = joblib.load(MODEL_PATH)
        
        if model is None:
            raise ValueError("Model failed to load")
            
        # Verify model has predict method
        if not hasattr(model, 'predict'):
            raise AttributeError("Loaded model does not have predict method")
            
        print("Model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Directory contents: {os.listdir(os.path.dirname(MODEL_PATH))}")
        return False

# Load model when starting the server
if not load_model():
    raise RuntimeError("Failed to load the model. Cannot start server without a working model.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Create DataFrame with user input
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
        
        print("\n=== Prediction Details ===")
        print("Input Data:")
        for key, value in input_data.items():
            print(f"{key}: {value}")
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Get abnormal values before prediction
        abnormal_values = {}
        severe_conditions = 0
        for param, value in input_data.items():
            if param in BLOOD_RANGES:
                ranges = BLOOD_RANGES[param]
                # Add a 5% margin to the normal ranges
                margin_low = ranges['low']['value'] * 0.95  # Allow values 5% below the lower limit
                margin_high = ranges['high']['value'] * 1.05  # Allow values 5% above the upper limit
                
                if value < margin_low or value > margin_high:
                    abnormal_values[param] = {'value': value}
                    # Check for severe conditions with margins
                    if value < margin_low * 0.8 or value > margin_high * 1.2:
                        severe_conditions += 1
                        print(f"\nSevere condition detected in {param}: {value}")

        # Check for disease patterns
        detected_diseases = check_disease_patterns(input_data)
        if detected_diseases:
            print("\nDetected Diseases:")
            for disease in detected_diseases:
                print(f"- {disease}")
        
        # Make prediction, but override if severe conditions are present
        model_prediction = model.predict(input_df)
        print(f"\nModel's raw prediction: {'Inpatient' if model_prediction[0] == 1 else 'Outpatient'}")
        
        # Remove the override logic and use raw model prediction
        final_prediction = model_prediction[0]
        print(f"Final prediction: {'Inpatient' if final_prediction == 1 else 'Outpatient'}")
        print(f"Number of severe conditions: {severe_conditions}")
        print("============================\n")
        
        # Generate medical report
        medical_report = format_medical_report(
            final_prediction,
            input_data,
            detected_diseases,
            abnormal_values
        )
        
        # Convert prediction to meaningful response
        result = "Inpatient" if final_prediction == 1 else "Outpatient"
        
        return jsonify({
            'status': 'success',
            'prediction': result,
            'prediction_code': int(final_prediction),
            'medical_report': medical_report,
            'recommendations': format_recommendations(detected_diseases) if detected_diseases else None,
            'blood_ranges': BLOOD_RANGES
        })
    except Exception as e:
        print(f"\nError in prediction: {str(e)}\n")
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
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No file selected'
        }), 400
        
    if not allowed_file(file.filename):
        return jsonify({
            'status': 'error',
            'message': 'Invalid file type. Only PDF files are allowed.'
        }), 400
        
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"\n=== Processing PDF File: {filename} ===")
        
        # Extract text from PDF using pdfplumber
        text = ""
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        
        # Extract medical values from the text
        extracted_data = extract_medical_values(text)
        print("\nExtracted Values:")
        for key, value in extracted_data.items():
            print(f"{key}: {value}")
        
        # Check if all required fields were found
        required_fields = ['HAEMATOCRIT', 'HAEMOGLOBINS', 'ERYTHROCYTE', 'LEUCOCYTE',
                           'THROMBOCYTE', 'MCH', 'MCHC', 'MCV', 'AGE', 'SEX']
        missing_fields = [field for field in required_fields if field not in extracted_data]
        
        if missing_fields:
            print(f"\nMissing Fields: {', '.join(missing_fields)}")
            return jsonify({
                'status': 'error',
                'message': f'Missing required fields in PDF: {", ".join(missing_fields)}'
            }), 400
        
        # Create input data dictionary
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
        
        # Prepare input data with correct feature names and order
        input_df = prepare_input_data(input_data)
        
        # Get abnormal values before prediction
        abnormal_values = {}
        severe_conditions = 0
        for param, value in input_data.items():
            if param in BLOOD_RANGES:
                ranges = BLOOD_RANGES[param]
                # Add a 5% margin to the normal ranges
                margin_low = ranges['low']['value'] * 0.95  # Allow values 5% below the lower limit
                margin_high = ranges['high']['value'] * 1.05  # Allow values 5% above the upper limit
                
                if value < margin_low or value > margin_high:
                    abnormal_values[param] = {'value': value}
                    # Check for severe conditions with margins
                    if value < margin_low * 0.8 or value > margin_high * 1.2:
                        severe_conditions += 1
                        print(f"\nSevere condition detected in {param}: {value}")

        # Check for disease patterns
        detected_diseases = check_disease_patterns(input_data)
        if detected_diseases:
            print("\nDetected Diseases:")
            for disease in detected_diseases:
                print(f"- {disease}")
        
        # Make prediction, but override if severe conditions are present
        model_prediction = model.predict(input_df)
        print(f"\nModel's raw prediction: {'Inpatient' if model_prediction[0] == 1 else 'Outpatient'}")
        
        # Remove the override logic and use raw model prediction
        final_prediction = model_prediction[0]
        print(f"Final prediction: {'Inpatient' if final_prediction == 1 else 'Outpatient'}")
        print(f"Number of severe conditions: {severe_conditions}")
        print("============================\n")
        
        # Generate medical report
        medical_report = format_medical_report(
            final_prediction,
            input_data,
            detected_diseases,
            abnormal_values
        )
        
        # Convert prediction to meaningful response
        result = "Inpatient" if final_prediction == 1 else "Outpatient"
        
        # Clean up - remove uploaded file
        os.remove(filepath)
        
        return jsonify({
            'status': 'success',
            'prediction': result,
            'prediction_code': int(final_prediction),
            'extracted_values': extracted_data,
            'medical_report': medical_report,
            'recommendations': format_recommendations(detected_diseases) if detected_diseases else None
        })
        
    except Exception as e:
        # Clean up in case of error
        if os.path.exists(filepath):
            os.remove(filepath)
        print(f"\nError in file prediction: {str(e)}\n")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

def check_disease_patterns(values):
    """Check if values match known disease patterns"""
    detected_diseases = {}
    
    for disease, pattern in DISEASE_PATTERNS.items():
        matches = {'primary': 0, 'secondary': 0}
        required = {'primary': 0, 'secondary': 0}
        
        for param, criteria in pattern['conditions'].items():
            if criteria['importance'] == 'primary':
                required['primary'] += 1
            else:
                required['secondary'] += 1
                
            if param in values:
                ranges = BLOOD_RANGES[param]
                value = values[param]
                # Add a 5% margin to the normal ranges
                margin_low = ranges['low']['value'] * 0.95  # Allow values 5% below the lower limit
                margin_high = ranges['high']['value'] * 1.05  # Allow values 5% above the upper limit
                
                if criteria['condition'] == 'low' and value < margin_low:
                    matches[criteria['importance']] += 1
                elif criteria['condition'] == 'high' and value > margin_high:
                    matches[criteria['importance']] += 1
        
        # Disease is detected if all primary conditions are met and at least half of secondary
        if (matches['primary'] == required['primary'] and 
            (required['secondary'] == 0 or matches['secondary'] >= required['secondary'] / 2)):
            detected_diseases[disease] = pattern['treatments']
    
    return detected_diseases

def format_medical_report(prediction, values, detected_diseases, abnormal_values):
    """Generate a formatted medical report with modern shadcn-inspired styling"""
    report = []
    severity = get_severity_level(abnormal_values)
    
    # Header
    report.append('<div class="report-header">')
    report.append('<h2>Medical Laboratory Report</h2>')
    report.append(f'<p>{datetime.now().strftime("%B %d, %Y")}</p>')
    report.append('</div>')
    
    # Patient Information
    report.append('<div class="section">')
    report.append('<h3>Patient Information</h3>')
    report.append('<div class="findings">')
    report.append(f'<div class="finding"><strong>Age:</strong> {int(values["AGE"])} years</div>')
    report.append(f'<div class="finding"><strong>Sex:</strong> {"Male" if values["SEX_ENCODED"] == 1 else "Female"}</div>')
    report.append('</div>')
    report.append('</div>')
    
    # Blood Analysis Results
    report.append('<div class="section">')
    report.append('<h3>Blood Analysis Results</h3>')
    report.append('<div class="results-table">')
    report.append('<table>')
    report.append('<tr><th>Parameter</th><th>Value</th><th>Normal Range</th><th>Status</th></tr>')
    
    for param, value in values.items():
        if param in BLOOD_RANGES:
            ranges = BLOOD_RANGES[param]
            # Add a 5% margin to the normal ranges
            margin_low = ranges['low']['value'] * 0.95  # Allow values 5% below the lower limit
            margin_high = ranges['high']['value'] * 1.05  # Allow values 5% above the upper limit
            
            severity_text = get_parameter_severity(param, value)
            status_class = 'status-normal'
            if 'Severe' in severity_text:
                status_class = 'status-critical'
            elif severity_text != 'Normal':
                status_class = 'status-warning'
            
            value_text = f"{value:.1f}"
            normal_range = f"{ranges['low']['value']}-{ranges['high']['value']} {ranges['low']['unit']}"
            
            report.append(
                f'<tr>'
                f'<td>{param}</td>'
                f'<td>{value_text}</td>'
                f'<td>{normal_range}</td>'
                f'<td><span class="parameter-status {status_class}">{severity_text}</span></td>'
                f'</tr>'
            )
    
    report.append('</table>')
    report.append('</div>')
    report.append('</div>')
    
    # Clinical Interpretation
    report.append('<div class="section">')
    report.append('<h3>Clinical Interpretation</h3>')
    
    # List all abnormal findings
    abnormal_findings = []
    for param, value in values.items():
        if param in BLOOD_RANGES:
            ranges = BLOOD_RANGES[param]
            # Add a 5% margin to the normal ranges
            margin_low = ranges['low']['value'] * 0.95  # Allow values 5% below the lower limit
            margin_high = ranges['high']['value'] * 1.05  # Allow values 5% above the upper limit
            
            if value < margin_low:
                abnormal_findings.append({
                    'severity': 'critical' if value < margin_low * 0.8 else 'warning',
                    'text': f"{param} is Low ({value:.1f} {ranges['low']['unit']}): {ranges['conditions']['low']}"
                })
            elif value > margin_high:
                abnormal_findings.append({
                    'severity': 'critical' if value > margin_high * 1.2 else 'warning',
                    'text': f"{param} is High ({value:.1f} {ranges['high']['unit']}): {ranges['conditions']['high']}"
                })
    
    if abnormal_findings:
        report.append('<div class="findings">')
        for finding in abnormal_findings:
            icon = 'alert-triangle' if finding['severity'] == 'warning' else 'alert-circle'
            report.append(
                f'<div class="finding">'
                f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
                f'stroke="currentColor" stroke-width="2" class="status-{finding["severity"]}">'
            )
            if icon == 'alert-triangle':
                report.append('<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>')
                report.append('<line x1="12" y1="9" x2="12" y2="13"/>')
                report.append('<line x1="12" y1="17" x2="12.01" y2="17"/>')
            else:
                report.append('<circle cx="12" cy="12" r="10"/>')
                report.append('<line x1="12" y1="8" x2="12" y2="12"/>')
                report.append('<line x1="12" y1="16" x2="12.01" y2="16"/>')
            report.append('</svg>')
            report.append(f'<span>{finding["text"]}</span>')
            report.append('</div>')
        report.append('</div>')
    
    # Overall interpretation
    if prediction == 1:
        report.append(
            '<div class="warning">'  # Changed from urgent to warning for less severe cases
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
            'stroke="currentColor" stroke-width="2">'
            '<circle cx="12" cy="12" r="10"/>'
            '<line x1="12" y1="8" x2="12" y2="12"/>'
            '<line x1="12" y1="16" x2="12.01" y2="16"/>'
            '</svg>'
            '<div><strong>Medical Attention Recommended</strong><br>Consider inpatient care based on clinical judgment</div>'
            '</div>'
        )
        if detected_diseases:
            for disease in detected_diseases:
                report.append(f'<div class="finding">Detected {disease.lower()} requiring medical attention</div>')
    else:
        if detected_diseases:
            for disease in detected_diseases:
                report.append(
                    '<div class="warning">'
                    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
                    'stroke="currentColor" stroke-width="2">'
                    '<circle cx="12" cy="12" r="10"/>'
                    '<line x1="12" y1="8" x2="12" y2="12"/>'
                    '<line x1="12" y1="16" x2="12.01" y2="16"/>'
                    '</svg>'
                    f'<div>{disease.title()} detected—manageable with outpatient care</div>'
                    '</div>'
                )
        else:
            report.append(
                '<div class="finding" style="background: var(--muted)">'
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
                'stroke="currentColor" stroke-width="2" class="status-normal">'
                '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>'
                '<polyline points="22 4 12 14.01 9 11.01"/>'
                '</svg>'
                '<span>All parameters are within normal range or show minor variations</span>'
                '</div>'
            )
    report.append('</div>')
    
    return "\n".join(report)

def get_severity_level(abnormal_values):
    """Determine the severity level based on abnormal values"""
    if not abnormal_values:
        return "No"
    
    severe_count = 0
    for param, details in abnormal_values.items():
        if is_severe_abnormality(param, details['value']):
            severe_count += 1
    
    if severe_count >= 2:
        return "Severe"
    elif severe_count == 1:
        return "Moderate"
    return "Mild"

def get_parameter_severity(param, value):
    """Get severity description for a parameter"""
    ranges = BLOOD_RANGES[param]
    # Add a 5% margin to the normal ranges
    margin_low = ranges['low']['value'] * 0.95  # Allow values 5% below the lower limit
    margin_high = ranges['high']['value'] * 1.05  # Allow values 5% above the upper limit
    
    if value < margin_low:
        if value < ranges['low']['value'] * 0.8:
            return f"Severe {ranges['conditions']['low']}"
        return ranges['conditions']['low']
    elif value > margin_high:
        if value > ranges['high']['value'] * 1.2:
            return f"Severe {ranges['conditions']['high']}"
        return ranges['conditions']['high']
    return "Normal"

def is_severe_abnormality(param, value):
    """Check if the abnormality is severe"""
    ranges = BLOOD_RANGES[param]
    # Add a 5% margin to the normal ranges
    margin_low = ranges['low']['value'] * 0.95  # Allow values 5% below the lower limit
    margin_high = ranges['high']['value'] * 1.05  # Allow values 5% above the upper limit
    return (value < margin_low * 0.8 or 
            value > margin_high * 1.2)

def format_recommendations(detected_diseases):
    """Format disease recommendations into HTML"""
    if not detected_diseases:
        return None
        
    recommendations = []
    for disease, treatments in detected_diseases.items():
        recommendations.append(f'<div class="disease-section">')
        recommendations.append(f'<h4>{disease}</h4>')
        
        for treatment in treatments:
            if "Primary Treatment:" in treatment:
                recommendations.append('<div class="treatment-subsection">')
                recommendations.append('<h5>Primary Treatment</h5>')
                recommendations.append('<ul class="treatment-list">')
                for line in treatment.split('\n')[1:]:  # Skip the header
                    if line.strip():
                        recommendations.append(f'<li>{line.strip("- ")}</li>')
                recommendations.append('</ul>')
                recommendations.append('</div>')
            
            elif "Precautions:" in treatment:
                recommendations.append('<div class="treatment-subsection">')
                recommendations.append('<h5>Precautions</h5>')
                recommendations.append('<ul class="treatment-list">')
                for line in treatment.split('\n')[1:]:  # Skip the header
                    if line.strip():
                        recommendations.append(f'<li>{line.strip("- ")}</li>')
                recommendations.append('</ul>')
                recommendations.append('</div>')
            
            elif "Monitoring:" in treatment:
                recommendations.append('<div class="treatment-subsection">')
                recommendations.append('<h5>Monitoring Plan</h5>')
                recommendations.append('<ul class="treatment-list">')
                for line in treatment.split('\n')[1:]:  # Skip the header
                    if line.strip():
                        recommendations.append(f'<li>{line.strip("- ")}</li>')
                recommendations.append('</ul>')
                recommendations.append('</div>')
        
        recommendations.append('</div>')
    
    return '\n'.join(recommendations)

def extract_numbers(text):
    """Extract all numbers from the given text using regex."""
    return re.findall(r'\d+\.?\d*', text)

if __name__ == '__main__':
    app.run(debug=True, port=5000) 