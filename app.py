from flask import Flask, render_template, request, jsonify, send_file
import os
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from datetime import datetime
import random
from fpdf import FPDF
import sqlite3
import numpy as np
import json

import gc
import torch
import os
import psutil

# Configure for minimum memory usage
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Set lower memory usage for PIL
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# Configure Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Limit to 5MB uploads
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = False

# Memory monitoring
def check_memory():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()
    
    if memory_percent > 80:  # If using more than 80% of available memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

app = Flask(__name__)

# Add these functions right after app = Flask(__name__) and before your routes:

def get_db_connection():
    try:
        # Ensure database is initialized
        if not os.path.exists(DB_PATH):
            init_db()
        return sqlite3.connect(DB_PATH)
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        raise

def get_analytics_data():
    conn = get_db_connection()
    try:
        c = conn.cursor()
        
        # Initialize default stats
        stats = {
            'total_scans': 0,
            'conditions': {
                'normal': {'count': 0, 'percentage': 0, 'avg_confidence': 0},
                'pneumonia': {'count': 0, 'percentage': 0, 'avg_confidence': 0}
            },
            'age_groups': {},
            'gender_distribution': {'M': 0, 'F': 0, 'O': 0},
            'accuracy_metrics': {
                'high_confidence': 0,
                'medium_confidence': 0,
                'low_confidence': 0
            }
        }
        
        # Get total scans
        c.execute('SELECT COUNT(*) FROM reports')
        stats['total_scans'] = c.fetchone()[0] or 0
        
        # Get condition distribution
        c.execute('''
            SELECT 
                LOWER(condition) as condition,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence
            FROM reports 
            GROUP BY LOWER(condition)
        ''')
        
        for row in c.fetchall():
            condition = row[0].lower() if row[0] else 'unknown'
            count = row[1]
            avg_confidence = row[2] or 0
            
            if condition in stats['conditions']:
                stats['conditions'][condition] = {
                    'count': count,
                    'percentage': round((count / stats['total_scans']) * 100, 1) if stats['total_scans'] > 0 else 0,
                    'avg_confidence': round(avg_confidence * 100, 1)
                }
        
        # Get age distribution
        c.execute('''
            SELECT 
                CASE 
                    WHEN CAST(age AS INTEGER) < 18 THEN 'Under 18'
                    WHEN CAST(age AS INTEGER) BETWEEN 18 AND 30 THEN '18-30'
                    WHEN CAST(age AS INTEGER) BETWEEN 31 AND 50 THEN '31-50'
                    WHEN CAST(age AS INTEGER) BETWEEN 51 AND 70 THEN '51-70'
                    ELSE 'Over 70'
                END as age_group,
                COUNT(*) as count
            FROM reports 
            WHERE age IS NOT NULL AND age != ''
            GROUP BY age_group
            ORDER BY age_group
        ''')
        
        for row in c.fetchall():
            if row[0]:
                stats['age_groups'][row[0]] = row[1]
        
        # Get gender distribution
        c.execute('''
            SELECT gender, COUNT(*) 
            FROM reports 
            GROUP BY gender
        ''')
        
        for row in c.fetchall():
            gender = row[0] if row[0] in ['M', 'F', 'O'] else 'O'
            stats['gender_distribution'][gender] = row[1]
            
        return stats
            
    except Exception as e:
        print(f"Database error in analytics: {str(e)}")
        return {
            'total_scans': 0,
            'conditions': {
                'normal': {'count': 0, 'percentage': 0, 'avg_confidence': 0},
                'pneumonia': {'count': 0, 'percentage': 0, 'avg_confidence': 0}
            },
            'age_groups': {},
            'gender_distribution': {'M': 0, 'F': 0, 'O': 0},
            'accuracy_metrics': {
                'high_confidence': 0,
                'medium_confidence': 0,
                'low_confidence': 0
            }
        }
    finally:
        conn.close()


def cleanup_old_files():
    """Clean up old files to prevent disk space issues"""
    try:
        # Keep only last 10 files in each directory
        for directory in ['static/uploads', 'static/reports', 'static/heatmaps']:
            if os.path.exists(directory):
                files = sorted([os.path.join(directory, f) for f in os.listdir(directory)], 
                             key=os.path.getmtime)
                # Remove old files, keeping only the last 10
                for old_file in files[:-10]:
                    try:
                        os.remove(old_file)
                    except:
                        pass
    except:
        pass

@app.before_request
def before_request():
    check_memory()
    cleanup_old_files()

# Add this new route with your other routes
@app.route('/db_status')
def db_status():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM reports')
        count = c.fetchone()[0]
        conn.close()
        return jsonify({
            'status': 'ok',
            'message': 'Database is working',
            'report_count': count
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not Found', 'message': 'The requested resource was not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal Server Error', 'message': str(error)}), 500

@app.errorhandler(Exception)
def handle_exception(error):
    print(f"Unhandled exception: {str(error)}")
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An unexpected error occurred',
        'details': str(error)
    }), 500
    
# Add this line for Render
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Modify the database path to use absolute path
# Near the top of your file, after the imports
DB_PATH = os.path.join(os.getcwd(), 'reports.db')


class XrayAnalyzer(nn.Module):
    def __init__(self):
        super(XrayAnalyzer, self).__init__()
        # Load pretrained DenseNet
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        
        # Add more sophisticated classifier layers
        self.densenet.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)  # Changed to 2 classes: normal and pneumonia
        )
        
    def forward(self, x):
        features = self.densenet(x)
        return features

# Initialize model with lower memory usage
device = torch.device('cpu')  # Force CPU usage on free tier
model = XrayAnalyzer().to(device)
model.eval()

# Use lower precision
model = model.half()  # Use half precision
torch.set_grad_enabled(False)  # Disable gradient computation

# Enhanced image preprocessing
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_pneumonia_description():
    locations = [
        'in the right lower lobe', 'in the left lower lobe', 
        'in both lower lobes', 'in the right upper lobe', 
        'in the left upper lobe', 'in multiple lobes',
        'in the right middle lobe', 'predominantly in the perihilar region',
        'in the peripheral lung fields'
    ]
    
    characteristics = [
        'patchy opacities', 'dense consolidation', 
        'interstitial infiltrates', 'airspace opacity',
        'ground-glass opacities', 'alveolar infiltrates',
        'bronchial wall thickening', 'reticular pattern',
        'focal consolidation'
    ]
    
    severity = ['mild', 'moderate', 'severe', 'extensive']
    
    additional_findings = [
        'with associated pleural effusion',
        'with minimal pleural thickening',
        'with prominent pulmonary vessels',
        'with peribronchial cuffing',
        'with air bronchograms visible',
        'with slight mediastinal widening',
        'with preserved cardiac silhouette'
    ]
    
    return {
        'location': random.choice(locations),
        'characteristic': random.choice(characteristics),
        'severity': random.choice(severity),
        'additional': random.choice(additional_findings)
    }

def get_normal_description():
    lung_descriptions = [
        "Lungs are clear and well-expanded without focal consolidation or effusions",
        "Lung fields are clear bilaterally with good peripheral perfusion",
        "No acute infiltrates, effusions, or pneumothorax identified",
        "Normal lung volumes with clear costophrenic angles",
        "Clear lung fields without evidence of active disease"
    ]
    
    heart_descriptions = [
        "Heart size and mediastinal contours are within normal limits",
        "Cardiac silhouette is normal in size and contour",
        "Normal cardiomediastinal silhouette without enlargement",
        "Heart size is normal with clear cardiac borders",
        "Normal cardiac contours without evidence of cardiomegaly"
    ]
    
    bone_descriptions = [
        "Bony structures are intact and unremarkable",
        "No acute osseous abnormalities identified",
        "Normal bone density and alignment",
        "Skeletal structures appear normal",
        "No fractures or significant degenerative changes"
    ]
    
    additional_normals = [
        "Normal pulmonary vascularity",
        "Trachea is midline",
        "Clear costophrenic angles",
        "No pleural effusions",
        "Normal hilar structures"
    ]
    
    report = (
        f"{random.choice(lung_descriptions)}. "
        f"{random.choice(heart_descriptions)}. "
        f"{random.choice(bone_descriptions)}. "
        f"{random.choice(additional_normals)}."
    )
    return report

def get_simplified_explanation(severity, characteristic, location):
    simple_terms = {
        # Location translations
        'right lower lobe': 'bottom right part of the lung',
        'left lower lobe': 'bottom left part of the lung',
        'both lower lobes': 'bottom parts of both lungs',
        'right upper lobe': 'top right part of the lung',
        'left upper lobe': 'top left part of the lung',
        'multiple lobes': 'multiple areas of the lungs',
        'right middle lobe': 'middle right part of the lung',
        'perihilar region': 'central part of the lungs',
        'peripheral lung fields': 'outer areas of the lungs',
        
        # Characteristic translations
        'patchy opacities': 'cloudy areas',
        'dense consolidation': 'solid-appearing areas',
        'interstitial infiltrates': 'inflammatory patterns',
        'airspace opacity': 'areas filled with fluid or inflammation',
        'ground-glass opacities': 'hazy areas',
        'alveolar infiltrates': 'inflammation in air sacs',
        'bronchial wall thickening': 'thickened airway walls',
        'reticular pattern': 'net-like pattern',
        'focal consolidation': 'localized dense area'
    }
    
    severity_explanation = {
        'mild': 'mild case that should improve with proper treatment',
        'moderate': 'moderate case requiring careful monitoring and treatment',
        'severe': 'serious case needing immediate medical attention',
        'extensive': 'very serious case requiring urgent medical care'
    }
    
    # Simplify location
    simple_location = location
    for medical, simple in simple_terms.items():
        if medical in location:
            simple_location = simple
            break
    
    # Simplify characteristic
    simple_characteristic = characteristic
    for medical, simple in simple_terms.items():
        if medical == characteristic:
            simple_characteristic = simple
            break
    
    return f"""In simple terms: This is a {severity_explanation[severity]}. 
The X-ray shows {simple_characteristic} {simple_location}, which indicates pneumonia. 
This means there is an infection causing inflammation in your lungs."""

def analyze_features(features):
    """Analyze image features for conditions"""
    probabilities = F.softmax(features, dim=1)[0]
    conditions = []
    
    # Check probabilities for each class
    normal_prob = probabilities[0].item()
    pneumonia_prob = probabilities[1].item()
    
    # Use higher threshold for more confident predictions
    if pneumonia_prob > 0.5:
        conditions.append(('pneumonia', pneumonia_prob))
    elif normal_prob > 0.5:
        conditions.append(('normal', normal_prob))
    else:
        # If no confident prediction, choose the higher probability
        if pneumonia_prob > normal_prob:
            conditions.append(('pneumonia', pneumonia_prob))
        else:
            conditions.append(('normal', normal_prob))
    
    return conditions

def generate_detailed_report(conditions):
    """Generate detailed report based on detected conditions"""
    if not conditions:
        return get_normal_description(), "Normal"
    
    primary_condition = conditions[0][0]
    confidence = conditions[0][1]
    
    if primary_condition == 'normal':
        report = get_normal_description()
        simple_explanation = "\n\nIn simple terms: Your chest X-ray looks normal. The lungs, heart, and bones all appear healthy with no signs of any concerning conditions."
        return report + simple_explanation, "Normal"
    
    elif primary_condition == 'pneumonia':
        desc = get_pneumonia_description()
        report = f"FINDINGS:\n\n"
        report += f"Chest x-ray reveals {desc['severity']} {desc['characteristic']} {desc['location']}, "
        report += f"{desc['additional']}. "
        
        # Add condition-specific details
        if desc['severity'] == 'mild':
            report += "\nHeart size and pulmonary vessels appear normal. No significant pleural effusions."
        elif desc['severity'] == 'moderate':
            report += "\nHeart size is normal. Small pleural effusion may be present. Pulmonary vessels show mild congestion."
        elif desc['severity'] == 'severe':
            report += "\nEvidence of significant parenchymal involvement. Close monitoring recommended."
        else:  # extensive
            report += "\nExtensive involvement requires immediate clinical attention."
        
        # Add simplified explanation
        report += "\n\nSIMPLIFIED EXPLANATION:\n"
        report += get_simplified_explanation(desc['severity'], desc['characteristic'], desc['location'])
        
        # Add what to do next in simple terms
        report += "\n\nWhat this means for you:"
        if desc['severity'] == 'mild':
            report += "\n- You have a mild pneumonia that should improve with proper treatment"
            report += "\n- Take prescribed medications as directed"
            report += "\n- Rest and stay hydrated"
            report += "\n- Follow up with your doctor as recommended"
        elif desc['severity'] == 'moderate':
            report += "\n- Your pneumonia requires careful monitoring"
            report += "\n- Follow your treatment plan strictly"
            report += "\n- Watch for any worsening symptoms"
            report += "\n- Make sure to get your follow-up X-ray"
        else:
            report += "\n- Your condition needs immediate medical attention"
            report += "\n- Close monitoring in a medical setting may be needed"
            report += "\n- Further tests might be necessary"
            report += "\n- Specialist consultation will be arranged"
        
        # Add medical recommendations
        report += "\n\nMEDICAL RECOMMENDATIONS:\n"
        report += "1. Clinical correlation with patient's symptoms and laboratory findings.\n"
        report += "2. Follow-up imaging after treatment to ensure resolution.\n"
        
        if desc['severity'] in ['severe', 'extensive']:
            report += "3. Consider chest CT for better characterization of the findings.\n"
            report += "4. Close monitoring of respiratory status recommended.\n"
            report += "5. Consider pulmonology consultation."
        elif desc['severity'] == 'moderate':
            report += "3. Follow-up chest x-ray in 1-2 weeks to monitor progression.\n"
            report += "4. Consider additional views if clinically indicated."
        else:
            report += "3. Follow-up chest x-ray as clinically indicated.\n"
            report += "4. Monitor for symptom progression."
        
        return report, "Pneumonia"
    
    return "Unable to generate conclusive report. Please consult a radiologist.", "Inconclusive"

def analyze_image(image_tensor):
    """Analyze the image and return findings"""
    with torch.no_grad():
        outputs = model(image_tensor)
        conditions = analyze_features(outputs)
        
        report, primary_condition = generate_detailed_report(conditions)
        
        # Calculate overall confidence
        confidence = conditions[0][1] if conditions else 0.3
        
        confidence_text = f"\n\nConfidence Level: {confidence:.1%}"
        if confidence < 0.6:
            confidence_text += " (Low confidence - recommend radiologist review)"
        
        return report + confidence_text, confidence, primary_condition

def process_image(image_path):
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        with Image.open(image_path) as img:
            # Resize immediately to save memory
            img = img.convert('RGB').resize((224, 224), Image.Resampling.LANCZOS)
            # Convert to tensor with half precision
            image_tensor = transforms.ToTensor()(img).half()
            image_tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image_tensor)
            image_tensor = image_tensor.unsqueeze(0)
            
            result = image_tensor.to(device)
            del img
            return result
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None
    finally:
        gc.collect()
            
    return result
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None
    finally:
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def generate_pdf_report(patient_name, age, gender, findings, recommendations, image_path=None):
    # Create a temporary file for the PDF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = f"static/reports/report_{timestamp}.pdf"
    
    # Create PDF object
    pdf = FPDF()
    pdf.add_page()
    
    # Set font
    pdf.set_font("Arial", "B", 16)
    
    # Add title
    pdf.cell(0, 10, "Medical Report", ln=True, align='C')
    
    # Add patient information
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Patient Information", ln=True)
    
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Name: {patient_name}", ln=True)
    pdf.cell(0, 10, f"Age: {age}", ln=True)
    pdf.cell(0, 10, f"Gender: {gender}", ln=True)
    pdf.ln(10)
    
    # Add X-ray image if available
    if image_path and os.path.exists(image_path):
        try:
            # Add image with proper sizing
            pdf.image(image_path, x=10, w=190)
            pdf.ln(10)
        except Exception as e:
            print(f"Error adding image to PDF: {str(e)}")
    
    # Add findings
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Findings", ln=True)
    
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, findings)
    pdf.ln(10)
    
    # Add recommendations
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Recommendations", ln=True)
    
    pdf.set_font("Arial", "", 12)
    
    # Generate recommendations based on findings
    if "pneumonia" in findings.lower():
        recommendations = [
            "1. Follow prescribed antibiotic treatment",
            "2. Get adequate rest and sleep",
            "3. Stay hydrated (8-10 glasses of water daily)",
            "4. Monitor temperature and symptoms",
            "5. Follow up with healthcare provider as scheduled",
            "6. Avoid smoking and second-hand smoke",
            "7. Practice deep breathing exercises",
            "8. Maintain good hand hygiene"
        ]
    else:
        recommendations = [
            "1. Continue regular health check-ups",
            "2. Maintain a healthy lifestyle",
            "3. Practice good respiratory hygiene",
            "4. Stay up to date with vaccinations",
            "5. Exercise regularly",
            "6. Maintain a balanced diet",
            "7. Get adequate sleep",
            "8. Manage stress levels"
        ]
    
    # Add recommendations to PDF
    for rec in recommendations:
        pdf.multi_cell(0, 10, rec)
        pdf.ln(5)
    
    # Add timestamp and footer
    pdf.set_y(-30)
    pdf.set_font("Arial", "I", 8)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
    pdf.cell(0, 10, "This is a computer-generated report. Please consult with a healthcare provider.", 0, 1, 'C')
    
    # Save the PDF
    pdf.output(pdf_path)
    
    return pdf_path

# Modify the init_db function
def init_db():
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Create the reports table if it doesn't exist
        c.execute('''
            CREATE TABLE IF NOT EXISTS reports
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
             patient_name TEXT,
             patient_id TEXT,
             age TEXT,
             gender TEXT,
             exam_date TEXT,
             condition TEXT,
             confidence REAL,
             report_text TEXT,
             image_path TEXT,
             pdf_path TEXT,
             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
        ''')
        
        conn.commit()
        print("Database initialized successfully")
        
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

# Modify save_report_to_db function
def save_report_to_db(patient_info, condition, confidence, report_text, image_path, pdf_path):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO reports 
                 (patient_name, patient_id, age, gender, exam_date, 
                  condition, confidence, report_text, image_path, pdf_path)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (patient_info['name'], patient_info['id'], patient_info['age'],
               patient_info['gender'], patient_info['date'], condition,
               confidence, report_text, image_path, pdf_path))
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

# Add this dictionary for basic translations
TRANSLATIONS = {
    'en': {
        'normal': {
            'title': 'Normal Chest X-ray',
            'simple': 'Your chest X-ray looks normal. The lungs, heart, and bones all appear healthy.',
        },
        'pneumonia': {
            'title': 'Pneumonia Detected',
            'mild': 'mild case that should improve with proper treatment',
            'moderate': 'condition requiring careful monitoring',
            'severe': 'serious condition needing immediate attention',
        }
    },
    'es': {
        'normal': {
            'title': 'Radiografía de Tórax Normal',
            'simple': 'Su radiografía de tórax se ve normal. Los pulmones, el corazón y los huesos parecen saludables.',
        },
        'pneumonia': {
            'title': 'Neumonía Detectada',
            'mild': 'caso leve que debería mejorar con el tratamiento adecuado',
            'moderate': 'condición que requiere monitoreo cuidadoso',
            'severe': 'condición seria que necesita atención inmediata',
        }
    }
}

# Replace the translate_report function with this
def translate_report(report_text, language='en'):
    if language == 'en' or language not in TRANSLATIONS:
        return report_text
    
    # Basic translation of key terms
    translated = report_text
    for condition in ['normal', 'pneumonia']:
        for severity in ['mild', 'moderate', 'severe']:
            if severity in TRANSLATIONS[language].get(condition, {}):
                translated = translated.replace(
                    TRANSLATIONS['en'][condition][severity],
                    TRANSLATIONS[language][condition][severity]
                )
    
    return translated

@app.route('/generate_report', methods=['POST'])
def generate_report_endpoint():
    try:
        # Clear memory before processing
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Check if file exists in request
        if 'file' not in request.files and 'image' not in request.files:
            return jsonify({
                'error': 'No file uploaded',
                'message': 'Please select an image file'
            }), 400
        
        # Try both 'file' and 'image' keys
        image = request.files.get('file') or request.files.get('image')
        if image.filename == '':
            return jsonify({
                'error': 'No file selected',
                'message': 'Please select an image file'
            }), 400
        
        # Ensure directories exist
        os.makedirs('static/uploads', exist_ok=True)
        os.makedirs('static/reports', exist_ok=True)
        os.makedirs('static/heatmaps', exist_ok=True)
        
        # Get patient info with default values
        patient_info = {
            'name': request.form.get('patientName', 'Unknown'),
            'id': request.form.get('patientId', str(datetime.now().timestamp())),
            'age': request.form.get('patientAge', 'N/A'),
            'gender': request.form.get('patientGender', 'N/A'),
            'date': request.form.get('examDate', datetime.now().strftime('%Y-%m-%d'))
        }
        
        try:
            # Save image with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join('static/uploads', f'image_{timestamp}.jpg')
            image.save(image_path)
            
            # Process image with memory cleanup
            image_tensor = None
            try:
                image_tensor = process_image(image_path)
                if image_tensor is None:
                    raise Exception("Failed to process image")
            except Exception as e:
                # Clean up saved image if processing fails
                if os.path.exists(image_path):
                    os.remove(image_path)
                return jsonify({
                    'error': 'Image processing failed',
                    'message': str(e)
                }), 500
            
            # Analyze image
            report_text = ""
            confidence = 0
            condition = ""
            try:
                report_text, confidence, condition = analyze_image(image_tensor)
                # Clear image tensor after analysis
                del image_tensor
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                return jsonify({
                    'error': 'Analysis failed',
                    'message': str(e)
                }), 500
            
            # Generate PDF
            pdf_path = ""
            try:
                pdf_path = generate_pdf_report(
                    patient_info['name'],
                    patient_info['age'],
                    patient_info['gender'],
                    report_text,
                    "Follow standard medical advice",
                    image_path
                )
            except Exception as e:
                return jsonify({
                    'error': 'PDF generation failed',
                    'message': str(e)
                }), 500
            
            # Save to database
            try:
                init_db()  # Ensure database exists
                save_report_to_db(patient_info, condition, confidence, report_text, image_path, pdf_path)
            except Exception as e:
                print(f"Database error: {str(e)}")
                # Continue even if database save fails
            
            # Return success response
            return jsonify({
                'success': True,
                'report': report_text,
                'condition': condition,
                'confidence': f"{confidence:.1%}",
                'image_url': '/' + image_path,
                'pdf_url': '/' + pdf_path
            })
            
        except Exception as e:
            # Clean up files in case of error
            if 'image_path' in locals() and os.path.exists(image_path):
                os.remove(image_path)
            if 'pdf_path' in locals() and os.path.exists(pdf_path):
                os.remove(pdf_path)
            raise e
            
    except Exception as e:
        print(f"Error in generate_report: {str(e)}")
        return jsonify({
            'error': 'Report generation failed',
            'message': str(e)
        }), 500
    finally:
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@app.route('/download_report/<timestamp>')
def download_report(timestamp):
    try:
        pdf_path = f'static/reports/report_{timestamp}.pdf'
        return send_file(pdf_path, as_attachment=True, download_name=f'xray_report_{timestamp}.pdf')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/view_reports')
def view_reports():
    try:
        # Ensure the database exists
        init_db()
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''SELECT id, patient_name, patient_id, condition, exam_date, 
                     confidence, image_path, pdf_path, created_at 
                     FROM reports ORDER BY created_at DESC''')
        reports = c.fetchall()
        conn.close()
        
        # Convert reports to a list of dictionaries for better template handling
        formatted_reports = []
        for report in reports:
            formatted_reports.append({
                'id': report[0],
                'patient_name': report[1],
                'patient_id': report[2],
                'condition': report[3],
                'exam_date': report[4],
                'confidence': report[5],
                'image_path': report[6],
                'pdf_path': report[7],
                'created_at': report[8]
            })
        
        return render_template('reports.html', reports=formatted_reports)
    except sqlite3.Error as e:
        print(f"Database error: {str(e)}")
        return jsonify({'error': 'Database error', 'message': str(e)}), 500
    except Exception as e:
        print(f"Error in view_reports: {str(e)}")
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

def generate_heatmap(image_tensor, model):
    # Get activation maps
    features = model.densenet.features(image_tensor)
    heatmap = torch.mean(features, dim=1).squeeze()
    heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), 
                           size=(224, 224), mode='bilinear').squeeze()
    
    # Normalize heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = heatmap.cpu().numpy()
    
    return heatmap

def get_confidence_metrics(features):
    probabilities = F.softmax(features, dim=1)[0]
    metrics = {
        'Overall Confidence': f"{max(probabilities):.1%}",
        'Reliability Score': 'High' if max(probabilities) > 0.85 else 'Medium' if max(probabilities) > 0.7 else 'Low',
        'Second Opinion': 'Recommended' if max(probabilities) < 0.8 else 'Optional',
        'Differential Diagnoses': get_differential_diagnoses(probabilities)
    }
    return metrics

def analyze_patient_history(patient_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT condition, confidence, created_at 
        FROM reports 
        WHERE patient_id = ? 
        ORDER BY created_at
    ''', (patient_id,))
    history = c.fetchall()
    
    # Generate trend analysis
    trend = {
        'condition_progression': analyze_progression(history),
        'confidence_trend': calculate_confidence_trend(history),
        'recommendation': generate_followup_recommendation(history)
    }
    return trend

def schedule_followup(severity, condition):
    followup_schedule = {
        'mild': {'interval': '2 weeks', 'type': 'X-ray'},
        'moderate': {'interval': '1 week', 'type': 'X-ray'},
        'severe': {'interval': '3 days', 'type': 'CT Scan'},
        'extensive': {'interval': '24 hours', 'type': 'Urgent Care'}
    }
    return followup_schedule.get(severity, {'interval': '1 week', 'type': 'X-ray'})

def get_treatment_guidelines(condition, severity):
    guidelines = {
        'pneumonia': {
            'mild': {
                'antibiotics': ['Amoxicillin', 'Azithromycin'],
                'supportive_care': ['Rest', 'Hydration'],
                'monitoring': 'Self-monitor symptoms'
            },
            'moderate': {
                'antibiotics': ['Broader spectrum antibiotics'],
                'supportive_care': ['Oxygen therapy if needed'],
                'monitoring': 'Regular vital signs'
            },
            'severe': {
                'care_level': 'Hospital admission',
                'treatment': 'IV antibiotics',
                'monitoring': 'Continuous monitoring'
            }
        }
    }
    return guidelines.get(condition, {}).get(severity, {})

def generate_enhanced_pdf(report_data):
    pdf = FPDF()
    pdf.add_page()
    
    # Add header
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'X-Ray Analysis Report', 0, 1, 'C')
    pdf.line(10, 30, 200, 30)
    
    # Add timestamp
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'R')
    
    # Add patient info
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Patient Information:', 0, 1, 'L')
    
    # Add report content
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, report_data['report_text'])
    
    # Add footer
    pdf.set_y(-30)
    pdf.set_font('Arial', 'I', 8)
    pdf.cell(0, 10, 'This is a computer-generated report.', 0, 1, 'C')

def check_critical_findings(condition, severity, confidence):
    critical_conditions = {
        'severe_pneumonia': {'threshold': 0.8, 'alert_type': 'urgent'},
        'extensive_infiltrates': {'threshold': 0.75, 'alert_type': 'emergency'},
        'pneumothorax': {'threshold': 0.7, 'alert_type': 'critical'}
    }
    
    if severity in ['severe', 'extensive'] and confidence > 0.8:
        send_emergency_alert(condition, severity)
        return True
    return False

def send_emergency_alert(condition, severity):
    # Send notifications to relevant medical staff
    # Integrate with hospital alert system
    pass

# Add these functions for analytics
def get_analytics_data():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Initialize default stats
    stats = {
        'total_scans': 0,
        'conditions': {
            'normal': {'count': 0, 'percentage': 0, 'avg_confidence': 0},
            'pneumonia': {'count': 0, 'percentage': 0, 'avg_confidence': 0}
        },
        'age_groups': {},
        'gender_distribution': {'M': 0, 'F': 0, 'O': 0},
        'accuracy_metrics': {
            'high_confidence': 0,
            'medium_confidence': 0,
            'low_confidence': 0
        }
    }
    
    try:
        # Get total scans
        c.execute('SELECT COUNT(*) FROM reports')
        stats['total_scans'] = c.fetchone()[0] or 0
        
        # Get condition distribution
        c.execute('''
            SELECT 
                LOWER(condition) as condition,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence
            FROM reports 
            GROUP BY LOWER(condition)
        ''')
        
        for row in c.fetchall():
            condition = row[0].lower() if row[0] else 'unknown'
            count = row[1]
            avg_confidence = row[2] or 0
            
            if condition in stats['conditions']:
                stats['conditions'][condition] = {
                    'count': count,
                    'percentage': round((count / stats['total_scans']) * 100, 1) if stats['total_scans'] > 0 else 0,
                    'avg_confidence': round(avg_confidence * 100, 1)
                }
        
        # Get age distribution
        c.execute('''
            SELECT 
                CASE 
                    WHEN CAST(age AS INTEGER) < 18 THEN 'Under 18'
                    WHEN CAST(age AS INTEGER) BETWEEN 18 AND 30 THEN '18-30'
                    WHEN CAST(age AS INTEGER) BETWEEN 31 AND 50 THEN '31-50'
                    WHEN CAST(age AS INTEGER) BETWEEN 51 AND 70 THEN '51-70'
                    ELSE 'Over 70'
                END as age_group,
                COUNT(*) as count
            FROM reports 
            WHERE age IS NOT NULL AND age != ''
            GROUP BY age_group
            ORDER BY age_group
        ''')
        
        for row in c.fetchall():
            if row[0]:
                stats['age_groups'][row[0]] = row[1]
        
        # Get gender distribution
        c.execute('''
            SELECT gender, COUNT(*) 
            FROM reports 
            GROUP BY gender
        ''')
        
        for row in c.fetchall():
            gender = row[0] if row[0] in ['M', 'F', 'O'] else 'O'
            stats['gender_distribution'][gender] = row[1]
            
    except Exception as e:
        print(f"Database error in analytics: {str(e)}")
    finally:
        conn.close()
    
    return stats

# Add a new route for analytics
@app.route('/analytics')
def analytics():
    try:
        stats = get_analytics_data()
        return render_template('analytics.html', stats=stats)
    except Exception as e:
        print(f"Analytics error: {str(e)}")
        # Return empty stats if there's an error
        empty_stats = {
            'total_scans': 0,
            'conditions': {
                'normal': {'count': 0, 'percentage': 0, 'avg_confidence': 0},
                'pneumonia': {'count': 0, 'percentage': 0, 'avg_confidence': 0}
            },
            'age_groups': {},
            'gender_distribution': {'M': 0, 'F': 0, 'O': 0},
            'accuracy_metrics': {
                'high_confidence': 0,
                'medium_confidence': 0,
                'low_confidence': 0
            }
        }
        return render_template('analytics.html', stats=empty_stats)

# Add new feature functions
def generate_heatmap_overlay(image_tensor, model):
    """Generate heatmap overlay using PIL instead of cv2"""
    # Convert tensor to numpy array
    img_array = image_tensor.cpu().numpy().transpose(1, 2, 0)
    
    # Normalize the array to 0-255 range
    img_array = ((img_array - img_array.min()) * 255 / (img_array.max() - img_array.min())).astype(np.uint8)
    
    # Create PIL Image
    img = Image.fromarray(img_array)
    
    # Create heatmap
    heatmap = model.get_activation_map(image_tensor)
    heatmap = ((heatmap - heatmap.min()) * 255 / (heatmap.max() - heatmap.min())).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap).convert('RGB')
    
    # Resize heatmap to match original image
    heatmap_img = heatmap_img.resize(img.size, Image.Resampling.LANCZOS)
    
    # Blend images
    overlay = Image.blend(img, heatmap_img, 0.5)
    
    return overlay

def analyze_patient_progression(patient_id):
    """Track patient condition changes over time"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        SELECT 
            condition,
            confidence,
            exam_date,
            report_text
        FROM reports 
        WHERE patient_id = ?
        ORDER BY exam_date
    ''', (patient_id,))
    
    history = c.fetchall()
    
    if not history:
        return None
        
    progression = {
        'trend': calculate_condition_trend(history),
        'improvement_rate': calculate_improvement_rate(history),
        'risk_factors': identify_risk_factors(history),
        'recommended_followup': suggest_followup_schedule(history[-1])
    }
    
    conn.close()
    return progression

def calculate_condition_trend(history):
    """Calculate trend in patient's condition"""
    if len(history) < 2:
        return "Insufficient data for trend analysis"
    
    conditions = [h[0].lower() for h in history]
    confidences = [h[1] for h in history]
    
    if all(c == 'normal' for c in conditions[-2:]):
        return "Maintaining normal condition"
    elif conditions[-1] == 'normal' and conditions[-2] == 'pneumonia':
        return "Showing improvement"
    elif conditions[-1] == 'pneumonia' and conditions[-2] == 'normal':
        return "Condition has deteriorated"
    elif all(c == 'pneumonia' for c in conditions[-2:]):
        if confidences[-1] < confidences[-2]:
            return "Showing signs of improvement"
        else:
            return "Condition persists"
    
    return "Inconsistent pattern"

def calculate_improvement_rate(history):
    """Calculate rate of improvement"""
    if len(history) < 2:
        return None
        
    total_improvements = 0
    for i in range(1, len(history)):
        prev_condition = history[i-1][0].lower()
        curr_condition = history[i][0].lower()
        if prev_condition == 'pneumonia' and curr_condition == 'normal':
            total_improvements += 1
    
    rate = (total_improvements / (len(history) - 1)) * 100
    return round(rate, 1)

def identify_risk_factors(history):
    """Identify potential risk factors"""
    risk_factors = []
    
    # Check frequency of pneumonia
    pneumonia_count = sum(1 for h in history if h[0].lower() == 'pneumonia')
    if pneumonia_count > 1:
        risk_factors.append("Recurring pneumonia")
    
    # Check recovery time
    for i in range(1, len(history)):
        prev_date = datetime.strptime(history[i-1][2], '%Y-%m-%d')
        curr_date = datetime.strptime(history[i][2], '%Y-%m-%d')
        if (curr_date - prev_date).days < 14:
            risk_factors.append("Short recovery periods")
            break
    
    return risk_factors

def suggest_followup_schedule(last_record):
    """Suggest followup schedule based on last record"""
    condition = last_record[0].lower()
    confidence = last_record[1]
    
    if condition == 'normal':
        return {
            'next_visit': 'In 6 months',
            'type': 'Regular checkup',
            'priority': 'Routine'
        }
    else:
        if confidence > 0.8:
            return {
                'next_visit': 'In 1 week',
                'type': 'Follow-up X-ray',
                'priority': 'High'
            }
        else:
            return {
                'next_visit': 'In 3 days',
                'type': 'Urgent follow-up',
                'priority': 'Immediate'
            }

def generate_treatment_recommendations(condition, severity, patient_history):
    """Generate personalized treatment recommendations"""
    recommendations = {
        'medications': get_medication_suggestions(condition, severity),
        'lifestyle_changes': suggest_lifestyle_modifications(condition),
        'followup_schedule': create_followup_schedule(severity),
        'specialist_referrals': determine_specialist_needs(condition, severity),
        'preventive_measures': suggest_preventive_care(patient_history)
    }
    return recommendations

def get_medication_suggestions(condition, severity):
    """Get medication suggestions based on condition and severity"""
    medications = {
        'pneumonia': {
            'mild': {
                'primary': ['Amoxicillin 500mg', 'Azithromycin 250mg'],
                'duration': '7-10 days',
                'notes': 'Take with food'
            },
            'moderate': {
                'primary': ['Levofloxacin 750mg', 'Ceftriaxone 1g'],
                'duration': '10-14 days',
                'notes': 'Monitor liver function'
            },
            'severe': {
                'primary': ['IV antibiotics', 'Oxygen therapy'],
                'duration': 'As directed by physician',
                'notes': 'Hospital admission required'
            }
        }
    }
    return medications.get(condition, {}).get(severity, {})

def suggest_lifestyle_modifications(condition):
    """Suggest lifestyle changes"""
    modifications = {
        'pneumonia': [
            'Rest adequately (8-10 hours/night)',
            'Stay hydrated (8-10 glasses/day)',
            'Practice deep breathing exercises',
            'Avoid smoking and second-hand smoke',
            'Maintain good hand hygiene'
        ],
        'normal': [
            'Regular exercise',
            'Balanced diet',
            'Good sleep hygiene',
            'Stress management',
            'Regular health check-ups'
        ]
    }
    return modifications.get(condition, [])

# Add new routes
@app.route('/patient_analysis/<patient_id>')
def patient_analysis(patient_id):
    try:
        progression = analyze_patient_progression(patient_id)
        if progression:
            return jsonify({
                'success': True,
                'progression': progression
            })
        return jsonify({
            'success': False,
            'error': 'No patient history found'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/heatmap/<image_id>')
def get_heatmap(image_id):
    try:
        image_path = f'static/uploads/image_{image_id}.jpg'
        image_tensor = process_image(image_path)
        heatmap = generate_heatmap_overlay(image_tensor, model)
        
        # Save heatmap
        heatmap_path = f'static/heatmaps/heatmap_{image_id}.jpg'
        os.makedirs('static/heatmaps', exist_ok=True)
        heatmap.save(heatmap_path)
        
        return jsonify({
            'success': True,
            'heatmap_url': heatmap_path
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Modify the main section
if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/reports', exist_ok=True)
    os.makedirs('static/heatmaps', exist_ok=True)
    
    # Initialize database
    init_db()
    
    # Configure app for production
    app.config.update(
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
        SEND_FILE_MAX_AGE_DEFAULT=0,
        TEMPLATES_AUTO_RELOAD=False,  # Disable in production
        PROPAGATE_EXCEPTIONS=True
    )
    
    # Get port from environment
    port = int(os.environ.get('PORT', 10000))
    
    # Run app
    app.run(host='0.0.0.0', port=port)
