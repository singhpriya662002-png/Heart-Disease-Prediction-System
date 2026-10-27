"""
Flask Web Application for Heart Disease Prediction System
Provides REST API endpoints and an interactive web dashboard.
"""

import os
import json
import random
import pandas as pd
from flask import Flask, render_template, request, jsonify
from src.predict import HeartDiseasePredictor
from src.data_loader import load_data, FEATURE_NAMES

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize predictor and load metrics
predictor = None
metrics_data = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = HeartDiseasePredictor(model_dir=os.path.join(BASE_DIR, 'models'))
    return predictor

def get_metrics_data():
    global metrics_data
    if metrics_data is None:
        metrics_path = os.path.join(BASE_DIR, 'models', 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
        else:
            metrics_data = {}
    return metrics_data

# Realistic clinical sample presets for one-click testing
SAMPLE_PATIENTS = {
    'low_risk': {
        'name': 'Healthy Athlete (Low Risk)',
        'description': '38-year-old female with optimal blood pressure, normal cholesterol, high cardiac reserve, and normal ECG.',
        'data': {
            'age': 38, 'sex': 0, 'cp': 1, 'trestbps': 114, 'chol': 172,
            'fbs': 0, 'restecg': 0, 'thalach': 182, 'exang': 0, 'oldpeak': 0.2,
            'slope': 0, 'ca': 0, 'thal': 1
        }
    },
    'moderate_risk': {
        'name': 'Borderline Patient (Moderate Risk)',
        'description': '54-year-old male with pre-hypertension, borderline cholesterol, mild ST depression, and non-anginal discomfort.',
        'data': {
            'age': 54, 'sex': 1, 'cp': 2, 'trestbps': 136, 'chol': 238,
            'fbs': 0, 'restecg': 1, 'thalach': 142, 'exang': 0, 'oldpeak': 1.2,
            'slope': 1, 'ca': 1, 'thal': 2
        }
    },
    'high_risk': {
        'name': 'Cardiac Patient (High Risk)',
        'description': '67-year-old male with stage 2 hypertension, hypercholesterolemia, exercise-induced angina, marked ST depression, and multi-vessel involvement.',
        'data': {
            'age': 67, 'sex': 1, 'cp': 3, 'trestbps': 160, 'chol': 286,
            'fbs': 1, 'restecg': 2, 'thalach': 108, 'exang': 1, 'oldpeak': 2.6,
            'slope': 1, 'ca': 2, 'thal': 3
        }
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    """Return model performance metrics and visual analytics data."""
    data = get_metrics_data()
    return jsonify(data)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Predict heart disease probability from patient attributes."""
    try:
        req_data = request.get_json(force=True)
        if not req_data:
            return jsonify({'error': 'No input data provided'}), 400
            
        pred_engine = get_predictor()
        result = pred_engine.predict_patient(req_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/sample-patients', methods=['GET'])
def api_samples():
    """Return clinical preset profiles and random dataset record."""
    presets = dict(SAMPLE_PATIENTS)
    
    # Add random record from dataset
    try:
        df = load_data(os.path.join(BASE_DIR, 'data', 'heart.csv'))
        random_row = df.sample(n=1, random_state=random.randint(1, 10000)).iloc[0]
        actual_target = int(random_row['target'])
        random_data = {f: float(random_row[f]) if f == 'oldpeak' else int(random_row[f]) for f in FEATURE_NAMES}
        presets['random_uci'] = {
            'name': f"Random UCI Patient (Actual: {'Heart Disease' if actual_target == 1 else 'Normal'})",
            'description': f"Randomly selected record from the UCI Cleveland cohort (Patient #{int(random_row.name)}). Ground truth label is {actual_target}.",
            'actual_target': actual_target,
            'data': random_data
        }
    except Exception:
        pass
        
    return jsonify(presets)

@app.route('/api/stats', methods=['GET'])
def api_stats():
    """Return top-level project statistics."""
    try:
        df = load_data(os.path.join(BASE_DIR, 'data', 'heart.csv'))
        metrics = get_metrics_data()
        rf_acc = metrics.get('model_comparison', {}).get('Random Forest', {}).get('test_metrics', {}).get('accuracy', 0.9016)
        
        return jsonify({
            'total_samples': len(df),
            'features_count': len(FEATURE_NAMES),
            'disease_cases': int(df['target'].sum()),
            'healthy_cases': int((df['target'] == 0).sum()),
            'best_accuracy': round(float(rf_acc) * 100, 1),
            'algorithms': ['Logistic Regression', 'Support Vector Machine', 'Random Forest']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    print(f"Starting Heart Disease Prediction System on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
