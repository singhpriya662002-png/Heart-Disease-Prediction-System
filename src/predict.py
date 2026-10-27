"""
Inference and Clinical Assessment Engine
Generates predictions from Logistic Regression, SVM, and Random Forest,
along with risk stratification and patient-specific risk factor breakdown.
"""

import numpy as np
import pandas as pd
from src.data_loader import FEATURE_NAMES
from src.model_trainer import load_all_models

CLINICAL_THRESHOLDS = {
    'trestbps': {'healthy_max': 120, 'elevated': 130, 'unit': 'mm Hg', 'label': 'Resting Blood Pressure'},
    'chol': {'healthy_max': 200, 'elevated': 240, 'unit': 'mg/dl', 'label': 'Serum Cholesterol'},
    'fbs': {'healthy_max': 0, 'elevated': 1, 'unit': '>120 mg/dl', 'label': 'Fasting Blood Sugar'},
    'thalach': {'healthy_min': 150, 'unit': 'bpm', 'label': 'Max Heart Rate Achieved'},
    'oldpeak': {'healthy_max': 1.0, 'elevated': 2.0, 'unit': 'mm', 'label': 'ST Depression (Oldpeak)'},
    'ca': {'healthy_max': 0, 'unit': 'vessels', 'label': 'Major Vessels Blocked (Fluoroscopy)'},
    'exang': {'healthy_max': 0, 'unit': 'yes/no', 'label': 'Exercise Induced Angina'},
    'cp': {'high_risk': 0, 'unit': 'type', 'label': 'Chest Pain Type'}
}

class HeartDiseasePredictor:
    def __init__(self, model_dir='models'):
        self.models, self.scaler = load_all_models(model_dir)
        if not self.models:
            raise RuntimeError(f"No trained models found in {model_dir}. Please run train.py first.")
        if self.scaler is None:
            raise RuntimeError(f"Scaler not found in {model_dir}. Please run train.py first.")
            
    def predict_patient(self, patient_data):
        """
        Predict heart disease probability and risk profile for a patient.
        patient_data: dict with all 13 features
        """
        # Ensure all features present
        missing = [f for f in FEATURE_NAMES if f not in patient_data]
        if missing:
            raise ValueError(f"Missing required patient features: {missing}")
            
        # Build DataFrame with exact feature order
        df_input = pd.DataFrame([{f: float(patient_data[f]) for f in FEATURE_NAMES}])
        
        # Scale
        X_scaled = self.scaler.transform(df_input)
        
        # Collect predictions
        model_results = {}
        probabilities = []
        
        for name, model in self.models.items():
            pred = int(model.predict(X_scaled)[0])
            if hasattr(model, 'predict_proba'):
                prob = float(model.predict_proba(X_scaled)[0, 1])
            elif hasattr(model, 'decision_function'):
                dec = float(model.decision_function(X_scaled)[0])
                prob = float(1 / (1 + np.exp(-dec)))
            else:
                prob = float(pred)
                
            model_results[name] = {
                'prediction': pred,
                'prediction_label': 'Heart Disease Detected' if pred == 1 else 'No Heart Disease',
                'probability': round(prob, 4),
                'confidence_percent': round(prob * 100, 1) if pred == 1 else round((1 - prob) * 100, 1)
            }
            probabilities.append(prob)
            
        # Ensemble Consensus (Weighted Average)
        # Weights: Random Forest: 0.40, Logistic Regression: 0.35, SVM: 0.25
        weights = {'Random Forest': 0.40, 'Logistic Regression': 0.35, 'SVM': 0.25}
        total_w = sum(weights.get(name, 0.33) for name in self.models.keys())
        ensemble_prob = sum(model_results[name]['probability'] * weights.get(name, 0.33) for name in self.models.keys()) / total_w
        ensemble_pred = 1 if ensemble_prob >= 0.50 else 0
        
        # Risk Stratification
        if ensemble_prob < 0.35:
            risk_tier = 'Low Risk'
            risk_color = '#10b981' # Emerald Green
            clinical_advice = "Patient displays low cardiac risk based on presented biomarkers. Continue regular wellness check-ups, balanced diet, and active lifestyle."
        elif ensemble_prob <= 0.65:
            risk_tier = 'Moderate Risk'
            risk_color = '#f59e0b' # Amber
            clinical_advice = "Borderline cardiac risk factors detected. Clinical correlation, lipid panel monitoring, and routine stress ECG evaluation are recommended."
        else:
            risk_tier = 'High Risk'
            risk_color = '#ef4444' # Coral Red
            clinical_advice = "Significant cardiac risk indicators present. Prompt formal cardiology evaluation (echocardiogram, diagnostic angiography, and physician consultation) is advised."
            
        # Clinical Risk Factors Breakdown
        risk_factors = []
        protective_factors = []
        
        # Check Blood Pressure
        bp = patient_data.get('trestbps', 120)
        if bp >= 140:
            risk_factors.append({
                'factor': 'Hypertension (Blood Pressure)',
                'value': f"{bp} mm Hg",
                'severity': 'High',
                'description': 'Elevated resting blood pressure increases cardiac workload and vascular strain.'
            })
        elif bp >= 130:
            risk_factors.append({
                'factor': 'Pre-hypertensive Blood Pressure',
                'value': f"{bp} mm Hg",
                'severity': 'Moderate',
                'description': 'Blood pressure is borderline high.'
            })
        else:
            protective_factors.append({
                'factor': 'Optimal Blood Pressure',
                'value': f"{bp} mm Hg",
                'description': 'Resting blood pressure within healthy parameters.'
            })
            
        # Check Cholesterol
        chol = patient_data.get('chol', 200)
        if chol >= 240:
            risk_factors.append({
                'factor': 'Hypercholesterolemia',
                'value': f"{chol} mg/dl",
                'severity': 'High',
                'description': 'High serum cholesterol accelerates coronary atherosclerotic plaque buildup.'
            })
        elif chol >= 200:
            risk_factors.append({
                'factor': 'Borderline Cholesterol',
                'value': f"{chol} mg/dl",
                'severity': 'Moderate',
                'description': 'Cholesterol is slightly elevated above optimal guidelines.'
            })
        else:
            protective_factors.append({
                'factor': 'Normal Cholesterol',
                'value': f"{chol} mg/dl",
                'description': 'Lipid levels are within normal range.'
            })
            
        # Check Heart Rate
        hr = patient_data.get('thalach', 150)
        if hr < 120:
            risk_factors.append({
                'factor': 'Chronotropic Incompetence (Low Max Heart Rate)',
                'value': f"{hr} bpm",
                'severity': 'High',
                'description': 'Inability to reach adequate heart rate during exercise is a strong cardiac risk marker.'
            })
        else:
            protective_factors.append({
                'factor': 'Good Maximum Heart Rate Capacity',
                'value': f"{hr} bpm",
                'description': 'Adequate cardiac reserve during exertion.'
            })
            
        # ST Depression
        oldpeak = patient_data.get('oldpeak', 0.0)
        if oldpeak >= 2.0:
            risk_factors.append({
                'factor': 'Marked ST Depression',
                'value': f"{oldpeak} mm",
                'severity': 'High',
                'description': 'Significant exercise-induced ST depression strongly correlates with coronary ischemia.'
            })
        elif oldpeak > 1.0:
            risk_factors.append({
                'factor': 'Moderate ST Depression',
                'value': f"{oldpeak} mm",
                'severity': 'Moderate',
                'description': 'Mild to moderate ST segment shift detected during stress.'
            })
            
        # Exercise Angina
        if patient_data.get('exang', 0) == 1:
            risk_factors.append({
                'factor': 'Exercise-Induced Angina',
                'value': 'Present',
                'severity': 'High',
                'description': 'Chest discomfort during exertion indicates compromised myocardial blood flow.'
            })
            
        # Major Vessels Colored
        ca = patient_data.get('ca', 0)
        if ca > 0:
            risk_factors.append({
                'factor': f'Coronary Vessels Narrowed: {int(ca)}',
                'value': f"{int(ca)} vessels",
                'severity': 'High',
                'description': 'Fluoroscopy indicates coronary vessel calcification/stenosis.'
            })
            
        # Thalassemia
        thal = patient_data.get('thal', 2)
        if thal == 3: # reversible defect
            risk_factors.append({
                'factor': 'Reversible Thalassemia Defect',
                'value': 'Type 3 (Reversible)',
                'severity': 'High',
                'description': 'Perfusion scintigraphy indicates reversible myocardial ischemia.'
            })
            
        return {
            'models': model_results,
            'ensemble': {
                'prediction': ensemble_pred,
                'prediction_label': 'Heart Disease Detected' if ensemble_pred == 1 else 'No Heart Disease',
                'probability': round(ensemble_prob, 4),
                'risk_percentage': round(ensemble_prob * 100, 1),
                'risk_tier': risk_tier,
                'risk_color': risk_color,
                'clinical_advice': clinical_advice
            },
            'risk_factors': risk_factors,
            'protective_factors': protective_factors,
            'patient_input': patient_data,
            'disclaimer': 'CRITICAL MEDICAL DISCLAIMER: Predictions generated by this system are for educational, research, and technical demonstration purposes only. They DO NOT constitute a medical diagnosis, clinical recommendation, or treatment plan. Always consult a certified healthcare professional for medical evaluations.'
        }
