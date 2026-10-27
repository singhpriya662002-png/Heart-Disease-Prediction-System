"""
Data Loader and Preprocessing Module for UCI Heart Disease Dataset
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

FEATURE_DESCRIPTIONS = {
    'age': 'Age in years',
    'sex': 'Sex (1 = male; 0 = female)',
    'cp': 'Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal, 3: asymptomatic)',
    'trestbps': 'Resting blood pressure (mm Hg)',
    'chol': 'Serum cholesterol (mg/dl)',
    'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
    'restecg': 'Resting ECG results (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy)',
    'thalach': 'Maximum heart rate achieved (bpm)',
    'exang': 'Exercise induced angina (1 = yes; 0 = no)',
    'oldpeak': 'ST depression induced by exercise relative to rest',
    'slope': 'Slope of peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping)',
    'ca': 'Number of major vessels (0-3) colored by flourosopy',
    'thal': 'Thalassemia (1: normal, 2: fixed defect, 3: reversible defect)'
}

def load_data(filepath='data/heart.csv'):
    """Load dataset from csv file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    # Strip any potential whitespace or BOM characters in column names
    df.columns = [col.strip().lstrip('\ufeff') for col in df.columns]
    return df

def get_feature_and_target(df):
    """Separate features and target variable."""
    X = df[FEATURE_NAMES].copy()
    y = df['target'].copy()
    return X, y

def prepare_train_test_data(filepath='data/heart.csv', test_size=0.2, random_state=42, save_scaler_path='models/scaler.joblib'):
    """
    Load data, split into train and test sets, and scale numerical features.
    Saves scaler if path provided.
    """
    df = load_data(filepath)
    X, y = get_feature_and_target(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if save_scaler_path:
        os.makedirs(os.path.dirname(save_scaler_path), exist_ok=True)
        joblib.dump(scaler, save_scaler_path)
        
    return {
        'df': df,
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler
    }
