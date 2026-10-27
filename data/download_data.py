"""
Dataset Downloader and Cleaner for UCI Cleveland Heart Disease Dataset.
Downloads directly from UCI Machine Learning Repository,
handles missing values ('?'), encodes binary target (0 = Normal, 1 = Heart Disease),
and formats features for standard clinical machine learning.
"""

import os
import subprocess
import pandas as pd
import numpy as np

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
]

def fetch_and_prepare_uci_data(output_path='data/heart.csv'):
    """Fetch raw UCI data, clean, encode, and save to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Fetching official UCI Cleveland dataset from {UCI_URL}...")
    
    res = subprocess.run(['curl', '-sL', UCI_URL], capture_output=True, text=True)
    if res.returncode != 0 or not res.stdout.strip():
        raise RuntimeError("Failed to download UCI dataset via curl.")
        
    lines = [line.strip().split(',') for line in res.stdout.strip().split('\n') if line.strip()]
    df = pd.DataFrame(lines, columns=COLUMNS)
    
    # Replace '?' with NaN
    df = df.replace('?', np.nan)
    
    # Cast all columns to float/numeric
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Impute missing values with median / mode
    df['ca'] = df['ca'].fillna(df['ca'].median()).astype(int)
    df['thal'] = df['thal'].fillna(df['thal'].mode()[0]).astype(int)
    
    # Harmonize thal coding: in UCI: 3=normal, 6=fixed defect, 7=reversible defect
    # Map to: 1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect
    thal_mapping = {3: 1, 6: 2, 7: 3}
    df['thal'] = df['thal'].map(lambda x: thal_mapping.get(x, 1)).astype(int)
    
    # Harmonize cp: 1: Typical angina, 2: Atypical angina, 3: Non-anginal, 4: Asymptomatic
    # Map to 0-3 for zero-indexed standards: 0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic
    df['cp'] = (df['cp'] - 1).astype(int)
    
    # Harmonize slope: 1: Upsloping, 2: Flat, 3: Downsloping
    # Map to 0-2: 0: Upsloping, 1: Flat, 2: Downsloping
    df['slope'] = (df['slope'] - 1).astype(int)
    
    # Target: 0 = Absence of heart disease (<50% diameter narrowing)
    #         1 = Presence of heart disease (>50% diameter narrowing)
    df['target'] = (df['num'] > 0).astype(int)
    df = df.drop(columns=['num'])
    
    # Integer conversions where appropriate
    for int_col in ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'slope', 'ca', 'thal', 'target']:
        df[int_col] = df[int_col].astype(int)
        
    df.to_csv(output_path, index=False)
    print(f"Dataset saved successfully to {output_path} ({len(df)} rows, {len(df.columns)} columns)")
    print(f"Target distribution: Normal (0): {(df['target'] == 0).sum()} | Heart Disease (1): {(df['target'] == 1).sum()}")
    return df

if __name__ == '__main__':
    fetch_and_prepare_uci_data()
