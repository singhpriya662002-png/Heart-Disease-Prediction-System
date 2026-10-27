"""
Evaluation and Visual Analytics Module
Computes comparative metrics, confusion matrices, ROC curves, and feature importances.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.inspection import permutation_importance
from src.data_loader import FEATURE_NAMES

def evaluate_models(trained_models, X_test, y_test, cv_results=None):
    """
    Evaluate trained models on test set and compile comprehensive comparison.
    """
    results = {}
    
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            dec = model.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(-dec))
        else:
            y_prob = y_pred.astype(float)
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        # Sample points evenly for json visualization
        indices = np.linspace(0, len(fpr) - 1, min(30, len(fpr))).astype(int)
        roc_data = [
            {'fpr': round(float(fpr[i]), 4), 'tpr': round(float(tpr[i]), 4)}
            for i in indices
        ]
        # Ensure 0,0 and 1,1
        if roc_data[0] != {'fpr': 0.0, 'tpr': 0.0}:
            roc_data.insert(0, {'fpr': 0.0, 'tpr': 0.0})
        if roc_data[-1] != {'fpr': 1.0, 'tpr': 1.0}:
            roc_data.append({'fpr': 1.0, 'tpr': 1.0})
            
        results[name] = {
            'test_metrics': {
                'accuracy': round(float(acc), 4),
                'precision': round(float(prec), 4),
                'recall': round(float(rec), 4),
                'f1_score': round(float(f1), 4),
                'roc_auc': round(float(auc), 4),
                'specificity': round(float(specificity), 4)
            },
            'confusion_matrix': {
                'matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp),
                'total': int(len(y_test))
            },
            'roc_curve': roc_data,
            'cv_metrics': cv_results.get(name, {}) if cv_results else {}
        }
        
    return results

def compute_feature_importances(trained_models, X_test, y_test):
    """
    Extract or compute feature importances for each model.
    """
    importances = {}
    
    # 1. Random Forest Feature Importance (MDI)
    if 'Random Forest' in trained_models:
        rf = trained_models['Random Forest']
        rf_imp = rf.feature_importances_
        importances['Random Forest'] = [
            {'feature': f, 'importance': round(float(imp), 4)}
            for f, imp in sorted(zip(FEATURE_NAMES, rf_imp), key=lambda x: x[1], reverse=True)
        ]
        
    # 2. Logistic Regression Coefficients (Absolute Magnitude & Direction)
    if 'Logistic Regression' in trained_models:
        lr = trained_models['Logistic Regression']
        lr_coef = lr.coef_[0]
        importances['Logistic Regression'] = [
            {
                'feature': f,
                'importance': round(float(abs(c)), 4),
                'coefficient': round(float(c), 4),
                'direction': 'increases_risk' if c > 0 else 'decreases_risk'
            }
            for f, c in sorted(zip(FEATURE_NAMES, lr_coef), key=lambda x: abs(x[1]), reverse=True)
        ]
        
    # 3. SVM Permutation Importance
    if 'SVM' in trained_models:
        svm = trained_models['SVM']
        perm = permutation_importance(svm, X_test, y_test, n_repeats=10, random_state=42)
        svm_imp = perm.importances_mean
        # Normalize to positive values for ranking
        norm_imp = [max(0.0, float(v)) for v in svm_imp]
        total = sum(norm_imp) if sum(norm_imp) > 0 else 1.0
        importances['SVM'] = [
            {'feature': f, 'importance': round(float(v / total), 4)}
            for f, v in sorted(zip(FEATURE_NAMES, norm_imp), key=lambda x: x[1], reverse=True)
        ]
        
    return importances

def generate_analytics_data(df):
    """
    Precompute analytics summaries for fast interactive dashboard rendering.
    """
    # 1. Correlation Matrix
    corr = df.corr().round(3).to_dict()
    
    # 2. Risk breakdown by Chest Pain (cp)
    cp_labels = {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-anginal Pain', 3: 'Asymptomatic'}
    cp_stats = []
    for cp_val, label in cp_labels.items():
        subset = df[df['cp'] == cp_val]
        total = len(subset)
        disease_count = int(subset['target'].sum()) if total > 0 else 0
        rate = round(disease_count / total, 3) if total > 0 else 0.0
        cp_stats.append({
            'code': cp_val,
            'label': label,
            'total': total,
            'disease_count': disease_count,
            'rate': rate
        })
        
    # 3. Risk breakdown by Sex
    sex_stats = []
    for sex_val, label in [(0, 'Female'), (1, 'Male')]:
        subset = df[df['sex'] == sex_val]
        total = len(subset)
        disease_count = int(subset['target'].sum())
        rate = round(disease_count / total, 3) if total > 0 else 0.0
        sex_stats.append({
            'sex': label,
            'total': total,
            'disease_count': disease_count,
            'rate': rate
        })
        
    # 4. Age groups distribution
    bins = [25, 40, 50, 60, 70, 85]
    labels = ['25-40', '41-50', '51-60', '61-70', '71+']
    df_temp = df.copy()
    df_temp['age_group'] = pd.cut(df_temp['age'], bins=bins, labels=labels, right=True)
    age_stats = []
    for group in labels:
        subset = df_temp[df_temp['age_group'] == group]
        total = len(subset)
        disease = int(subset['target'].sum()) if total > 0 else 0
        healthy = total - disease
        age_stats.append({
            'group': group,
            'total': total,
            'disease': disease,
            'healthy': healthy,
            'rate': round(disease / total, 3) if total > 0 else 0.0
        })
        
    # 5. Scatter points for Age vs Thalach (max heart rate)
    scatter_age_thalach = [
        {'age': int(r['age']), 'thalach': int(r['thalach']), 'target': int(r['target'])}
        for _, r in df.iterrows()
    ]
    
    # 6. Scatter points for Chol vs Trestbps
    scatter_chol_trestbps = [
        {'chol': int(r['chol']), 'trestbps': int(r['trestbps']), 'target': int(r['target'])}
        for _, r in df.iterrows()
    ]
    
    # 7. Summary averages for healthy vs disease
    healthy_means = df[df['target'] == 0].mean().round(2).to_dict()
    disease_means = df[df['target'] == 1].mean().round(2).to_dict()
    
    return {
        'correlation': corr,
        'cp_stats': cp_stats,
        'sex_stats': sex_stats,
        'age_stats': age_stats,
        'scatter_age_thalach': scatter_age_thalach,
        'scatter_chol_trestbps': scatter_chol_trestbps,
        'healthy_means': healthy_means,
        'disease_means': disease_means,
        'feature_names': FEATURE_NAMES,
        'total_samples': len(df),
        'disease_total': int(df['target'].sum()),
        'healthy_total': int((df['target'] == 0).sum())
    }

def save_metrics_and_analytics(results, importances, analytics, output_path='models/metrics.json'):
    """Save full results payload to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        'model_comparison': results,
        'feature_importances': importances,
        'visual_analytics': analytics
    }
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"Metrics and visual analytics saved to {output_path}")
    return payload
