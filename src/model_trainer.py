"""
Model Trainer for Heart Disease Prediction
Trains Logistic Regression, Support Vector Machine (SVM), and Random Forest.
"""

import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from sklearn.calibration import CalibratedClassifierCV

def get_models():
    """Return dictionary of the target algorithms."""
    models = {
        'Logistic Regression': LogisticRegression(
            C=0.8,
            max_iter=1000,
            solver='lbfgs',
            random_state=42
        ),
        'SVM': CalibratedClassifierCV(
            estimator=SVC(
                C=1.2,
                kernel='rbf',
                gamma='scale',
                random_state=42
            ),
            ensemble=False
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=150,
            max_depth=5,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42
        )
    }
    return models

def train_and_cross_validate(models, X_train, y_train, cv_folds=5):
    """
    Train each model and calculate cross-validation metrics.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    cv_results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"--> Training and cross-validating {name}...")
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False)
        
        cv_results[name] = {
            'accuracy_mean': float(scores['test_accuracy'].mean()),
            'accuracy_std': float(scores['test_accuracy'].std()),
            'precision_mean': float(scores['test_precision'].mean()),
            'precision_std': float(scores['test_precision'].std()),
            'recall_mean': float(scores['test_recall'].mean()),
            'recall_std': float(scores['test_recall'].std()),
            'f1_mean': float(scores['test_f1'].mean()),
            'f1_std': float(scores['test_f1'].std()),
            'roc_auc_mean': float(scores['test_roc_auc'].mean()),
            'roc_auc_std': float(scores['test_roc_auc'].std())
        }
        
        # Fit on full training data
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    return trained_models, cv_results

def save_models(trained_models, model_dir='models'):
    """Save trained models to disk."""
    os.makedirs(model_dir, exist_ok=True)
    filename_map = {
        'Logistic Regression': 'logistic_regression.joblib',
        'SVM': 'svm.joblib',
        'Random Forest': 'random_forest.joblib'
    }
    paths = {}
    for name, model in trained_models.items():
        filename = filename_map.get(name, f"{name.lower().replace(' ', '_')}.joblib")
        path = os.path.join(model_dir, filename)
        joblib.dump(model, path)
        paths[name] = path
        print(f"Saved {name} to {path}")
    return paths

def load_all_models(model_dir='models'):
    """Load all saved models and scaler."""
    filename_map = {
        'Logistic Regression': 'logistic_regression.joblib',
        'SVM': 'svm.joblib',
        'Random Forest': 'random_forest.joblib'
    }
    models = {}
    for name, filename in filename_map.items():
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return models, scaler
