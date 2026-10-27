"""
Complete Pipeline Training and Evaluation Script
Executes data loading, cross-validation, test evaluation, model serialization,
and publication plot generation for Logistic Regression, SVM, and Random Forest.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import prepare_train_test_data, FEATURE_NAMES, load_data
from src.model_trainer import get_models, train_and_cross_validate, save_models
from src.evaluation import evaluate_models, compute_feature_importances, generate_analytics_data, save_metrics_and_analytics

def generate_visual_artifacts(results, importances, df, output_dir='static/images'):
    """Generate high-resolution PNG charts for visual analytics and reporting."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style='darkgrid')
    plt.rcParams.update({'font.sans-serif': 'Helvetica', 'axes.edgecolor': '#334155'})

    # 1. Model Comparison Bar Chart
    plt.figure(figsize=(10, 6), facecolor='#0f172a')
    ax = plt.subplot(111, facecolor='#1e293b')
    metrics_list = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    models = list(results.keys())
    x = np.arange(len(metric_labels))
    width = 0.25
    colors = ['#38bdf8', '#818cf8', '#34d399']

    for i, (model_name, col) in enumerate(zip(models, colors)):
        vals = [results[model_name]['test_metrics'][m] * 100 for m in metrics_list]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=model_name, color=col, alpha=0.9, edgecolor='#0f172a')
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, color='#e2e8f0', fontweight='bold')

    ax.set_ylabel('Score (%)', color='#e2e8f0', fontsize=12)
    ax.set_title('Heart Disease Classifier Benchmark: Test Set Performance', color='#f8fafc', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, color='#e2e8f0', fontsize=11)
    ax.tick_params(colors='#94a3b8')
    ax.set_ylim(50, 105)
    ax.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=200, facecolor='#0f172a')
    plt.close()

    # 2. Combined ROC Curves
    plt.figure(figsize=(8, 7), facecolor='#0f172a')
    ax = plt.subplot(111, facecolor='#1e293b')
    for model_name, col in zip(models, colors):
        curve_data = results[model_name]['roc_curve']
        fpr = [p['fpr'] for p in curve_data]
        tpr = [p['tpr'] for p in curve_data]
        auc_score = results[model_name]['test_metrics']['roc_auc']
        ax.plot(fpr, tpr, color=col, lw=2.5, label=f"{model_name} (AUC = {auc_score:.3f})")

    ax.plot([0, 1], [0, 1], color='#64748b', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.50)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', color='#e2e8f0', fontsize=11)
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', color='#e2e8f0', fontsize=11)
    ax.set_title('Receiver Operating Characteristic (ROC) Curves', color='#f8fafc', fontsize=13, fontweight='bold', pad=15)
    ax.tick_params(colors='#94a3b8')
    ax.legend(loc="lower right", facecolor='#0f172a', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=200, facecolor='#0f172a')
    plt.close()

    # 3. Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor='#0f172a')
    for ax, model_name in zip(axes, models):
        ax.set_facecolor('#1e293b')
        cm = np.array(results[model_name]['confusion_matrix']['matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    annot_kws={'size': 14, 'weight': 'bold'},
                    xticklabels=['No Disease (0)', 'Disease (1)'],
                    yticklabels=['No Disease (0)', 'Disease (1)'])
        ax.set_title(f"{model_name}", color='#f8fafc', fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Predicted Label', color='#cbd5e1', fontsize=10)
        ax.set_ylabel('True Label', color='#cbd5e1', fontsize=10)
        ax.tick_params(colors='#94a3b8')
    fig.suptitle('Confusion Matrices Comparison (Held-out Test Cohort)', color='#f8fafc', fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=200, facecolor='#0f172a')
    plt.close()

    # 4. Feature Importance (Random Forest vs Logistic Regression)
    plt.figure(figsize=(12, 6), facecolor='#0f172a')
    ax = plt.subplot(111, facecolor='#1e293b')
    rf_items = importances.get('Random Forest', [])
    feat_names = [item['feature'] for item in rf_items]
    feat_vals = [item['importance'] * 100 for item in rf_items]

    y_pos = np.arange(len(feat_names))
    ax.barh(y_pos, feat_vals, color='#38bdf8', alpha=0.9, edgecolor='#0f172a')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_names, color='#e2e8f0', fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Relative Importance (%)', color='#e2e8f0', fontsize=11)
    ax.set_title('Random Forest Feature Importance (MDI)', color='#f8fafc', fontsize=13, fontweight='bold', pad=15)
    ax.tick_params(colors='#94a3b8')
    for i, v in enumerate(feat_vals):
        ax.text(v + 0.3, i, f"{v:.1f}%", va='center', color='#cbd5e1', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=200, facecolor='#0f172a')
    plt.close()

    # 5. Correlation Heatmap
    plt.figure(figsize=(11, 9), facecolor='#0f172a')
    ax = plt.subplot(111, facecolor='#1e293b')
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.8, vmin=-0.8, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8}, annot=True, fmt='.2f',
                annot_kws={'size': 8}, ax=ax)
    ax.set_title('UCI Heart Disease Feature Correlation Matrix', color='#f8fafc', fontsize=14, fontweight='bold', pad=15)
    ax.tick_params(colors='#94a3b8')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=200, facecolor='#0f172a')
    plt.close()

    print(f"All visual analytics charts successfully saved to {output_dir}/")

def main():
    print("===================================================================")
    print("        HEART DISEASE PREDICTION SYSTEM - ML PIPELINE             ")
    print("===================================================================")
    
    # 1. Prepare data
    print("\n[Step 1/5] Loading & Preprocessing UCI Heart Disease Dataset...")
    data = prepare_train_test_data('data/heart.csv', test_size=0.2, random_state=42)
    print(f"Total instances: {len(data['df'])}")
    print(f"Training instances: {len(data['y_train'])} | Test instances: {len(data['y_test'])}")
    print(f"Features: {len(FEATURE_NAMES)}")
    
    # 2. Initialize and Train Models
    print("\n[Step 2/5] Training Algorithms with 5-Fold Stratified Cross-Validation...")
    models = get_models()
    trained_models, cv_results = train_and_cross_validate(
        models, data['X_train_scaled'], data['y_train'], cv_folds=5
    )
    
    # 3. Test Evaluation
    print("\n[Step 3/5] Evaluating on Independent Test Set (20%)...")
    test_results = evaluate_models(trained_models, data['X_test_scaled'], data['y_test'], cv_results)
    
    # Print formatted comparison table
    print("\n" + "=" * 80)
    print(f"{'Algorithm':<22} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
    print("-" * 80)
    for model_name, info in test_results.items():
        m = info['test_metrics']
        print(f"{model_name:<22} | {m['accuracy']*100:6.2f}%   | {m['precision']*100:6.2f}%   | {m['recall']*100:6.2f}%   | {m['f1_score']*100:6.2f}%   | {m['roc_auc']:6.4f}")
    print("=" * 80)
    
    # Cross Validation Summary
    print("\n5-Fold Stratified Cross-Validation (Mean ± Std):")
    for model_name, info in test_results.items():
        cv = info['cv_metrics']
        print(f" - {model_name:<20}: Acc = {cv['accuracy_mean']*100:.2f}% (±{cv['accuracy_std']*100:.2f}%) | ROC-AUC = {cv['roc_auc_mean']:.3f} (±{cv['roc_auc_std']:.3f})")
    
    # 4. Feature Importance & Analytics
    print("\n[Step 4/5] Computing Feature Importances & Visual Analytics...")
    importances = compute_feature_importances(trained_models, data['X_test_scaled'], data['y_test'])
    analytics = generate_analytics_data(data['df'])
    
    # 5. Serialization
    print("\n[Step 5/5] Saving Models, Metrics, and Generating Visual Charts...")
    save_models(trained_models, model_dir='models')
    save_metrics_and_analytics(test_results, importances, analytics, output_path='models/metrics.json')
    generate_visual_artifacts(test_results, importances, data['df'], output_dir='static/images')
    
    print("\n[SUCCESS] Pipeline training complete! All artifacts and models ready.")
    print("===================================================================\n")

if __name__ == '__main__':
    main()
