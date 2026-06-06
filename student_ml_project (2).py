# ============================================================
#  Predictive Modeling Using Machine Learning
#  Dataset: Student Performance
#  Internship Task 2 - Thiranex
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)

# ── Save folder: same folder as this script ──────────────────
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"📁 Plots will be saved to: {OUTPUT_DIR}\n")

print("✅ Step 1: Libraries imported successfully\n")

# ── STEP 2: Load Dataset ──────────────────────────────────────
np.random.seed(42)
n = 500

data = pd.DataFrame({
    'gender':             np.random.choice(['male', 'female'], n),
    'race_ethnicity':     np.random.choice(['group A','group B','group C','group D','group E'], n),
    'parental_education': np.random.choice(['high school','some college','associate degree','bachelor degree','master degree'], n),
    'lunch':              np.random.choice(['standard', 'free/reduced'], n),
    'test_preparation':   np.random.choice(['none', 'completed'], n),
    'math_score':         np.random.randint(20, 100, n),
    'reading_score':      np.random.randint(20, 100, n),
    'writing_score':      np.random.randint(20, 100, n),
})

data['average_score'] = (data['math_score'] + data['reading_score'] + data['writing_score']) / 3
data['pass_fail']     = (data['average_score'] >= 60).astype(int)

print("✅ Step 2: Dataset loaded")
print(f"   Shape: {data.shape}")
print(data.head(), "\n")

# ── STEP 3: EDA ───────────────────────────────────────────────
print("✅ Step 3: EDA")
print(data.describe())
print("\nMissing values:\n", data.isnull().sum())
print("\nPass/Fail distribution:\n", data['pass_fail'].value_counts())

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ['math_score', 'reading_score', 'writing_score']):
    ax.hist(data[col], bins=20, color='steelblue', edgecolor='white')
    ax.set_title(f'{col} Distribution')
    ax.set_xlabel('Score')
    ax.set_ylabel('Count')
plt.suptitle('Score Distributions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   → Saved: eda_distributions.png\n")

# ── STEP 4: Preprocessing ─────────────────────────────────────
print("✅ Step 4: Preprocessing")
le = LabelEncoder()
cat_cols = ['gender', 'race_ethnicity', 'parental_education', 'lunch', 'test_preparation']
df = data.copy()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])
print("   Categorical columns encoded.\n")

# ── STEP 5A: Linear Regression ────────────────────────────────
print("✅ Step 5A: Linear Regression")
features_lr = ['gender','race_ethnicity','parental_education','lunch','test_preparation',
               'math_score','reading_score','writing_score']
X_lr = df[features_lr]
y_lr = df['average_score']

X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42)
lr = LinearRegression()
lr.fit(X_train_lr, y_train_lr)
y_pred_lr = lr.predict(X_test_lr)

mse = mean_squared_error(y_test_lr, y_pred_lr)
r2  = r2_score(y_test_lr, y_pred_lr)
print(f"   MSE : {mse:.2f}")
print(f"   R²  : {r2:.4f}")

plt.figure(figsize=(7, 5))
plt.scatter(y_test_lr, y_pred_lr, alpha=0.6, color='steelblue', edgecolors='white')
plt.plot([y_test_lr.min(), y_test_lr.max()], [y_test_lr.min(), y_test_lr.max()], 'r--', lw=2)
plt.xlabel('Actual Average Score')
plt.ylabel('Predicted Average Score')
plt.title(f'Linear Regression: Actual vs Predicted\nR² = {r2:.4f}')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'linear_regression_plot.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   → Saved: linear_regression_plot.png\n")

# ── STEP 5B: Decision Tree ────────────────────────────────────
print("✅ Step 5B: Decision Tree Classifier")
features_clf = ['gender','race_ethnicity','parental_education','lunch','test_preparation',
                'math_score','reading_score','writing_score']
X_clf = df[features_clf]
y_clf = df['pass_fail']

X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

acc_dt = accuracy_score(y_test, y_pred_dt)
print(f"   Accuracy: {acc_dt:.4f}")
print(classification_report(y_test, y_pred_dt, target_names=['Fail','Pass']))

cm_dt = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fail','Pass'], yticklabels=['Fail','Pass'])
plt.title(f'Decision Tree — Confusion Matrix\nAccuracy: {acc_dt:.4f}')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dt_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   → Saved: dt_confusion_matrix.png\n")

# ── STEP 5C: Random Forest ────────────────────────────────────
print("✅ Step 5C: Random Forest Classifier")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf  = rf.predict_proba(X_test)[:, 1]

acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"   Accuracy: {acc_rf:.4f}")
print(classification_report(y_test, y_pred_rf, target_names=['Fail','Pass']))

cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Fail','Pass'], yticklabels=['Fail','Pass'])
plt.title(f'Random Forest — Confusion Matrix\nAccuracy: {acc_rf:.4f}')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'rf_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   → Saved: rf_confusion_matrix.png\n")

# ── STEP 6: ROC Curve ─────────────────────────────────────────
print("✅ Step 6: ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob_rf)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Random Forest (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Random Forest')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   → Saved: roc_curve.png\n")

# ── STEP 7: Feature Importance ────────────────────────────────
print("✅ Step 7: Feature Importance")
importances = rf.feature_importances_
feat_df = pd.DataFrame({'Feature': features_clf, 'Importance': importances})
feat_df = feat_df.sort_values('Importance', ascending=True)

plt.figure(figsize=(8, 5))
plt.barh(feat_df['Feature'], feat_df['Importance'], color='steelblue', edgecolor='white')
plt.xlabel('Importance Score')
plt.title('Feature Importance — Random Forest')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   → Saved: feature_importance.png\n")

# ── STEP 8: Summary ───────────────────────────────────────────
print("=" * 50)
print("  MODEL COMPARISON SUMMARY")
print("=" * 50)
print(f"  Linear Regression  →  R² Score  : {r2:.4f}")
print(f"  Decision Tree      →  Accuracy  : {acc_dt:.4f}")
print(f"  Random Forest      →  Accuracy  : {acc_rf:.4f}  ← Best")
print(f"  Random Forest ROC  →  AUC       : {roc_auc:.4f}")
print("=" * 50)
print(f"\n🎉 All 6 plots saved in: {OUTPUT_DIR}")
