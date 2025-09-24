import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
import pickle
from imblearn.over_sampling import SMOTE  # Import SMOTE for upsampling

# Step 1: Load Preprocessed Data
file_path = 'C:\\Users\\Muhammad Adib\\TRY2\\MyMedicalApp\\Preprocessing\\preprocessed_diabetic_retinopathy_data (2).xlsx'
data = pd.read_excel(file_path)

# Step 2: Split Features and Target
X = data.drop(columns=['Diabetic_Retinopathy'], errors='ignore')  # Features
y = data['Diabetic_Retinopathy']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Apply SMOTE to Upsample the Training Data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"Original Training Data Shape: {X_train.shape}, {y_train.shape}")
print(f"Resampled Training Data Shape: {X_train_res.shape}, {y_train_res.shape}")

# Step 4: Function to Train, Evaluate, and Save Ensemble Models with Visualizations
def train_and_save_ensemble(model, model_name, param_grid=None):
    print(f"Training {model_name}...")
    if param_grid:
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train_res, y_train_res)
        model = grid_search.best_estimator_
        print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
    else:
        model.fit(X_train_res, y_train_res)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    if roc_auc:
        print(f"{model_name} ROC-AUC: {roc_auc:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    model_file = f"C:\\Users\\Muhammad Adib\\TRY2\\MyMedicalApp\\Preprocessing\\{model_name.replace(' ', '_')}_Ensemble_Upsampled.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"{model_name} saved to {model_file}")
    
    # Visualizations
    print(f"Generating visualizations for {model_name}...")

    # Feature Importance (if applicable)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]  # Sort features by importance
        feature_names = X.columns

        plt.figure(figsize=(10, 6))
        plt.title(f"{model_name} - Feature Importance")
        plt.bar(range(X.shape[1]), importances[indices], align="center")
        plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
        plt.tight_layout()
        plt.show()
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()

    # ROC Curve (if applicable)
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")  # Diagonal line
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} - Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.show()

# Step 5: Ensemble Methods
# XGBoost
train_and_save_ensemble(XGBClassifier(eval_metric='logloss', random_state=42), "XGBoost")

# AdaBoost
train_and_save_ensemble(AdaBoostClassifier(random_state=42, algorithm='SAMME'), "AdaBoost")

meta_learner = LogisticRegression()


