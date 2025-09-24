import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
import numpy as np
import pickle
from xgboost import XGBClassifier  # Importing XGBoost

# Step 1: Load Preprocessed Data
file_path = r'C:\Users\Muhammad Adib\TRY2\MyMedicalApp\Preprocessing\Mostly.ai\combined_600_dataset.csv'
data = pd.read_csv(file_path)

# Step 2: Split Features and Target
X = data.drop(columns=['Diabetic_Retinopathy'], errors='ignore')  # Features
y = data['Diabetic_Retinopathy']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Function to Train, Evaluate, and Save Models with Visualizations
# Dictionary to store ROC curve data for comparison
roc_curves = {}

def train_and_save_model(model, model_name, param_grid=None):
    print(f"Training {model_name}...")
    if param_grid:
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
    else:
        model.fit(X_train, y_train)
    
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
    
    # Visualizations
    print(f"Generating visualizations for {model_name}...")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()

    # ROC Curve
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

        # Save ROC curve data for comparison
        roc_curves[model_name] = (fpr, tpr, roc_auc)

# Step 4: Train Models
# Logistic Regression
train_and_save_model(LogisticRegression(max_iter=1000, random_state=42), "Logistic Regression")

# Support Vector Machine
train_and_save_model(SVC(probability=True, random_state=42), "Support Vector Machine")

# Random Forest
train_and_save_model(RandomForestClassifier(random_state=42), "Random Forest")

# K-Nearest Neighbors
train_and_save_model(KNeighborsClassifier(), "K-Nearest Neighbors")

# AdaBoost
train_and_save_model(AdaBoostClassifier(random_state=42), "AdaBoost")

# XGBoost
train_and_save_model(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), "XGBoost")

# Step 5: Plot Combined ROC Curve for All Models
plt.figure(figsize=(10, 8))
for model_name, (fpr, tpr, roc_auc) in roc_curves.items():
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Random Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Comparison of ROC Curves")
plt.legend(loc="lower right")
plt.show()
