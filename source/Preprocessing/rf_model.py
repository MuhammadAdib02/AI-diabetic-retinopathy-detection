# rf_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib

# Load the cleaned dataset
file_path = 'C:\\Users\\Muhammad Adib\\TRY2\\MyMedicalApp\\Preprocessing\\cleaned_diabetic_retinopathy_data.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')

# Split data into features and target
X = data.drop('Diabetic_Retinopathy', axis=1)
y = data['Diabetic_Retinopathy']

# Handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate and save the model
y_pred = rf_model.predict(X_test)
print("=== Random Forest Model ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model to .pkl
joblib.dump(rf_model, 'random_forest_model.pkl')
print("Random Forest model saved as random_forest_model.pkl.")
