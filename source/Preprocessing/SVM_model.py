# svm_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib

# Load the cleaned dataset
file_path = 'C:\\Users\\Muhammad Adib\\TRY2\\MyMedicalApp\\Preprocessing\\cleaned_diabetic_retinopathy_data.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')

# Split the data into features and target
X = data.drop('Diabetic_Retinopathy', axis=1)
y = data['Diabetic_Retinopathy']

# Apply SMOTE for handling class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train SVM model
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Evaluate the model
predictions = svm_model.predict(X_test)
print("=== Support Vector Machine Model ===")
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))

# Save the model
joblib.dump(svm_model, 'svm_model.pkl')
print("Support Vector Machine model saved as svm_model.pkl.")
