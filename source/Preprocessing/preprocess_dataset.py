import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the raw dataset
file_path = 'C:\\Users\\Muhammad Adib\\TRY2\\MyMedicalApp\\Preprocessing\\Diabetic Retinopathy dataset full (from dr azimah).xlsx'
data = pd.ExcelFile(file_path).parse('Sheet1')

# Drop unnecessary columns
columns_to_drop = ['ID', 'Name', 'FirstVisit']
data = data.drop(columns=columns_to_drop, errors='ignore')

# Handle missing data
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Fill missing values
for col in numerical_columns:
    data[col].fillna(data[col].median(), inplace=True)
for col in categorical_columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Encode categorical variables
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Scale numerical features
scaler = MinMaxScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Save the preprocessed data
output_path = 'C:\\Users\\Muhammad Adib\\TRY2\\MyMedicalApp\\Preprocessing\\preprocessed_diabetic_retinopathy_data (2).xlsx'
data.to_excel(output_path, index=False)
print(f"Preprocessed data saved to {output_path}")

