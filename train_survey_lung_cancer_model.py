# train_survey_lung_cancer_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load the lung cancer survey dataset
df = pd.read_csv('Datasets/survey_lung_cancer.csv')

# Preprocessing (adjust based on your dataset)
for column in df.select_dtypes(include='object').columns:
    df[column] = LabelEncoder().fit_transform(df[column])

# Separate features and target (replace 'LUNG_CANCER' if your target column is different)
X = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model and scaler
joblib.dump(model, 'Models/lung_cancer_model.joblib')
joblib.dump(scaler, 'Models/lung_cancer_scaler.joblib')

print("Lung cancer model trained and saved.")