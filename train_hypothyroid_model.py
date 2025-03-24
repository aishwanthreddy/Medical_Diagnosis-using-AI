import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load the dataset
df = pd.read_csv('Datasets/hypothyroid.csv')

# Count missing values before replacement
print("Missing values before replacement:")
print(df.isin(['?']).sum())

# Preprocessing
df = df.replace('?', pd.NA)

# Count missing values after replacement
print("\nMissing values after replacement:")
print(df.isna().sum())

# Impute numerical columns with mean
for column in df.select_dtypes(include=['number']).columns:
    df[column] = df[column].fillna(df[column].mean())

# Impute categorical columns with mode, handling empty modes
for column in df.select_dtypes(include=['object']).columns:
    mode_values = df[column].mode()
    if not mode_values.empty:
        df[column] = df[column].fillna(mode_values.iloc[0])
    else:
        df[column] = df[column].fillna('unknown')  # Or another appropriate default

# Verify that there are no more null values.
print("\nMissing values after imputation:")
print(df.isna().sum())

# Convert categorical columns to numerical using LabelEncoder
for column in df.select_dtypes(include=['object']).columns:
    df[column] = LabelEncoder().fit_transform(df[column])

# Separate features and target
X = df.drop('binaryClass', axis=1)
y = df['binaryClass']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model and scaler
joblib.dump(model, 'Models/hypothyroid_model.joblib')
joblib.dump(scaler, 'Models/hypothyroid_scaler.joblib')

print("Hypothyroid model trained and saved.")