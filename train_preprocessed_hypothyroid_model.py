# train_preprocessed_hypothyroid_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the preprocessed dataset
df = pd.read_csv('Datasets/preprocessed_hypothyroid.csv')

# Separate features and target
X = df.drop('binaryClass', axis=1)
y = df['binaryClass']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'Models/preprocessed_hypothyroid_model.joblib')

print("Preprocessed hypothyroid model trained and saved.")