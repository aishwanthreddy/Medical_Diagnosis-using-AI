import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import io
import numpy as np  # Important: Keep the numpy import!
import os

# Load the diabetes dataset from the provided string.
csv_string = """Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
8,183,64,0,0,23.3,0.672,32,1
1,89,66,23,94,28.1,0.167,21,0
0,137,40,35,168,43.1,2.288,33,1
5,116,74,0,0,25.6,0.201,30,0
3,78,50,32,88,31,0.248,26,1
10,115,0,0,0,35.3,0.134,29,0
2,197,70,45,543,30.5,0.158,53,1
8,125,96,0,0,0,0.232,54,1
4,110,92,0,0,37.6,0.191,30,0
10,168,74,0,0,38,0.537,34,1
10,139,80,0,0,27.1,1.441,57,0
1,189,60,23,846,30.1,0.398,59,1
1,146,56,0,0,29.7,0.564,29,0
2,71,70,27,0,28,0.586,22,0
7,103,66,32,0,39.1,0.344,31,1
7,105,0,0,0,0,0.305,24,0
1,103,80,11,82,19.4,0.491,22,0
1,101,50,15,36,24.2,0.526,26,0
5,88,66,21,23,24.4,0.342,30,0
8,176,90,34,300,33.7,0.467,58,1
7,150,66,42,342,34.7,0.718,42,0
1,73,50,10,0,23,0.248,21,0
7,187,68,39,304,37.7,0.254,41,1
0,100,88,60,110,46.8,0.962,31,0
0,146,82,0,0,40.5,1.781,44,0
0,105,64,41,142,41.5,0.173,22,0
2,84,0,0,0,0,0.304,21,0
8,133,72,0,0,32.9,0.27,39,1
5,44,62,0,0,25,0.587,36,0
2,141,58,34,128,25.4,0.699,24,0
7,114,66,0,0,32.8,0.258,42,1
5,99,68,0,0,22.2,0.145,32,0
"""

df = pd.read_csv(io.StringIO(csv_string))

# Separate features (X) and target (y).
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Handle missing values (replace 0 with NaN for relevant columns, then impute)
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
X[cols_to_replace] = X[cols_to_replace].replace(0, np.nan)  # Corrected line!

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Scale the features.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the logistic regression model.
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the trained model and scaler to the Models directory.
joblib.dump(model, 'Models/diabetes_model.joblib')
joblib.dump(scaler, 'Models/scaler.joblib')

print("Model retrained and saved as Models/diabetes_model.joblib")
print("Scaler saved as Models/scaler.joblib")

#Example of how to load the model and make a prediction.
if os.path.exists('Models/diabetes_model.joblib') and os.path.exists('Models/scaler.joblib'):
    loaded_model = joblib.load('Models/diabetes_model.joblib')
    loaded_scaler = joblib.load('Models/scaler.joblib')

    example_input = [[6,148,72,35,0,33.6,0.627,50]]
    scaled_example = loaded_scaler.transform(example_input)
    prediction = loaded_model.predict(scaled_example)
    print(f"Example prediction: {prediction}")
else:
    print("Error: Models/diabetes_model.joblib or Models/scaler.joblib not found.")