import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import io
import numpy as np #Add this import

def retrain_heart_disease_scaler(data_string, scaler_path='Models/heart_disease_scaler.joblib'):
    """
    Retrains the StandardScaler using the given dataset and saves it to a .joblib file.

    Args:
        data_string (str): A string containing the dataset in CSV format.
        scaler_path (str): The path to save the retrained scaler.
    """
    try:
        # Load the dataset from the string
        data = pd.read_csv(io.StringIO(data_string))

        # Select the numerical columns to scale
        numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'] #add more columns if you need to.
        X_scaled = data[numerical_cols]

        # Handle potential zero values in 'trestbps' and 'chol' before scaling
        X_scaled['trestbps'] = X_scaled['trestbps'].replace(0, np.nan)
        X_scaled['chol'] = X_scaled['chol'].replace(0, np.nan)

        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X_scaled = pd.DataFrame(imputer.fit_transform(X_scaled), columns=X_scaled.columns)

        # Train the scaler
        scaler = StandardScaler()
        scaler.fit(X_scaled)

        # Save the scaler
        joblib.dump(scaler, scaler_path)

        return f"Scaler retrained and saved to {scaler_path}"

    except Exception as e:
        return f"An error occurred during scaler retraining: {e}"

# Example Usage (with your provided dataset string):
data_string = """
age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target
63,1,3,145,233,1,0,150,0,2.3,0,0,1,1
37,1,2,130,250,0,1,187,0,3.5,0,0,2,1
41,0,1,130,204,0,0,172,0,1.4,2,0,2,1
56,1,1,120,236,0,1,178,0,0.8,2,0,2,1
57,0,0,120,354,0,1,163,1,0.6,2,0,2,1
57,1,0,140,192,0,1,148,0,0.4,1,0,1,1
56,0,1,140,294,0,0,153,0,1.3,1,0,2,1
44,1,1,120,263,0,1,173,0,0,2,0,3,1
52,1,2,172,199,1,1,162,0,0.5,2,0,3,1
57,1,2,150,168,0,1,174,0,1.6,2,0,2,1
54,1,0,140,239,0,1,160,0,1.2,2,0,2,1
48,0,2,130,275,0,1,139,0,0.2,2,0,2,1
49,1,1,130,266,0,1,171,0,0.6,2,0,2,1
64,1,3,110,211,0,0,144,1,1.8,1,0,2,1
58,0,3,150,283,1,0,162,0,1,2,0,2,1
50,0,2,120,219,0,1,158,0,1.6,1,0,2,1
58,0,2,120,340,0,1,172,0,0,2,0,2,1
66,0,3,150,226,0,1,114,0,2.6,0,0,2,1
43,1,0,150,247,0,1,171,0,1.5,2,0,2,1
69,0,3,140,239,0,1,151,0,1.8,2,2,2,1
59,1,0,135,234,0,1,161,0,0.5,1,0,3,1
# ... (rest of your dataset string)
"""




result = retrain_heart_disease_scaler(data_string)
print(result)