import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Load Models and Datasets ---
try:
    parkinsons_model = joblib.load('Models/parkinsons_model.joblib')
    print("Parkinson's model loaded successfully")
    parkinsons_df = pd.read_csv('Datasets/parkinson_data.csv')
    print("Parkinson's data loaded successfully")
except Exception as e:
    st.error(f"Error loading Parkinson's files: {e}")
    st.stop()

# ... (rest of your model loading, as you provided) ...

# --- Parkinson's Prediction Function ---
import streamlit as st
import pandas as pd
import joblib  # Import joblib for loading the scaler and model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from io import StringIO

# ... (your predict_parkinsons function from the previous response) ...

def parkinsons_prediction():
    st.subheader("Parkinson's Disease Prediction 🧠")
    st.write("Enter Patient Information 📝")

    # Input fields for the features
    MDVP_Fo_Hz = st.number_input("MDVP:Fo(Hz)", value=120.0)
    MDVP_Fhi_Hz = st.number_input("MDVP:Fhi(Hz)", value=150.0)
    MDVP_Flo_Hz = st.number_input("MDVP:Flo(Hz)", value=80.0)
    MDVP_Jitter_percent = st.number_input("MDVP:Jitter(%)", value=0.008)
    MDVP_Jitter_Abs = st.number_input("MDVP:Jitter(Abs)", value=0.00007)
    MDVP_RAP = st.number_input("MDVP:RAP", value=0.004)
    MDVP_PPQ = st.number_input("MDVP:PPQ", value=0.006)
    Jitter_DDP = st.number_input("Jitter:DDP", value=0.012)
    MDVP_Shimmer = st.number_input("MDVP:Shimmer", value=0.045)
    MDVP_Shimmer_dB = st.number_input("MDVP:Shimmer(dB)", value=0.43)
    Shimmer_APQ3 = st.number_input("Shimmer:APQ3", value=0.022)
    Shimmer_APQ5 = st.number_input("Shimmer:APQ5", value=0.032)
    MDVP_APQ = st.number_input("MDVP:APQ", value=0.030)
    Shimmer_DDA = st.number_input("Shimmer:DDA", value=0.066)
    NHR = st.number_input("NHR", value=0.022)
    HNR = st.number_input("HNR", value=21.0)
    RPDE = st.number_input("RPDE", value=0.41)
    DFA = st.number_input("DFA", value=0.82)
    spread1 = st.number_input("spread1", value=-4.8)
    spread2 = st.number_input("spread2", value=0.27)
    D2 = st.number_input("D2", value=2.3)
    PPE = st.number_input("PPE", value=0.28)

    if st.button("Predict Parkinson's Disease 🧪"):
        input_data = {
            'MDVP:Fo(Hz)': MDVP_Fo_Hz,
            'MDVP:Fhi(Hz)': MDVP_Fhi_Hz,
            'MDVP:Flo(Hz)': MDVP_Flo_Hz,
            'MDVP:Jitter(%)': MDVP_Jitter_percent,
            'MDVP:Jitter(Abs)': MDVP_Jitter_Abs,
            'MDVP:RAP': MDVP_RAP,
            'MDVP:PPQ': MDVP_PPQ,
            'Jitter:DDP': Jitter_DDP,
            'MDVP:Shimmer': MDVP_Shimmer,
            'MDVP:Shimmer(dB)': MDVP_Shimmer_dB,
            'Shimmer:APQ3': Shimmer_APQ3,
            'Shimmer:APQ5': Shimmer_APQ5,
            'MDVP:APQ': MDVP_APQ,
            'Shimmer:DDA': Shimmer_DDA,
            'NHR': NHR,
            'HNR': HNR,
            'RPDE': RPDE,
            'DFA': DFA,
            'spread1': spread1,
            'spread2': spread2,
            'D2': D2,
            'PPE': PPE,
        }

        try:
            # Load the scaler and model
            scaler = joblib.load('Models/Scaler.joblib')
            model = joblib.load('Models/parkinsons_model.joblib')

            # Scale the input data
            scaled_input = scaler.transform(pd.DataFrame([input_data]))

            # Make the prediction
            prediction = model.predict(scaled_input)[0]

            st.write(f"Prediction: {prediction}")

        except FileNotFoundError:
            st.error("Error: Model or scaler file not found. Please ensure they are in the 'Models' directory.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# ... (rest of your Streamlit app code) ...

# Example of how to call the function:
if st.session_state.get("selected_disease") == "Parkinson's Disease":
    parkinsons_prediction()


















import streamlit as st
import joblib
import pandas as pd

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

def diabetes_prediction():
    st.title("Diabetes Prediction 🩺")
    st.markdown("## Enter Patient Information 📝")
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input('Pregnancies', help="Number of pregnancies")
        glucose = st.number_input('Glucose', help="Plasma glucose concentration a 2 hours in an oral glucose tolerance test")
        blood_pressure = st.number_input('BloodPressure', help="Diastolic blood pressure (mm Hg)")
        skin_thickness = st.number_input('SkinThickness', help="Triceps skin fold thickness (mm)")

    with col2:
        insulin = st.number_input('Insulin', help="2-Hour serum insulin (mu U/ml)")
        bmi = st.number_input('BMI', help="Body mass index (weight in kg/(height in m)^2)")
        diabetes_pedigree_function = st.number_input('DiabetesPedigreeFunction', help="Diabetes pedigree function")
        age = st.number_input('Age', help="Age (years)")

    if st.button('Predict Diabetes 🧪'):
        # Check if all inputs are numbers and not zero
        if (isinstance(pregnancies, (int, float)) and
            isinstance(glucose, (int, float)) and glucose != 0 and
            isinstance(blood_pressure, (int, float)) and blood_pressure != 0 and
            isinstance(skin_thickness, (int, float)) and skin_thickness != 0 and
            isinstance(insulin, (int, float)) and insulin != 0 and
            isinstance(bmi, (int, float)) and bmi != 0 and
            isinstance(diabetes_pedigree_function, (int, float)) and
            isinstance(age, (int, float)) and age != 0):

            input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]],
                                      columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

            try:
                if not os.path.exists('Models/diabetes_model.joblib') or not os.path.exists('Models/scaler.joblib'):
                    st.error("Model or scaler file not found. Please check the file paths.")
                    return

                scaler = joblib.load('Models/scaler.joblib')
                model = joblib.load('Models/diabetes_model.joblib')

                cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
                input_data[cols_to_replace] = input_data[cols_to_replace].replace(0, np.nan)

                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='mean')
                input_data = pd.DataFrame(imputer.fit_transform(input_data), columns=input_data.columns)

                scaled_input = scaler.transform(input_data)
                prediction = model.predict(scaled_input)

                if prediction[0] == 1:
                    st.error('The model predicts diabetes. 🔴')
                else:
                    st.success('The model predicts no diabetes. ✅')
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("Please enter valid numerical values for all input fields. Glucose, BloodPressure, SkinThickness, Insulin, BMI, and Age cannot be 0.")

def heart_disease_prediction():
    st.title("Heart Disease Prediction 🫀")
    st.markdown("## Enter Patient Information 📝")
    # ... (Similar input fields for heart disease features) ...
    age = st.number_input('age')
    sex = st.number_input('sex')
    cp = st.number_input('cp')
    trestbps = st.number_input('trestbps')
    chol = st.number_input('chol')
    fbs = st.number_input('fbs')
    restecg = st.number_input('restecg')
    thalach = st.number_input('thalach')
    exang = st.number_input('exang')
    oldpeak = st.number_input('oldpeak')
    slope = st.number_input('slope')
    ca = st.number_input('ca')
    thal = st.number_input('thal')
    if st.button('Predict Heart Disease 🧪'):
        input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
        try:
            scaler = joblib.load('Models/heart_disease_scaler.joblib')
            scaled_input = scaler.transform(input_data)
            model = joblib.load('Models/heart_disease_model.joblib')
            prediction = model.predict(scaled_input)
            if prediction[0] == 1:
                st.error('The model predicts heart disease. 🔴')
            else:
                st.success('The model predicts no heart disease. ✅')
        except:
            st.error('Model or Scaler not found')

def hypothyroid_prediction():
    st.title("Hypothyroid Prediction 🦋")
    # ... (Similar input fields for hypothyroid features) ...
    age = st.number_input('age')
    sex = st.number_input('sex')
    on_thyroxine = st.number_input('on_thyroxine')
    query_on_thyroxine = st.number_input('query_on_thyroxine')
    on_antithyroid_medication = st.number_input('on_antithyroid_medication')
    sick = st.number_input('sick')
    pregnant = st.number_input('pregnant')
    thyroid_surgery = st.number_input('thyroid_surgery')
    I131_treatment = st.number_input('I131_treatment')
    query_hypothyroid = st.number_input('query_hypothyroid')
    query_hyperthyroid = st.number_input('query_hyperthyroid')
    lithium = st.number_input('lithium')
    goitre = st.number_input('goitre')
    tumor = st.number_input('tumor')
    psych = st.number_input('psych')
    TSH = st.number_input('TSH')
    T3 = st.number_input('T3')
    TT4 = st.number_input('TT4')
    T4U = st.number_input('T4U')
    FTI = st.number_input('FTI')

    if st.button('Predict Hypothyroid 🧪'):
     input_data = pd.DataFrame([[age, sex, on_thyroxine, query_on_thyroxine, on_antithyroid_medication, sick, pregnant, thyroid_surgery, I131_treatment, query_hypothyroid, query_hyperthyroid, lithium, goitre, tumor, psych, TSH, T3, TT4, T4U, FTI]], columns=['age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'psych', 'TSH', 'T3', 'TT4', 'T4U', 'FTI'])
    try:
        scaler = joblib.load('Models/heart_disease_scaler.joblib')
        scaled_input = scaler.transform(input_data)
        model = joblib.load('Models/hypothyroid_model.joblib')
        prediction = model.predict(scaled_input)
        if prediction[0] == 1:
            st.error('The model predicts hypothyroid. 🔴')
        else:
            st.success('The model predicts no hypothyroid. ✅')
    except:
        st.error('Model or Scaler not found')

def lungs_prediction():
    st.title("Lung Cancer Prediction 🫁")
    # ... (Similar input fields for lung cancer features) ...
    GENDER = st.number_input('GENDER')
    AGE = st.number_input('AGE')
    SMOKING = st.number_input('SMOKING')
    YELLOW_FINGERS = st.number_input('YELLOW_FINGERS')
    ANXIETY = st.number_input('ANXIETY')
    PEER_PRESSURE = st.number_input('PEER_PRESSURE')
    CHRONIC_DISEASE = st.number_input('CHRONIC DISEASE')
    FATIGUE = st.number_input('FATIGUE ')
    ALLERGY = st.number_input('ALLERGY ')
    WHEEZING = st.number_input('WHEEZING')
    ALCOHOL_CONSUMING = st.number_input('ALCOHOL CONSUMING')
    COUGHING = st.number_input('COUGHING')
    SHORTNESS_OF_BREATH = st.number_input('SHORTNESS OF BREATH')
    SWALLOWING_DIFFICULTY = st.number_input('SWALLOWING DIFFICULTY')
    CHEST_PAIN = st.number_input('CHEST PAIN')

    if st.button('Predict Lung Cancer 🧪'):
        input_data = pd.DataFrame([[GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN]], columns=['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN'])
        try:
            scaler = joblib.load('Models/Scaler.joblib')
            scaled_input = scaler.transform(input_data)
            model = joblib.load('Models/lung_cancer_gb.joblib')
            prediction = model.predict(scaled_input)
            if prediction[0] == 1:
                st.error('The model predicts lung cancer. 🔴')
            else:
                st.success('The model predicts no lung cancer. ✅')
        except:
            st.error('Model or Scaler not found')

# Example of how to place the function into your main app.py file.
selected_disease = st.sidebar.selectbox("Select Disease", ["Parkinson's Disease", "Diabetes", "Heart Disease", "Hypothyroid", "Lungs"])

if selected_disease == "Diabetes":
    diabetes_prediction()
elif selected_disease == "Heart Disease":
    heart_disease_prediction()
elif selected_disease == "Hypothyroid":
    hypothyroid_prediction()
elif selected_disease == "Lungs":
    lungs_prediction()
elif selected_disease == "Parkinson's Disease":
    parkinsons_prediction()


















