import joblib
import pickle
import os

def convert_sav_to_joblib(sav_file_path, joblib_file_path):
    """Converts a .sav model file to .joblib format."""
    try:
        with open(sav_file_path, 'rb') as file:
            model = pickle.load(file)

        joblib.dump(model, joblib_file_path)
        print(f"Model converted: {sav_file_path} -> {joblib_file_path}")

    except Exception as e:
        print(f"Error converting {sav_file_path}: {e}")

# Example Usage (Modify these paths to your actual file paths)
sav_files = [
    "Models/diabetes_model.sav",
    "Models/heart_disease_model.sav",
    "Models/hypothyroid_model.sav",
    "Models/preprocessed_hypothyroid_model.sav",
    "Models/lungs_disease_model.sav",
    "Models/survey_lung_cancer_model.sav"
]

for sav_file in sav_files:
    if os.path.exists(sav_file):
        joblib_file = sav_file.replace(".sav", ".joblib")
        convert_sav_to_joblib(sav_file, joblib_file)
    else:
        print(f"File not found: {sav_file}")