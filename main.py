from typing import Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

model_path = "./trainedModel2.h5"
binary_model = load_model(model_path)

class Item(BaseModel):
    gender: str
    age: int
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str
    

@app.post("/predict")
def predict(item: Item):
    # Map categorical variables to numerical values
    gender_mapping = {'Male': 0, 'Female': 1, 'Other': 2}
    ever_married_mapping = {'No': 0, 'Yes': 1}
    work_type_mapping = {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4}
    residence_type_mapping = {'Urban': 0, 'Rural': 1}
    smoking_status_mapping = {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3}

    # Convert input data to a NumPy array
    input_data = np.array([[
        gender_mapping.get(item.gender, 2),
        item.age, item.hypertension, item.heart_disease,
        ever_married_mapping.get(item.ever_married, 2),
        work_type_mapping.get(item.work_type, 4),
        residence_type_mapping.get(item.Residence_type, 1),
        item.avg_glucose_level, item.bmi,
        smoking_status_mapping.get(item.smoking_status, 3)
    ]])

    # Make prediction using the loaded model
    prediction = binary_model.predict(input_data)

    # The output is a probability, you can convert it to a class (0 or 1) based on a threshold
    threshold = 0.5
    predicted_class = 1 if prediction[0, 0] > threshold else 0

    # Define output messages based on the predicted class
    output_messages = {
        0: "No Brain Stroke predicted for the patient.",
        1: "Brain Stroke may develop for the patient!"
    }

    return {"prediction": predicted_class, "message": output_messages[predicted_class]}
