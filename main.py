import numpy as np
from typing import Union
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel  # Validators

# from mangum import Mangum

main = FastAPI()

model = pickle.load(open(r"random_forest_model.pkl", "rb"))
# The logistic regression model was used as it was determined to be the best fit based on evaluation.
scaler = pickle.load(open(r"scaler.pkl", "rb"))

label_encoders = pickle.load(open(r"label_encoders.pkl", "rb"))



# The text type validation
class TemplateRequest(BaseModel):
    district: str
    season:str
    rainfall:float
    temperature:float
    soil_moisture:float
    pest_infestation:float
    fertilizer_usage:float



@main.post("/predict-yield/")
def predict_coffee(request: TemplateRequest):
    try:
        encoded_district = label_encoders['District'].transform([request.district])[0]
        encoded_season = label_encoders['Season'].transform([request.season])[0]
        input_vector = np.array([[encoded_district, encoded_season, request.rainfall, request.temperature,
                                    request.soil_moisture, request.pest_infestation, request.fertilizer_usage]])
        input_scaled = scaler.transform(input_vector)
        predicted_log_yield = model.predict(input_scaled)[0]

        prediction = model.predict(input_scaled)[0]
        predicted_yield = np.expm1(predicted_log_yield)
        predicted_yield = round(predicted_yield, 2)

        return {
        "message": "Coffee yield prediction was successful.",
        "predicted_coffee_yield": predicted_yield}
    except Exception as e:
        raise HTTPException(status_code=204, detail=f"Error Occured : {e}")


# handler = Mangum(main)
