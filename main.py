from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import logging
import os

# Load model from local directory
model = joblib.load(os.path.join("saved_model", "model.pkl"))

# Set up logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

class HouseData(BaseModel):
    OverallQual: int
    GrLivArea: int
    GarageCars: int
    TotalBsmtSF: int
    GarageArea: int
    FullBath: int
    YearBuilt: int

@app.get("/")
def home():
    return {"message": "Welcome to the House Price Prediction API using a locally saved model!"}

@app.post("/predict")
def predict(data: HouseData):
    input_data = data.dict()
    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]

    logging.info(f"Input data: {input_data}")
    logging.info(f"Predicted price: {prediction}")

    return {"predicted_price": f"${prediction:.2f}"}
