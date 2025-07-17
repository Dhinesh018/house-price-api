from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import logging

# Load the latest Production version of the registered model
model = mlflow.pyfunc.load_model(model_uri="models:/house_price_model_rf/6")

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
    return {"message": "Welcome to the House Price Prediction API using MLflow model!"}

@app.post("/predict")
def predict(data: HouseData):
    input_data = data.dict()
    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]

    # ✅ Log input and prediction
    logging.info(f"Input data: {input_data}")
    logging.info(f"Predicted price: {prediction}")

    return {"predicted_price": f"${prediction:.2f}"}
