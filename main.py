from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load the trained model
model = joblib.load("model.joblib")

app = FastAPI()

class HouseData(BaseModel):
    OverallQual: int
    GrLivArea: float
    GarageCars: int
    TotalBsmtSF: float

@app.get("/")
def home():
    return {"message": "Welcome to the House Price Prediction API!"}

@app.post("/predict")
def predict(data: HouseData):
    input_data = [[
        data.OverallQual,
        data.GrLivArea,
        data.GarageCars,
        data.TotalBsmtSF
    ]]
    prediction = model.predict(input_data)[0]
    return {"predicted_price": prediction}
