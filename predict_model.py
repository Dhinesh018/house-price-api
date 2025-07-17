import pandas as pd
import mlflow.pyfunc

# Load model from MLflow model registry
model = mlflow.pyfunc.load_model(model_uri="models:/house_price_model_rf/1")

# ✅ Updated input data with all 7 features
input_data = pd.DataFrame([{
    "OverallQual": 7,
    "GrLivArea": 1710,
    "GarageCars": 2,
    "GarageArea": 480,
    "TotalBsmtSF": 856,
    "FullBath": 2,
    "YearBuilt": 2005
}])

# Make prediction
prediction = model.predict(input_data)
print(f"Predicted Price: ₹{prediction[0]:,.2f}")

