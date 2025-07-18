import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import joblib
import os

# Load data
df = pd.read_csv("train.csv")

# Feature selection
features = [
    "OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF",
    "GarageArea", "FullBath", "YearBuilt"
]
target = "SalePrice"

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# Start MLflow run
with mlflow.start_run():
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)

    input_example = X_test.iloc[:1]
    signature = infer_signature(X_test, model.predict(X_test))

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="house_price_model_rf",
        signature=signature,
        input_example=input_example
    )

    # Optional: Save locally for FastAPI use
    os.makedirs("saved_model", exist_ok=True)
    joblib.dump(model, "saved_model/model.pkl")
    print("Model saved locally to saved_model/model.pkl")
