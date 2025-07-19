# 🏡 House Price Prediction API with MLflow, FastAPI & Docker

This project is an end-to-end **MLOps application** that predicts house prices based on key features using a trained **Random Forest** model. It demonstrates model training, experiment tracking with **MLflow**, API deployment with **FastAPI**, and containerization with **Docker** — deployed on **Render**.

---

## 🚀 Project Features

- ✅ Trained RandomForestRegressor on Ames Housing Dataset
- ✅ MLflow tracking for metrics, parameters & artifacts
- ✅ Model versioning & registry
- ✅ REST API using FastAPI for real-time prediction
- ✅ Dockerized and deployed on Render
- ✅ 7 essential house features used for prediction

---

## 🧠 Technologies Used

| Tool        | Purpose                              |
|-------------|--------------------------------------|
| `Python`    | Core programming                     |
| `scikit-learn` | Machine learning                   |
| `pandas`    | Data processing                      |
| `MLflow`    | Experiment tracking & model registry |
| `FastAPI`   | API serving                          |
| `Docker`    | Containerization                     |
| `Render`    | Deployment                           |

---

## 📁 Project Structure

├── train_model.py # Train and log model with MLflow
├── main.py # FastAPI application
├── Dockerfile # Docker configuration
├── requirements.txt # Project dependencies
├── saved_model/ # MLflow-saved model
├── mlruns/ # MLflow run artifacts
└── README.md


---

## 🧾 API Input Format


POST /predict

{
  "OverallQual": 7,
  "GrLivArea": 1710,
  "GarageCars": 2,
  "TotalBsmtSF": 856,
  "GarageArea": 548,
  "FullBath": 2,
  "YearBuilt": 2003
}
✅ Response:

{
  "predicted_price": "$208678.23"
}
⚙️ Run Locally
1. Clone the repository

git clone https://github.com/Dhinesh018/house-price-mlflow-api.git
cd house-price-mlflow-api

2. Install dependencies

pip install -r requirements.txt
3. Train the model & log to MLflow

python train_model.py
4. Run the API

uvicorn main:app --reload
🐳 Docker Support
1. Build Docker Image

docker build -t house-price-api .
2. Run Container

docker run -p 8000:8000 house-price-api
🌐 Live Demo
🔗 API Endpoint: https://your-render-url.onrender.com
🧪 Swagger UI: /docs

📦 GitHub Repository
📂 https://github.com/Dhinesh018/house-price-mlflow-api

📬 Contact
Built by Dhinesh Kumar S
💼 LinkedIn | 💻 GitHub

📌 License:
This project is licensed under the MIT License.



---











