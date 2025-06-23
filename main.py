from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from pathlib import Path
from fastapi.responses import RedirectResponse  # ✅ Added

# Load environment variables
env_path = Path(".env-live")
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# Load dataset
engineDf = pd.read_excel(
    "https://www.dropbox.com/scl/fi/dcoz9yw3f8yywtzy82f4z/Engine.xlsx?rlkey=n53hfjjsrddywktksj156jra5&dl=1")

# Prepare features and target
X = engineDf[['Miles', 'Load', 'Speed', 'Oil']]
y = engineDf['Time']

# Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

# Register the model
model_registry = {
    "TING_261301330_engineModel": model,
    "TING_261301330_scaler": scaler
}

# FastAPI setup
app = FastAPI()

# Input schema


class InputData(BaseModel):
    Miles: float
    Load: float
    Speed: float
    Oil: float

# POST method for prediction


@app.post("/dashboard")
def predict(data: InputData):
    model = model_registry["TING_261301330_engineModel"]
    scaler = model_registry["TING_261301330_scaler"]

    input_df = pd.DataFrame([{
        "Miles": data.Miles,
        "Load": data.Load,
        "Speed": data.Speed,
        "Oil": data.Oil
    }])

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    return {"predicted_years_until_overhaul": round(prediction, 2)}

# GET method — redirect /dashboard to Swagger UI


@app.get("/dashboard", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse(url="/docs")


@app.get("/")
def root():
    return {"message": "Hello World from TING_261301330_engineModel"}
