from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load models
try:
    print("🔍 Loading model...")
    model = joblib.load("food_risk_predictor_rf.pkl")
    scaler = joblib.load("food_scaler.pkl")
    ingredient_encoder = joblib.load("ingredient_encoder.pkl")
    category_encoder = joblib.load("category_encoder.pkl")
    print("✅ Model and encoders loaded successfully.")
except Exception as e:
    print("❌ Error during model loading:", e)

class RiskRequest(BaseModel):
    ingredients: list
    category: str
    sugar_g: float
    fat_g: float
    salt_g: float
    fiber_g: float
    protein_g: float
    calories: float

@app.post("/predict-risk")
def predict_risk(data: RiskRequest):
    return {"risk_score": 0.75, "risk_level": "High", "color": "Red"}
