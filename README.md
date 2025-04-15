# Food Risk Predictor API

This is a simple FastAPI-based backend for analyzing food health risks using ingredients, nutrition info, and category.

## Endpoints
- POST `/predict-risk` – returns risk score and level

## Deployment
Tested to deploy on **Render** using this command in `start.sh`:
```
uvicorn food_risk_api:app --host 0.0.0.0 --port 10000
```

Models used here are dummy and must be replaced with your trained versions later.
