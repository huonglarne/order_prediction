import pandas as pd
from joblib import load
from fastapi import FastAPI
import uvicorn
from src.inference_utils import get_product_latest_features, postprocess_prediction
from src.constants import MODEL_JOBLIB_CHECKPOINT

app = FastAPI()
model = load(MODEL_JOBLIB_CHECKPOINT)
weekly_sales_data = pd.read_csv(
    "data/weekly_sales_data.csv"
)  # this is pulled from the database or feature store


@app.get("/predict/{product_id}")
async def predict_product_sale(product_id: int):
    features = get_product_latest_features(product_id, weekly_sales_data)
    result = model.predict(features)
    result = postprocess_prediction(result)[0]

    return {"sales for next week": result}
