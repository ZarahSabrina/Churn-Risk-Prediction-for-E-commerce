from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import numpy as np
import joblib
import logging
import traceback
import uvicorn

# === Inisialisasi Logging ke File ===
logging.basicConfig(
    filename="fastapi_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI()

# === Load model bundle ===
model_bundle = joblib.load("xgb_churn_model.joblib")
model = model_bundle["model"]
scaler = model_bundle["scaler"]
threshold = model_bundle["threshold"]
columns = model_bundle["columns"]

# === Mapping: state → region → one-hot ===
state_to_region = {
    "jammu & kashmir": "north", "himachal pradesh": "north", "punjab": "north", "uttar pradesh": "north",
    "delhi": "north", "rajasthan": "north", "uttaranchal": "north", "chhattisgarh": "north", "haryana": "north",
    "west bengal": "east", "orissa": "east", "arunachal pradesh": "east",
    "maharashtra": "west", "goa": "west", "gujarat": "west", "madhya pradesh": "west",
    "tamil nadu": "south", "karnataka": "south", "kerala": "south", "andhra pradesh": "south"
}

region_map = {
    "east":  [1, 0, 0, 0],
    "north": [0, 1, 0, 0],
    "south": [0, 0, 1, 0],
    "west":  [0, 0, 0, 1]
}

spending_map = {
    "low": 0,
    "medium": 1,
    "high": 2
}

def classify_spending(payment_value):
    if payment_value <= 61:
        return "low"
    elif payment_value <= 189:
        return "medium"
    else:
        return "high"

# === Schema Input ===
class ChurnRequest(BaseModel):
    price: float
    payment_value: float
    review_score: float
    customer_state: str

    class Config:
        schema_extra = {
            "example": {
                "price": 20,
                "payment_value": 200,
                "review_score": 4.5,
                "customer_state": "gujarat"
            }
        }

# === Routes ===
@app.get("/")
async def root():
    return {"message": "Customer Churn Prediction API - Manual Version"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict_churn(input: ChurnRequest):
    try:
        logging.info(f"📥 Input: {input}")

        if not 1 <= input.review_score <= 5:
            return JSONResponse(status_code=400, content={"error": "Review score harus antara 1 dan 5."})

        state = input.customer_state.lower()
        region = state_to_region.get(state, "unknown")
        if region == "unknown":
            return JSONResponse(status_code=400, content={"error": f"State '{state}' tidak dikenali."})

        region_encoded = region_map[region]
        logging.info(f"🧭 Region: {region}")

        spending_category = classify_spending(input.payment_value)
        spending_encoded = spending_map[spending_category]
        logging.info(f"💸 Spending: {spending_category}")

        log_payment = np.log1p(input.payment_value)
        log_price = np.log1p(input.price)
        scaled = scaler.transform([[log_payment, log_price]])[0]
        logging.info(f"📊 Scaled: {scaled}")

        features = [
            scaled[0],
            scaled[1],
            input.review_score,
            spending_encoded,
            *region_encoded
        ]
        logging.info(f"🧩 Final features: {features}")

        if len(features) != len(columns):
            return JSONResponse(status_code=500, content={
                "error": f"Jumlah fitur tidak sesuai. Expected {len(columns)}, got {len(features)}"
            })

        proba = model.predict_proba([features])[0][1]
        pred = int(proba >= threshold)
        logging.info(f"📈 Probability: {proba}")

        return JSONResponse(content={
            "prediction": "Churn" if pred else "Not Churn",
            "churn_probability": float(round(proba, 4)),  # convert to Python float
            "region": region,
            "spending_category": spending_category
        })

    except Exception as e:
        traceback_str = traceback.format_exc()
        logging.error("❌ INTERNAL SERVER ERROR", exc_info=True)
        return JSONResponse(status_code=500, content={
            "error": str(e),
            "trace": traceback_str
        })

# === Jalankan dari terminal ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
