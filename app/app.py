import streamlit as st
import numpy as np
import joblib

# Load model
model_bundle = joblib.load("xgb_churn_model.joblib")
model = model_bundle["model"]
scaler = model_bundle["scaler"]
threshold = model_bundle["threshold"]
columns = model_bundle["columns"]

# Mapping region & spending
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

# ======================
# Streamlit App
# ======================

st.title("💡 Customer Churn Predictor")

price = st.number_input("Price", min_value=0.0)
payment_value = st.number_input("Payment Value", min_value=0.0)
review_score = st.slider("Review Score (1–5)", 1, 5, 3)
customer_state = st.selectbox("Customer State", list(state_to_region.keys()))

if st.button("Predict"):
    region = state_to_region.get(customer_state.lower(), "unknown")
    if region == "unknown":
        st.error("❌ State tidak dikenali")
    else:
        region_encoded = region_map[region]
        spending_category = classify_spending(payment_value)
        spending_encoded = spending_map[spending_category]

        # Log & Scale
        log_payment = np.log1p(payment_value)
        log_price = np.log1p(price)
        scaled = scaler.transform([[log_payment, log_price]])[0]

        features = [
            scaled[0],
            scaled[1],
            review_score,
            spending_encoded,
            *region_encoded
        ]

        # Predict
        proba = model.predict_proba([features])[0][1]
        pred = int(proba >= threshold)

        st.subheader("📊 Prediction Result")
        st.write(f"**Prediction:** {'Churn' if pred else 'Not Churn'}")
        st.write(f"**Probability:** {round(proba, 4)}")
        st.write(f"**Region:** {region.capitalize()}")
        st.write(f"**Spending Category:** {spending_category.capitalize()}")
