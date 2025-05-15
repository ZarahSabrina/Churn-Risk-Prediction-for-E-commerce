import streamlit as st
import numpy as np
import joblib
import pandas as pd

# === Load Model Bundle ===
model_bundle = joblib.load("xgb_churn_model.joblib")
model = model_bundle["model"]
threshold = model_bundle["threshold"]
columns = model_bundle["columns"]

# === Region Mapping ===
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

# === Recommendation Map with Category-specific Emojis ===
recommendation_map = {
    "High Risk": "🚨 Send exclusive bundles and high-value discounts with urgency (e.g., limited-time offers) via email, SMS, or phone.\n🎯 Offer loyalty points, free shipping, or VIP access to prevent drop-offs.",
    "Medium Risk": "💡 Suggest products based on browsing and purchase history.\n🎁 Encourage joining loyalty programs with tiered rewards to increase stickiness.",
    "Low Risk": "🌟 Offer premium-tier products or early-access deals to increase order value.\n🔁 Maintain regular engagement via content, promotions, and fast checkout experience."
}

# === Helper Functions ===
def classify_spending(value):
    if value <= 61:
        return "low"
    elif value <= 189:
        return "medium"
    else:
        return "high"

def classify_risk(prob):
    if prob >= 0.70:
        return "High Risk"
    elif prob >= 0.35:
        return "Medium Risk"
    else:
        return "Low Risk"

def get_recommendations(prob):
    return recommendation_map[classify_risk(prob)].split("\n")

# === Streamlit Layout ===
st.set_page_config(layout="wide")
st.title("💡 Customer Churn Risk Predictor")

col1, col2 = st.columns([1, 2])

# === Column 1: Input Form ===
with col1:
    st.markdown("### 🧾 Input Data")
    mean_price = st.number_input("🍒 Average Product Price", min_value=0.0)
    total_payment_value = st.number_input("💰 Total Payment Value", min_value=0.0)
    avg_review_score = st.slider("⭐ Average Review Score (1.0–5.0)", 1.0, 5.0, 3.0, step=0.1)
    customer_state = st.selectbox("📍 Customer State", list(state_to_region.keys()))
    predict_btn = st.button("🔍 Predict")

# === Column 2: Output ===
with col2:
    if predict_btn:
        region = state_to_region.get(customer_state.lower(), "unknown")
        if region == "unknown":
            st.error("❌ Unrecognized state")
            st.stop()

        region_encoded = region_map[region]
        spending_category = classify_spending(total_payment_value)
        log_payment = np.log1p(total_payment_value)
        log_price = np.log1p(mean_price)

        features = [
            log_payment,
            log_price,
            avg_review_score,
            *region_encoded
        ]

        proba = model.predict_proba([features])[0][1]
        percent = round(proba * 100, 2)
        risk_label = classify_risk(proba)
        actions = get_recommendations(proba)

        st.markdown("### 📊 Prediction Result")
        st.markdown(f"""
        <style>
        .bar-container {{ position: relative; height: 36px; background-color: #ddd; border-radius: 6px; margin-top: 10px; margin-bottom: 6px; }}
        .bar-fill {{ height: 100%; width: {percent}%; background-color: {'#4caf50' if percent < 35 else '#ff9800' if percent < 70 else '#f44336'}; border-radius: 6px; }}
        .bar-label {{ position: absolute; top: 4px; left: 50%; transform: translateX(-50%); font-size: 18px; font-weight: bold; color: black; }}
        </style>
        <div class="bar-container">
          <div class="bar-fill"></div>
          <div class="bar-label">{percent:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.info(f"📌 This customer falls under the category: **{risk_label} Churn**")
        st.markdown(f"**Customer Region:** `{region.capitalize()}`")
        st.markdown(f"**Spending Category:** `{spending_category.capitalize()}`")
        st.markdown("### 💡 Recommended Actions:")
        for rec in actions:
            st.markdown(f"{rec}")

# === Batch Prediction: Upload CSV ===
st.markdown("---")
st.subheader("📂 Batch Prediction from CSV")

uploaded_file = st.file_uploader(
    "Upload CSV with columns (including optional: customer_unique_id, and exactly): total_payment_value, mean_price, avg_review_score, customer_region_East, North, South, West",
    type=["csv"]
)

if uploaded_file:
    try:
        df_csv = pd.read_csv(uploaded_file)

        # Validasi kolom model
        missing = set(columns) - set(df_csv.columns)
        if missing:
            st.error(f"❌ Missing required columns in CSV: {missing}")
        else:
            X = df_csv.loc[:, columns]
            X.columns.name = None
            df_csv["churn_proba"] = model.predict_proba(X.values)[:, 1].round(2)
            df_csv["churn_risk"] = df_csv["churn_proba"].apply(classify_risk)
            df_csv["recommendations"] = df_csv["churn_risk"].map(recommendation_map)

            st.success("✅ Prediction completed.")

            preview_cols = ['churn_proba', 'churn_risk', 'recommendations']
            if 'customer_unique_id' in df_csv.columns:
                preview_cols.insert(0, 'customer_unique_id')

            st.dataframe(df_csv[preview_cols].round(2).head(100)) 

            csv_out = df_csv.to_csv(index=False).encode("utf-8")
            st.download_button("📅 Download Result CSV", csv_out, "churn_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
