import streamlit as st
import numpy as np
import joblib
import pandas as pd

#Set Page Config
st.set_page_config(layout="wide")

# Membuat dua kolom untuk logo dan judul
col1, col2 = st.columns([1, 4])

# Menambahkan logo di kolom pertama (kiri) dan judul di kolom kedua (kanan)
with col1:
    st.image("logo_ashirvada.png", width=100)  # Ganti dengan path logo kamu

with col2:
    st.title("Customer Churn Risk Predictor")

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
    recommendations = recommendation_map[classify_risk(prob)].split("\n")
    return "\n".join([f"{rec}" for rec in recommendations])

def run_prediction(mean_price, total_payment_value, avg_review_score, customer_state):
    region = state_to_region.get(customer_state.lower(), "unknown")
    if region == "unknown":
        st.error("❌ Unrecognized state")
        return

    region_encoded = region_map[region]
    spending_category = classify_spending(total_payment_value)
    log_payment = np.log1p(total_payment_value)
    log_price = np.log1p(mean_price)

    features = [log_payment, log_price, avg_review_score, *region_encoded]

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
    for rec in actions.split("\n"):  # Split recommendations into separate lines
        st.markdown(f"- {rec}")

# === Streamlit App ===

tab1, tab2, tab3 = st.tabs(["🧾 Manual Input", "🔠 Input by Customer ID", "📂 Batch CSV"])

# === Tab 1: Manual Input ===
with tab1:
    st.subheader("🔧 Manual Input Prediction")
    mean_price = st.number_input("🍒 Average Product Price ($)", min_value=0.0)
    total_payment_value = st.number_input("💰 Total Payment Value ($)", min_value=0.0)
    avg_review_score = st.slider("⭐ Average Review Score", 1.0, 5.0, 3.0, 0.1)
    customer_state = st.selectbox("📍 Customer State", list(state_to_region.keys()))
    if st.button("🔍 Predict from Manual Input"):
        run_prediction(mean_price, total_payment_value, avg_review_score, customer_state)

# === Tab 2: Input by Customer Unique ID (with autocomplete) ===
with tab2:
    st.subheader("🔡 Input by Customer Unique ID")
    
    # Ambil data customer dari CSV
    try:
        df_customer_data = pd.read_csv("FINAL_DATASET_WITH_CUSTOMER_ID.csv")
        
        # Ambil semua customer ID untuk autocomplete
        customer_ids = df_customer_data["customer_unique_id"].unique()
        
        # Buat selectbox untuk autocomplete berdasarkan customer ID
        selected_id = st.selectbox("Select Customer ID", customer_ids)

        # Jika ada ID yang dipilih
        if selected_id:
            selected_row = df_customer_data[df_customer_data["customer_unique_id"] == selected_id]

            if not selected_row.empty:
                mean_price = selected_row["mean_price"].values[0]
                total_payment_value = selected_row["total_payment_value"].values[0]
                avg_review_score = selected_row["avg_review_score"].values[0]
                customer_state = selected_row["customer_state"].values[0]

                st.markdown(f"**Mean Price:** {mean_price:.2f}")
                st.markdown(f"**Total Payment Value:** {total_payment_value:.2f}")
                st.markdown(f"**Review Score:** {avg_review_score:.2f}")
                st.markdown(f"**Customer State:** {customer_state}")

                if st.button("🔍 Predict from Selected Customer"):
                    run_prediction(mean_price, total_payment_value, avg_review_score, customer_state)
            else:
                st.warning("❗ ID tidak ditemukan dalam file.")
    except FileNotFoundError:
        st.error("❌ File 'FINAL_DATASET_WITH_CUSTOMER_ID.csv' tidak ditemukan.")

# === Tab 3: Batch Prediction from CSV ===
with tab3:
    st.subheader("📁 Batch Prediction from CSV")
    uploaded_file = st.file_uploader("Upload a CSV with required columns", type=["csv"])

    if uploaded_file:
        try:
            df_csv = pd.read_csv(uploaded_file)
            missing = set(columns) - set(df_csv.columns)
            if missing:
                st.error(f"❌ Missing columns: {missing}")
            else:
                X = df_csv.loc[:, columns]
                df_csv["churn_proba"] = model.predict_proba(X.values)[:, 1].round(2)
                df_csv["churn_risk"] = df_csv["churn_proba"].apply(classify_risk)
                df_csv["recommendations"] = df_csv["churn_risk"].map(recommendation_map)

                st.success("✅ Prediction Complete")
                preview_cols = ['churn_proba', 'churn_risk', 'recommendations']
                if 'customer_unique_id' in df_csv.columns:
                    preview_cols.insert(0, 'customer_unique_id')

                # Display dataframe with larger column width
                st.dataframe(df_csv[preview_cols].round(2).head(100), use_container_width=True)

                # Allow downloading the prediction result as CSV
                csv_out = df_csv.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Result", csv_out, "batch_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"❌ Error reading CSV: {e}")
