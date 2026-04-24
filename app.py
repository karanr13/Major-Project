import streamlit as st
import pickle
import pandas as pd

# -------------------------
# LOAD MODEL
# -------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------
# MAPPINGS (SAME AS YOUR CODE)
# -------------------------
likert_map = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly Agree": 5
}

income_map = {
    "Below ₹20k": 1,
    "₹20k–40k": 2,
    "₹40k–60k": 3,
    "₹60k–1L": 4,
    "Above ₹1L": 5
}

emi_map = {
    "Less than 20%": 1,
    "20%–30%": 2,
    "30%–40%": 3,
    "Above 40%": 4
}

# -------------------------
# FEATURE ENGINEERING
# -------------------------
def create_features(data):
    fli = (data['tracking'] + data['interest'] + data['emi_awareness'] + data['debt']) / 4
    inflation = (data['inflation'] + data['inflation_loan'] + data['lifestyle']) / 3
    behavior = (data['fomo'] + data['social'] + data['optimism']) / 3
    stress = (data['emi'] + inflation) / 2

    return fli, inflation, behavior, stress

# -------------------------
# FHS SCORE
# -------------------------
def calculate_fhs(fli, inflation, behavior, stress, emi):
    def norm(x): return (x-1)/4

    score = (
        0.25*(1-norm(emi)) +
        0.20*(1-norm(stress)) +
        0.20*(1-norm(inflation)) +
        0.20*(1-norm(behavior)) +
        0.15*(norm(fli))
    ) * 100

    return round(score,2)

# -------------------------
# ML PREDICTION
# -------------------------
def predict_ml(emi, fomo, social, optimism, fli, inflation):
    df = pd.DataFrame([{
        'emi_percentage': emi,
        'fomo': fomo,
        'social_influence': social,
        'optimism_bias': optimism,
        'fli': fli,
        'inflation_index': inflation
    }])

    pred = int(model.predict(df)[0])

    return {
        1: "Financially Vulnerable 🔴",
        2: "Somewhat Strained 🟡",
        3: "Financially Stable 🟢"
    }[pred]

# -------------------------
# FINAL RISK (HYBRID)
# -------------------------
def final_risk(fhs, ml):
    if fhs < 40 or "Vulnerable" in ml:
        return "At Risk 🟠"
    elif fhs > 60 and "Stable" in ml:
        return "Stable 🟢"
    else:
        return "Moderate 🟡"

# -------------------------
# UI
# -------------------------
st.title("💰 Financial Intelligence System")

likert = list(likert_map.keys())

income = st.selectbox("Income", list(income_map.keys()))
emi = st.selectbox("EMI %", list(emi_map.keys()))

fomo = st.selectbox("FOMO", likert)
social = st.selectbox("Social Influence", likert)
optimism = st.selectbox("Optimism Bias", likert)

tracking = st.selectbox("Track finances", likert)
interest = st.selectbox("Understand interest", likert)
emi_awareness = st.selectbox("Understand EMI", likert)
debt = st.selectbox("Understand debt", likert)

inflation = st.selectbox("Inflation impact", likert)
inflation_loan = st.selectbox("Borrow due to inflation", likert)
lifestyle = st.selectbox("Lifestyle borrowing", likert)

# -------------------------
# RUN
# -------------------------
if st.button("Analyze"):

    data = {
        "emi": emi_map[emi],
        "fomo": likert_map[fomo],
        "social": likert_map[social],
        "optimism": likert_map[optimism],
        "tracking": likert_map[tracking],
        "interest": likert_map[interest],
        "emi_awareness": likert_map[emi_awareness],
        "debt": likert_map[debt],
        "inflation": likert_map[inflation],
        "inflation_loan": likert_map[inflation_loan],
        "lifestyle": likert_map[lifestyle]
    }

    fli, inflation_idx, behavior, stress = create_features(data)

    fhs = calculate_fhs(fli, inflation_idx, behavior, stress, data['emi'])

    ml = predict_ml(data['emi'], data['fomo'], data['social'],
                    data['optimism'], fli, inflation_idx)

    final = final_risk(fhs, ml)

    st.success("Done!")

    st.metric("FHS Score", fhs)
    st.metric("Final Risk", final)
    st.write("ML Insight:", ml)