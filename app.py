import streamlit as st
import pickle
import pandas as pd

# =========================
# LOAD MODEL (YOUR RF MODEL)
# =========================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Financial Intelligence System", layout="centered")

st.title("💰 Financial Intelligence System")
st.caption("Evaluate financial health, behavioral risk, and decision patterns")

# =========================
# ===== YOUR ORIGINAL CODE (UNCHANGED) =====
# =========================

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

likert_map = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly Agree": 5
}

binary_map = {"No": 0, "Yes": 1}

def safe_map(value, mapping, field_name):
    value = value.strip()
    if value not in mapping:
        raise ValueError(f"Invalid input for {field_name}: {value}")
    return mapping[value]

def preprocess_input(user_input):
    processed = {}

    processed['monthly_income'] = safe_map(user_input['income'], income_map, "income")
    processed['emi_percentage'] = safe_map(user_input['emi'], emi_map, "emi")

    processed['fomo'] = safe_map(user_input['fomo'], likert_map, "fomo")
    processed['social_influence'] = safe_map(user_input['social_influence'], likert_map, "social")
    processed['optimism_bias'] = safe_map(user_input['optimism'], likert_map, "optimism")

    processed['financial_tracking'] = safe_map(user_input['tracking'], likert_map, "tracking")
    processed['interest_understanding'] = safe_map(user_input['interest'], likert_map, "interest")
    processed['emi_awareness'] = safe_map(user_input['emi_awareness'], likert_map, "emi awareness")
    processed['debt_knowledge'] = safe_map(user_input['debt_knowledge'], likert_map, "debt")

    processed['inflation_impact'] = safe_map(user_input['inflation_impact'], likert_map, "inflation")
    processed['inflation_loan_dependency'] = safe_map(user_input['inflation_loan'], likert_map, "inflation loan")
    processed['inflation_lifestyle_borrowing'] = safe_map(user_input['lifestyle_borrowing'], likert_map, "lifestyle")

    processed['has_loan'] = safe_map(user_input['has_loan'], binary_map, "loan")

    return processed


def create_features(features):
    features['fli'] = (
        features['financial_tracking'] +
        features['interest_understanding'] +
        features['emi_awareness'] +
        features['debt_knowledge']
    ) / 4

    features['inflation_index'] = (
        features['inflation_impact'] +
        features['inflation_loan_dependency'] +
        features['inflation_lifestyle_borrowing']
    ) / 3

    features['behavior_score'] = (
        features['fomo'] +
        features['social_influence'] +
        features['optimism_bias']
    ) / 3

    features['stress_proxy'] = (
        features['emi_percentage'] +
        features['inflation_index']
    ) / 2

    return features


def normalize(value):
    return (value - 1) / 4


def calculate_fhs(features):

    emi_score = 1 - normalize(features['emi_percentage'])
    stress_score = 1 - normalize(features['stress_proxy'])
    inflation_score = 1 - normalize(features['inflation_index'])
    behavior_score = 1 - normalize(features['behavior_score'])
    fli_score = normalize(features['fli'])

    fhs = (
        0.25 * emi_score +
        0.20 * stress_score +
        0.20 * inflation_score +
        0.20 * behavior_score +
        0.15 * fli_score
    ) * 100

    return round(fhs, 2)


def classify_risk(fhs):
    if fhs >= 80:
        return "Financially Healthy"
    elif fhs >= 60:
        return "Stable"
    elif fhs >= 40:
        return "At Risk"
    else:
        return "Financially Vulnerable"


def predict_ml_risk(features, model):

    input_df = pd.DataFrame([{
        'emi_percentage': features['emi_percentage'],
        'fomo': features['fomo'],
        'social_influence': features['social_influence'],
        'optimism_bias': features['optimism_bias'],
        'fli': features['fli'],
        'inflation_index': features['inflation_index']
    }])

    prediction = int(model.predict(input_df)[0])

    label_map = {
        1: "Financially Vulnerable",
        2: "Somewhat Strained",
        3: "Financially Stable"
    }

    return label_map[prediction]


# =========================
# FVPM (UNCHANGED)
# =========================

model_weights = {
    "emi_percentage": -0.7799,
    "social_influence": -1.3270,
    "optimism_bias": 0.8895,
    "fomo": -0.0381
}

def calculate_risk_score(features):
    return round(
        model_weights["emi_percentage"] * features["emi_percentage"] +
        model_weights["social_influence"] * features["social_influence"] +
        model_weights["optimism_bias"] * features["optimism_bias"] +
        model_weights["fomo"] * features["fomo"],
        2
    )

def classify_model_risk(score):
    if score <= -3:
        return "Low Stress"
    elif score <= 0:
        return "Moderate Stress"
    else:
        return "High Stress"


def final_risk_label(fhs_category, ml_risk):
    if "Vulnerable" in ml_risk or "At Risk" in fhs_category:
        return "At Risk"
    elif "Stable" in ml_risk and "Stable" in fhs_category:
        return "Stable"
    else:
        return "Moderate"


# =========================
# UI (ONLY CHANGE)
# =========================

likert = list(likert_map.keys())

st.header("Basic Information")

loan = st.selectbox("Do you have an active loan?", ["Yes", "No"])

income = st.selectbox("Monthly Income", list(income_map.keys()))

if loan == "Yes":
    emi = st.selectbox("EMI % of income", list(emi_map.keys()))
else:
    emi = "Less than 20%"

st.header("Behavior")

fomo = st.selectbox("Fear of missing financial opportunities", likert)
social = st.selectbox("Influence of others on decisions", likert)
optimism = st.selectbox("Expectation of future income growth", likert)

st.header("Awareness")

tracking = st.selectbox("Tracking finances", likert)
interest = st.selectbox("Understanding interest", likert)
emi_awareness = st.selectbox("Understanding EMI", likert)
debt = st.selectbox("Understanding debt", likert)

st.header("Inflation")

inflation = st.selectbox("Inflation impact", likert)
inflation_loan = st.selectbox("Borrowing due to inflation", likert)
lifestyle = st.selectbox("Lifestyle borrowing", likert)

# =========================
# RUN
# =========================

if st.button("Analyze"):

    user_input = {
        "income": income,
        "emi": emi,
        "fomo": fomo,
        "social_influence": social,
        "optimism": optimism,
        "tracking": tracking,
        "interest": interest,
        "emi_awareness": emi_awareness,
        "debt_knowledge": debt,
        "inflation_impact": inflation,
        "inflation_loan": inflation_loan,
        "lifestyle_borrowing": lifestyle,
        "has_loan": loan
    }

    processed = preprocess_input(user_input)
    features = create_features(processed)

    fhs = calculate_fhs(features)
    fhs_cat = classify_risk(fhs)

    ml_risk = predict_ml_risk(features, model)

    fvpm_score = calculate_risk_score(features)
    fvpm_risk = classify_model_risk(fvpm_score)

    final = final_risk_label(fhs_cat, ml_risk)

    st.header("Results")

    st.write(f"Financial Health Score: {fhs} ({fhs_cat})")
    st.write(f"Final Risk: {final}")
    st.write(f"Regression Risk (FVPM): {fvpm_risk}")
