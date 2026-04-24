import streamlit as st
import pickle
import pandas as pd

# =========================
# LOAD MODEL (YOUR RF MODEL)
# =========================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# =========================
# ===== COPY YOUR CODE BELOW (UNCHANGED) =====
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
        return "Financially Healthy 🟢"
    elif fhs >= 60:
        return "Stable 🟡"
    elif fhs >= 40:
        return "At Risk 🟠"
    else:
        return "Financially Vulnerable 🔴"


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
        1: "Financially Vulnerable 🔴",
        2: "Somewhat Strained 🟡",
        3: "Financially Stable 🟢"
    }

    return label_map[prediction], prediction


def final_risk_label(fhs_category, ml_risk):

    if "Vulnerable" in ml_risk or "At Risk" in fhs_category:
        return "At Risk 🟠"

    elif "Stable" in ml_risk and "Stable" in fhs_category:
        return "Stable 🟢"

    else:
        return "Moderate 🟡"


# ===== INSIGHTS =====
def generate_behavioral_insights(features):
    insights = []

    if features['fomo'] >= 4:
        insights.append({"insight":"FOMO detected","interpretation":"May lead to impulsive decisions"})

    if features['social_influence'] >= 4:
        insights.append({"insight":"Social influence high","interpretation":"Decisions may not match your capacity"})

    if features['optimism_bias'] >= 4:
        insights.append({"insight":"Optimism bias high","interpretation":"Overestimation of future income risk"})

    if features['emi_percentage'] >= 3:
        insights.append({"insight":"High EMI burden","interpretation":"Reduces financial flexibility"})

    return insights


# ===== RECOMMENDATIONS =====
def generate_recommendations(features, fhs, risk):
    rec = []

    if features['emi_percentage'] >= 3:
        rec.append({"action":"Reduce EMI","why":"High EMI","impact":"Improves flexibility"})

    if features['fomo'] >= 4:
        rec.append({"action":"Avoid FOMO decisions","why":"Impulse","impact":"Better stability"})

    if fhs < 40:
        rec.append({"action":"Build emergency fund","why":"High risk","impact":"Improves resilience"})

    return rec


def financial_health_analysis_final(user_input, model):

    processed = preprocess_input(user_input)
    features = create_features(processed)

    fhs = calculate_fhs(features)
    fhs_cat = classify_risk(fhs)

    ml_risk, _ = predict_ml_risk(features, model)

    insights = generate_behavioral_insights(features)
    rec = generate_recommendations(features, fhs, fhs_cat)

    final = final_risk_label(fhs_cat, ml_risk)

    return {
        "FHS Score": fhs,
        "FHS Category": fhs_cat,
        "ML Risk": ml_risk,
        "Final Risk": final,
        "Insights": insights,
        "Recommendations": rec
    }

# =========================
# UI (ONLY PART WE ADDED)
# =========================

st.title("💰 Financial Intelligence System")

likert = list(likert_map.keys())

user_input = {
    "income": st.selectbox("Income", list(income_map.keys())),
    "emi": st.selectbox("EMI %", list(emi_map.keys())),
    "fomo": st.selectbox("FOMO", likert),
    "social_influence": st.selectbox("Social Influence", likert),
    "optimism": st.selectbox("Optimism Bias", likert),
    "tracking": st.selectbox("Track finances", likert),
    "interest": st.selectbox("Understand interest", likert),
    "emi_awareness": st.selectbox("Understand EMI", likert),
    "debt_knowledge": st.selectbox("Understand debt", likert),
    "inflation_impact": st.selectbox("Inflation impact", likert),
    "inflation_loan": st.selectbox("Borrow due to inflation", likert),
    "lifestyle_borrowing": st.selectbox("Lifestyle borrowing", likert),
    "has_loan": st.selectbox("Loan?", ["Yes","No"])
}

if st.button("Analyze"):
    result = financial_health_analysis_final(user_input, model)

    st.metric("FHS Score", result["FHS Score"])
    st.metric("Final Risk", result["Final Risk"])
    st.write("ML:", result["ML Risk"])

    st.subheader("Insights")
    for i in result["Insights"]:
        st.write(i["insight"], "-", i["interpretation"])

    st.subheader("Recommendations")
    for r in result["Recommendations"]:
        st.write(r["action"], "|", r["why"], "|", r["impact"])
