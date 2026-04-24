import streamlit as st
import pickle
import pandas as pd

# -------------------------
# LOAD MODEL
# -------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Financial Intelligence System", layout="centered")

st.title("💰 Financial Intelligence System")
st.caption("A structured system to evaluate financial health, behavioral patterns, and risk.")

# =========================
# MAPPINGS
# =========================

income_map = {"Below ₹20k":1,"₹20k–40k":2,"₹40k–60k":3,"₹60k–1L":4,"Above ₹1L":5}
emi_map = {"Less than 20%":1,"20%–30%":2,"30%–40%":3,"Above 40%":4}
likert_map = {"Strongly Disagree":1,"Disagree":2,"Neutral":3,"Agree":4,"Strongly Agree":5}
binary_map = {"No":0,"Yes":1}

# =========================
# CORE PIPELINE (UNCHANGED)
# =========================

def preprocess_input(user_input):
    return {
        'monthly_income': income_map[user_input['income']],
        'emi_percentage': emi_map[user_input['emi']],
        'fomo': likert_map[user_input['fomo']],
        'social_influence': likert_map[user_input['social_influence']],
        'optimism_bias': likert_map[user_input['optimism']],
        'financial_tracking': likert_map[user_input['tracking']],
        'interest_understanding': likert_map[user_input['interest']],
        'emi_awareness': likert_map[user_input['emi_awareness']],
        'debt_knowledge': likert_map[user_input['debt_knowledge']],
        'inflation_impact': likert_map[user_input['inflation_impact']],
        'inflation_loan_dependency': likert_map[user_input['inflation_loan']],
        'inflation_lifestyle_borrowing': likert_map[user_input['lifestyle_borrowing']],
        'has_loan': binary_map[user_input['has_loan']]
    }

def create_features(features):
    features['fli'] = (features['financial_tracking'] + features['interest_understanding'] + features['emi_awareness'] + features['debt_knowledge']) / 4
    features['inflation_index'] = (features['inflation_impact'] + features['inflation_loan_dependency'] + features['inflation_lifestyle_borrowing']) / 3
    features['behavior_score'] = (features['fomo'] + features['social_influence'] + features['optimism_bias']) / 3
    features['stress_proxy'] = (features['emi_percentage'] + features['inflation_index']) / 2
    return features

def normalize(x): return (x-1)/4

def calculate_fhs(f):
    return round((
        0.25*(1-normalize(f['emi_percentage']))+
        0.20*(1-normalize(f['stress_proxy']))+
        0.20*(1-normalize(f['inflation_index']))+
        0.20*(1-normalize(f['behavior_score']))+
        0.15*(normalize(f['fli']))
    )*100,2)

def classify_risk(fhs):
    if fhs>=80:return "Financially Healthy"
    elif fhs>=60:return "Stable"
    elif fhs>=40:return "At Risk"
    else:return "Financially Vulnerable"

def predict_ml_risk(features, model):
    df = pd.DataFrame([{
        'emi_percentage': features['emi_percentage'],
        'fomo': features['fomo'],
        'social_influence': features['social_influence'],
        'optimism_bias': features['optimism_bias'],
        'fli': features['fli'],
        'inflation_index': features['inflation_index']
    }])
    return {1:"Financially Vulnerable",2:"Somewhat Strained",3:"Financially Stable"}[int(model.predict(df)[0])]

# =========================
# FVPM
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
        model_weights["fomo"] * features["fomo"], 2)

def classify_model_risk(score):
    if score <= -3:return "Low Stress"
    elif score <= 0:return "Moderate Stress"
    else:return "High Stress"

def final_risk_label(fhs_category, ml_risk):
    if "Vulnerable" in ml_risk or "At Risk" in fhs_category:return "At Risk"
    elif "Stable" in ml_risk and "Stable" in fhs_category:return "Stable"
    else:return "Moderate"

# =========================
# YOUR EXACT INSIGHTS FUNCTION
# =========================

def generate_behavioral_insights(features):

    insights = []

    if features['fomo'] >= 4:
        insights.append({
            "insight": "You show a tendency to fear missing out on opportunities.",
            "interpretation": "This may lead to impulsive financial decisions or unnecessary borrowing."
        })

    if features['social_influence'] >= 4:
        insights.append({
            "insight": "Your financial decisions are influenced by people around you.",
            "interpretation": "This may cause you to take financial actions that may not align with your personal capacity."
        })

    if features['optimism_bias'] >= 4:
        insights.append({
            "insight": "You rely on expectations of future income growth.",
            "interpretation": "This can lead to over-commitment and increased financial risk if expectations are not met."
        })

    if features['emi_percentage'] >= 3:
        insights.append({
            "insight": "Your EMI burden is relatively high.",
            "interpretation": "This reduces your financial flexibility and limits your ability to save."
        })

    if features['fli'] <= 2.5:
        insights.append({
            "insight": "Your financial awareness appears to be limited.",
            "interpretation": "Improving financial knowledge can help you make better long-term decisions."
        })

    if features['inflation_index'] >= 4:
        insights.append({
            "insight": "You are experiencing strong financial pressure due to inflation.",
            "interpretation": "This may increase reliance on borrowing and affect financial stability."
        })

    return insights

# =========================
# YOUR EXACT RECOMMENDATION FUNCTION
# =========================

def generate_recommendations(features, fhs_score, risk_category):

    recommendations = []

    if features['emi_percentage'] >= 3:
        recommendations.append({
            "action": "Reduce your EMI burden below 30% of your income.",
            "why": "Your current EMI level is limiting your ability to save and increasing financial pressure.",
            "impact": "Lower EMI will improve your financial flexibility and reduce long-term stress."
        })

    if features['fomo'] >= 4:
        recommendations.append({
            "action": "Avoid making financial decisions driven by fear of missing out.",
            "why": "Your responses indicate a tendency toward impulsive or opportunity-driven spending.",
            "impact": "This will help you avoid unnecessary financial commitments and improve stability."
        })

    if features['optimism_bias'] >= 4:
        recommendations.append({
            "action": "Avoid relying heavily on expected future income while making financial commitments.",
            "why": "You show a tendency to assume future income growth, which may not always materialize.",
            "impact": "This reduces the risk of over-commitment and unexpected financial stress."
        })

    if features['social_influence'] >= 4:
        recommendations.append({
            "action": "Make financial decisions based on your own financial capacity rather than peer behavior.",
            "why": "Your decisions appear influenced by external social factors.",
            "impact": "This ensures your financial decisions remain sustainable and aligned with your situation."
        })

    if features['fli'] <= 2.5:
        recommendations.append({
            "action": "Improve your financial knowledge regarding loans, EMIs, and savings.",
            "why": "Your responses suggest limited awareness of financial concepts.",
            "impact": "Better understanding will lead to more informed and effective financial decisions."
        })

    if features['inflation_index'] >= 4:
        recommendations.append({
            "action": "Focus on controlling discretionary spending during periods of high inflation.",
            "why": "You are experiencing strong financial pressure due to rising costs.",
            "impact": "This will help maintain financial stability and reduce reliance on borrowing."
        })

    if fhs_score < 40:
        recommendations.append({
            "action": "Prioritize reducing financial risk and building an emergency fund.",
            "why": "Your overall financial health score indicates high vulnerability.",
            "impact": "This will improve your resilience against financial shocks."
        })

    elif fhs_score >= 60:
        recommendations.append({
            "action": "Consider increasing investments for long-term financial growth.",
            "why": "Your financial profile shows relative stability.",
            "impact": "This will help build wealth and improve financial security."
        })

    return recommendations

# =========================
# UI
# =========================
# =========================
# UI LAYER (FINAL VERSION)
# =========================

import matplotlib.pyplot as plt

# ---------- STYLING ----------
st.markdown("""
<style>
body {background-color: #f5f7fa;}
.section {
    background-color: white;
    padding: 18px;
    border-radius: 10px;
    margin-bottom: 15px;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
}
.title {color: #1f3c88; font-weight: 600;}
.subtitle {color: #555; font-size: 14px;}
</style>
""", unsafe_allow_html=True)

likert = list(likert_map.keys())

# ---------- INTRO ----------
st.markdown("""
## Financial Intelligence Assessment

This system evaluates your **financial health, behavioral tendencies, and risk exposure** using:

- Structured financial scoring
- Machine learning prediction
- Behavioral insights
- Regression-based risk modeling

Please answer honestly based on your real financial behavior.
""")

# ---------- SECTION 1 ----------
st.markdown("### Financial Background")

loan = st.selectbox(
"Do you currently have an active loan?",
["Yes","No"],
help="Loans increase fixed obligations and affect financial flexibility."
)

income = st.selectbox(
"Which option best describes your monthly income?",
list(income_map.keys()),
help="Income determines your financial capacity and stability."
)

if loan == "Yes":
    emi = st.selectbox(
    "What percentage of your income goes towards EMI payments?",
    list(emi_map.keys()),
    help="Higher EMI increases financial pressure and reduces savings ability."
    )
else:
    emi = "Less than 20%"

# ---------- SECTION 2 ----------
st.markdown("### Behavioral Assessment")

fomo = st.selectbox(
"I feel pressure to act on financial opportunities so that I do not miss out.",
likert
)

social = st.selectbox(
"My financial decisions are influenced by people around me.",
likert
)

optimism = st.selectbox(
"I expect my income to increase significantly in the future.",
likert
)

# ---------- SECTION 3 ----------
st.markdown("### Financial Awareness")

tracking = st.selectbox("I actively track my financial activities.", likert)
interest = st.selectbox("I understand how interest rates affect finances.", likert)
emi_awareness = st.selectbox("I understand the impact of EMI commitments.", likert)
debt = st.selectbox("I understand my overall debt obligations.", likert)

# ---------- SECTION 4 ----------
st.markdown("### Inflation & Spending")

inflation = st.selectbox("Inflation has significantly affected my finances.", likert)
inflation_loan = st.selectbox("I borrow more due to rising expenses.", likert)
lifestyle = st.selectbox("I borrow to maintain my lifestyle.", likert)

# ---------- RUN ----------
st.markdown("---")

if st.button("Analyze Financial Profile"):

    # ⚠️ DO NOT MODIFY (YOUR ORIGINAL LOGIC)
    u={"income":income,"emi":emi,"fomo":fomo,"social_influence":social,"optimism":optimism,
       "tracking":tracking,"interest":interest,"emi_awareness":emi_awareness,
       "debt_knowledge":debt,"inflation_impact":inflation,"inflation_loan":inflation_loan,
       "lifestyle_borrowing":lifestyle,"has_loan":loan}

    f=create_features(preprocess_input(u))

    fhs=calculate_fhs(f)
    fhs_cat=classify_risk(fhs)
    ml=predict_ml_risk(f,model)
    fvpm=classify_model_risk(calculate_risk_score(f))
    final=final_risk_label(fhs_cat,ml)

    insights = generate_behavioral_insights(f)
    recs = generate_recommendations(f, fhs, final)

    # ---------- RESULTS ----------
    st.markdown("## Financial Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="section">
        <div class="title">Financial Health Score</div>
        <div class="subtitle">{fhs} ({fhs_cat})</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="section">
        <div class="title">Overall Risk Level</div>
        <div class="subtitle">{final}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="section">
    <div class="title">Behavioral Stress (Regression Model)</div>
    <div class="subtitle">{fvpm}</div>
    </div>
    """, unsafe_allow_html=True)

    # ---------- RISK METER ----------
    st.markdown("### Financial Health Visualization")

    fig, ax = plt.subplots()
    ax.barh(["Score"], [fhs])
    ax.set_xlim(0, 100)
    ax.set_title("Financial Health Score (0–100)")
    st.pyplot(fig)

    # ---------- PERSONA ----------
    st.markdown("### Financial Behavior Profile")

    persona = ""
    if fhs >= 75:
        persona = "Disciplined Planner"
    elif fhs >= 60:
        persona = "Stable but Reactive"
    elif fhs >= 40:
        persona = "Vulnerable Decision Maker"
    else:
        persona = "High Risk Financial Behavior"

    if f['fomo'] >= 4:
        persona += " with Impulsive Tendencies"
    if f['social_influence'] >= 4:
        persona += " influenced by External Factors"
    if f['optimism_bias'] >= 4:
        persona += " relying on Future Expectations"

    st.markdown(f"""
    <div class="section">
    <b>{persona}</b><br>
    <span class="subtitle">
    This profile summarizes your financial behavior pattern.
    </span>
    </div>
    """, unsafe_allow_html=True)

    # ---------- INSIGHTS ----------
    st.markdown("### Behavioral Insights")

    for i in insights:
        st.markdown(f"""
        <div class="section">
        <b>{i['insight']}</b><br>
        <span class="subtitle">{i['interpretation']}</span>
        </div>
        """, unsafe_allow_html=True)

    # ---------- RECOMMENDATIONS ----------
    st.markdown("### Recommended Actions")

    for r in recs:
        st.markdown(f"""
        <div class="section">
        <b>{r['action']}</b><br>
        <span class="subtitle"><b>Why:</b> {r['why']}</span><br>
        <span class="subtitle"><b>Impact:</b> {r['impact']}</span>
        </div>
        """, unsafe_allow_html=True)

    # ---------- LEARNING ----------
    st.markdown("### Learning & Improvement Resources")

    links = []

    if f['emi_percentage'] >= 3:
        links.append(("Debt Management", "https://www.rbi.org.in/financialeducation"))

    if f['fli'] <= 2.5:
        links.append(("Financial Basics", "https://www.investopedia.com/financial-term-dictionary-4769738"))

    if f['fomo'] >= 4:
        links.append(("Behavioral Finance", "https://www.investopedia.com/terms/b/behavioralfinance.asp"))

    if f['inflation_index'] >= 4:
        links.append(("Understanding Inflation", "https://www.investopedia.com/terms/i/inflation.asp"))

    for title, link in links:
        st.markdown(f"- [{title}]({link})")

    st.markdown("""
Additional trusted resources:
- https://www.rbi.org.in  
- https://www.sebi.gov.in  
- https://www.investopedia.com  
""")
