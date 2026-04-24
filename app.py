import streamlit as st
import pickle
import pandas as pd

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Financial Intelligence System", layout="wide")

# -------------------------
# LOAD MODEL
# -------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------
# STYLING (CLEAN FINTECH LOOK)
# -------------------------
st.markdown("""
<style>
.main {background-color: #f5f7fb;}
h1, h2, h3 {color: #1f2c56;}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

st.title("💰 Financial Intelligence System")
st.caption("A Hybrid Financial Risk Assessment Engine")

# =========================
# MAPPINGS (UNCHANGED)
# =========================

income_map = {"Below ₹20k":1,"₹20k–40k":2,"₹40k–60k":3,"₹60k–1L":4,"Above ₹1L":5}
emi_map = {"Less than 20%":1,"20%–30%":2,"30%–40%":3,"Above 40%":4}
likert_map = {"Strongly Disagree":1,"Disagree":2,"Neutral":3,"Agree":4,"Strongly Agree":5}
binary_map = {"No":0,"Yes":1}

# =========================
# CORE LOGIC (UNCHANGED)
# =========================

def safe_map(value, mapping):
    return mapping[value.strip()]

def preprocess_input(u):
    return {
        'monthly_income': safe_map(u['income'], income_map),
        'emi_percentage': safe_map(u['emi'], emi_map),
        'fomo': safe_map(u['fomo'], likert_map),
        'social_influence': safe_map(u['social'], likert_map),
        'optimism_bias': safe_map(u['optimism'], likert_map),
        'financial_tracking': safe_map(u['tracking'], likert_map),
        'interest_understanding': safe_map(u['interest'], likert_map),
        'emi_awareness': safe_map(u['emi_awareness'], likert_map),
        'debt_knowledge': safe_map(u['debt'], likert_map),
        'inflation_impact': safe_map(u['inflation'], likert_map),
        'inflation_loan_dependency': safe_map(u['inflation_loan'], likert_map),
        'inflation_lifestyle_borrowing': safe_map(u['lifestyle'], likert_map),
        'has_loan': safe_map(u['loan'], binary_map)
    }

def create_features(f):
    f['fli']=(f['financial_tracking']+f['interest_understanding']+f['emi_awareness']+f['debt_knowledge'])/4
    f['inflation_index']=(f['inflation_impact']+f['inflation_loan_dependency']+f['inflation_lifestyle_borrowing'])/3
    f['behavior_score']=(f['fomo']+f['social_influence']+f['optimism_bias'])/3
    f['stress_proxy']=(f['emi_percentage']+f['inflation_index'])/2
    return f

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
    if fhs>=80:return "Financially Healthy 🟢"
    elif fhs>=60:return "Stable 🟡"
    elif fhs>=40:return "At Risk 🟠"
    else:return "Financially Vulnerable 🔴"

def predict_ml_risk(f,model):
    df=pd.DataFrame([{
        'emi_percentage':f['emi_percentage'],
        'fomo':f['fomo'],
        'social_influence':f['social_influence'],
        'optimism_bias':f['optimism_bias'],
        'fli':f['fli'],
        'inflation_index':f['inflation_index']
    }])
    pred=int(model.predict(df)[0])
    return {1:"Financially Vulnerable 🔴",2:"Somewhat Strained 🟡",3:"Financially Stable 🟢"}[pred]

# =========================
# FVPM (UNCHANGED)
# =========================

model_weights={"emi_percentage":-0.7799,"social_influence":-1.3270,"optimism_bias":0.8895,"fomo":-0.0381}

def calculate_risk_score(f):
    return round(
        model_weights["emi_percentage"]*f["emi_percentage"]+
        model_weights["social_influence"]*f["social_influence"]+
        model_weights["optimism_bias"]*f["optimism_bias"]+
        model_weights["fomo"]*f["fomo"],2)

def classify_model_risk(score):
    if score<=-3:return "Low Stress (Stable) 🟢"
    elif score<=0:return "Moderate Stress 🟡"
    else:return "High Stress 🔴"

def final_risk_label(fhs_cat,ml):
    if "Vulnerable" in ml or "At Risk" in fhs_cat:return "At Risk 🟠"
    elif "Stable" in ml and "Stable" in fhs_cat:return "Stable 🟢"
    else:return "Moderate 🟡"

def insights(f):
    out=[]
    if f['fomo']>=4: out.append(("FOMO behavior","Impulsive risk"))
    if f['social_influence']>=4: out.append(("Social influence","External pressure"))
    if f['optimism_bias']>=4: out.append(("Optimism bias","Future reliance"))
    if f['emi_percentage']>=3: out.append(("High EMI","Cash flow strain"))
    return out

def recommendations(f,fhs):
    rec=[]
    if f['emi_percentage']>=3: rec.append(("Reduce EMI","Lower burden","Better liquidity"))
    if f['fomo']>=4: rec.append(("Control impulses","Avoid FOMO","Better decisions"))
    if fhs<40: rec.append(("Emergency fund","High risk","Stability"))
    if fhs>60: rec.append(("Start investing","Good base","Growth"))
    return rec

# =========================
# UI INPUT
# =========================

st.header("📊 Financial Profile")

loan=st.selectbox("Do you have a loan?",["Yes","No"])
income=st.selectbox("Income",list(income_map.keys()))
emi=st.selectbox("EMI %",list(emi_map.keys()))

st.subheader("Behavior")
likert=list(likert_map.keys())
fomo=st.selectbox("FOMO",likert)
social=st.selectbox("Social Influence",likert)
optimism=st.selectbox("Optimism",likert)

st.subheader("Awareness")
tracking=st.selectbox("Tracking",likert)
interest=st.selectbox("Interest Knowledge",likert)
emi_awareness=st.selectbox("EMI Awareness",likert)
debt=st.selectbox("Debt Knowledge",likert)

st.subheader("Inflation")
inflation=st.selectbox("Inflation Impact",likert)
inflation_loan=st.selectbox("Borrow due to inflation",likert)
lifestyle=st.selectbox("Lifestyle borrowing",likert)

# =========================
# RUN
# =========================

if st.button("Analyze"):

    u={"income":income,"emi":emi,"fomo":fomo,"social":social,"optimism":optimism,
       "tracking":tracking,"interest":interest,"emi_awareness":emi_awareness,
       "debt":debt,"inflation":inflation,"inflation_loan":inflation_loan,
       "lifestyle":lifestyle,"loan":loan}

    f=create_features(preprocess_input(u))

    fhs=calculate_fhs(f)
    fhs_cat=classify_risk(fhs)
    ml=predict_ml_risk(f,model)
    fvpm_score=calculate_risk_score(f)
    fvpm=classify_model_risk(fvpm_score)
    final=final_risk_label(fhs_cat,ml)

    st.markdown("### 📊 Results")
    c1,c2,c3=st.columns(3)
    c1.metric("FHS",fhs)
    c2.metric("Final Risk",final)
    c3.metric("FVPM",fvpm)

    st.write("🤖 ML:",ml)

    st.markdown("### 🧠 Insights")
    ins=insights(f)
    if not ins: st.write("✔ Balanced behavior")
    else:
        for i in ins:
            st.write(f"**{i[0]}** → {i[1]}")

    st.markdown("### 📌 Recommendations")
    rec=recommendations(f,fhs)
    if not rec: st.write("✔ Maintain consistency")
    else:
        for r in rec:
            st.write(f"**{r[0]}** | {r[1]} | {r[2]}")
