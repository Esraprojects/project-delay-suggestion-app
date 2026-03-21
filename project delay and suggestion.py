# ===============================
# IMPORT LIBRARIES
# ===============================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Project Delay AI",
    page_icon="🏗️",
    layout="wide"
)

# ===============================
# CUSTOM CSS (PREMIUM DESIGN)
# ===============================
st.markdown("""
<style>
body {
    background-color: #0e1117;
}

.main-title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    color: #ffffff;
}

.sub-title {
    text-align: center;
    color: #bbbbbb;
}

.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.5);
}

.metric-box {
    background: linear-gradient(135deg, #1f4037, #99f2c8);
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    color: black;
    font-weight: bold;
}

.footer {
    text-align:center;
    margin-top:40px;
    color:gray;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
st.markdown("<div class='main-title'>🏗️ Project Delay AI System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Smart Prediction • Risk Analysis • Decision Support</div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#00ffcc;'>Made by Esra</div>", unsafe_allow_html=True)

st.divider()

# ===============================
# SAMPLE DATA + TEMPLATE
# ===============================
st.markdown("## 📥 Download Sample Template")

sample_data = pd.DataFrame({
    "Project_Size":["Small","Medium","Large"],
    "Budget":[120000,300000,600000],
    "Team_Size":[6,12,20],
    "Planned_Duration":[30,70,120],
    "Weather":["Good","Bad","Good"],
    "Material_Availability":["Yes","No","Yes"],
    "Manager_Experience":[5,4,8],
    "Contractor_Experience":[7,10,15],
    "Labor_Availability":["High","Medium","Low"],
    "Equipment_Availability":["Yes","No","Yes"],
    "Site_Location":["Urban","Rural","Urban"],
    "Permit_Approval":["Yes","No","Yes"],
    "Inflation_Rate":[12,18,10],
    "Supply_Delay":["No","Yes","No"]
})

csv_template = sample_data.to_csv(index=False).encode('utf-8')

st.download_button("⬇️ Download Template", csv_template, "template.csv")
st.dataframe(sample_data)

st.divider()

# ===============================
# MODEL TRAINING
# ===============================
@st.cache_data
def train_model():
    np.random.seed(42)
    data = []

    for _ in range(1200):
        size = np.random.choice(['Small','Medium','Large'])
        budget = np.random.randint(80000,700000)
        team = np.random.randint(3,30)
        duration = np.random.randint(20,180)
        weather = np.random.choice(['Good','Bad'])
        material = np.random.choice(['Yes','No'])
        manager_exp = np.random.randint(1,15)
        contractor_exp = np.random.randint(1,20)
        labor = np.random.choice(['High','Medium','Low'])
        equipment = np.random.choice(['Yes','No'])
        location = np.random.choice(['Urban','Rural'])
        permit = np.random.choice(['Yes','No'])
        inflation = np.random.uniform(5,30)
        supply = np.random.choice(['Yes','No'])

        delay = 0
        if weather == 'Bad': delay += 10
        if material == 'No': delay += 15
        if labor == 'Low': delay += 10
        if equipment == 'No': delay += 8
        if permit == 'No': delay += 12
        if supply == 'Yes': delay += 10
        if location == 'Rural': delay += 5
        if manager_exp < 5: delay += 7

        delay += int(inflation / 5)
        delay += np.random.randint(0,5)

        data.append([
            size,budget,team,duration,weather,material,
            manager_exp,contractor_exp,labor,equipment,
            location,permit,inflation,supply,delay
        ])

    df = pd.DataFrame(data, columns=[
        'Project_Size','Budget','Team_Size','Planned_Duration','Weather',
        'Material_Availability','Manager_Experience','Contractor_Experience',
        'Labor_Availability','Equipment_Availability','Site_Location',
        'Permit_Approval','Inflation_Rate','Supply_Delay','Delay_Days'
    ])

    df['Status'] = df['Delay_Days'].apply(lambda x: 1 if x > 10 else 0)

    mapping = {
        "Small":0,"Medium":1,"Large":2,
        "Bad":0,"Good":1,
        "No":0,"Yes":1,
        "Low":0,"Medium":1,"High":2,
        "Rural":0,"Urban":1
    }

    df = df.replace(mapping)

    X = df.drop(['Delay_Days','Status'], axis=1)
    y_class = df['Status']
    y_reg = df['Delay_Days']

    clf = RandomForestClassifier().fit(X, y_class)
    reg = RandomForestRegressor().fit(X, y_reg)

    return clf, reg

clf, reg = train_model()

# ===============================
# FILE UPLOAD
# ===============================
st.markdown("## 📂 Upload Excel File")

file = st.file_uploader("Upload your project data (.xlsx)", type=["xlsx"])

if file:
    df = pd.read_excel(file)

    st.success("File uploaded successfully")

    st.dataframe(df)

    mapping = {
        "Small":0,"Medium":1,"Large":2,
        "Bad":0,"Good":1,
        "No":0,"Yes":1,
        "Low":0,"Medium":1,"High":2,
        "Rural":0,"Urban":1
    }

    df = df.replace(mapping)

    df['Delay_Status'] = clf.predict(df)
    df['Delay_Days'] = reg.predict(df)
    df['Delay_Status'] = df['Delay_Status'].map({1:"Delayed",0:"On Time"})

    def risk(x):
        if x > 20: return "🔴 High"
        elif x > 10: return "🟡 Medium"
        else: return "🟢 Low"

    df['Risk_Level'] = df['Delay_Days'].apply(risk)

    st.markdown("## 📊 Results")
    st.dataframe(df)

    # KPI
    st.markdown("## 📈 Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Projects", len(df))
    c2.metric("Delayed", (df['Delay_Status']=="Delayed").sum())
    c3.metric("High Risk", (df['Risk_Level']=="🔴 High").sum())

    st.bar_chart(df['Risk_Level'].value_counts())

    # Download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Results", csv, "results.csv")

# ===============================
# FOOTER
# ===============================
st.markdown("<div class='footer'>Made by Esra • AI Project System</div>", unsafe_allow_html=True)