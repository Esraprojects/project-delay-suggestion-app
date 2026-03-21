import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Project Delay AI", layout="wide")

# ===============================
# UI HEADER
# ===============================
st.title("🏗️ Project Delay AI System")
st.write("Smart Prediction • Risk Analysis • Decision Support")
st.markdown("<div style='text-align:center; color:#00ffcc;'>Made by Esra</div>", unsafe_allow_html=True)

# ===============================
# SAMPLE DATA
# ===============================
st.subheader("📥 Sample Template")
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

st.download_button("Download Template", sample_data.to_csv(index=False), "template.csv")
st.dataframe(sample_data)

# ===============================
# FEATURE SET
# ===============================
feature_cols = [
    'Project_Size','Budget','Team_Size','Planned_Duration','Weather',
    'Material_Availability','Manager_Experience','Contractor_Experience',
    'Labor_Availability','Equipment_Availability','Site_Location',
    'Permit_Approval','Inflation_Rate','Supply_Delay'
]

categorical_cols = [
    'Project_Size','Weather','Material_Availability','Labor_Availability',
    'Equipment_Availability','Site_Location','Permit_Approval','Supply_Delay'
]

# ===============================
# COLUMN NORMALIZATION + AUTO MATCHING
# ===============================

def normalize_columns(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

# Alias mapping for flexible input handling
COLUMN_ALIASES = {
    'project_size': 'Project_Size',
    'budget': 'Budget',
    'team_size': 'Team_Size',
    'planned_duration': 'Planned_Duration',
    'weather': 'Weather',
    'material_availability': 'Material_Availability',
    'manager_experience': 'Manager_Experience',
    'contractor_experience': 'Contractor_Experience',
    'labor_availability': 'Labor_Availability',
    'equipment_availability': 'Equipment_Availability',
    'site_location': 'Site_Location',
    'permit_approval': 'Permit_Approval',
    'inflation_rate': 'Inflation_Rate',
    'supply_delay': 'Supply_Delay'
}

# ===============================
# MODEL TRAINING PIPELINE
# ===============================
@st.cache_resource
def train_models():
    np.random.seed(42)
    data = []

    for _ in range(1200):
        row = {
            'Project_Size': np.random.choice(['Small','Medium','Large']),
            'Budget': np.random.randint(80000,700000),
            'Team_Size': np.random.randint(3,30),
            'Planned_Duration': np.random.randint(20,180),
            'Weather': np.random.choice(['Good','Bad']),
            'Material_Availability': np.random.choice(['Yes','No']),
            'Manager_Experience': np.random.randint(1,15),
            'Contractor_Experience': np.random.randint(1,20),
            'Labor_Availability': np.random.choice(['High','Medium','Low']),
            'Equipment_Availability': np.random.choice(['Yes','No']),
            'Site_Location': np.random.choice(['Urban','Rural']),
            'Permit_Approval': np.random.choice(['Yes','No']),
            'Inflation_Rate': np.random.uniform(5,30),
            'Supply_Delay': np.random.choice(['Yes','No'])
        }

        delay = 0
        if row['Weather']=='Bad': delay += 10
        if row['Material_Availability']=='No': delay += 15
        if row['Labor_Availability']=='Low': delay += 10
        if row['Equipment_Availability']=='No': delay += 8
        if row['Permit_Approval']=='No': delay += 12
        if row['Supply_Delay']=='Yes': delay += 10
        if row['Site_Location']=='Rural': delay += 5
        if row['Manager_Experience'] < 5: delay += 7

        delay += int(row['Inflation_Rate']/5)
        delay += np.random.randint(0,5)

        row['Delay_Days'] = delay
        row['Status'] = 1 if delay > 10 else 0

        data.append(row)

    df = pd.DataFrame(data)

    X = df[feature_cols]
    y_class = df['Status']
    y_reg = df['Delay_Days']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )

    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier())
    ])

    reg = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor())
    ])

    clf.fit(X, y_class)
    reg.fit(X, y_reg)

    return clf, reg

clf, reg = train_models()

# ===============================
# FILE UPLOAD
# ===============================
st.subheader("📂 Upload File (Excel or CSV)")
file = st.file_uploader("Upload your dataset", type=["csv","xlsx"])

if file:
    # Read file
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    # Normalize column names
    df = normalize_columns(df)

    # Rename columns using aliases
    df = df.rename(columns=COLUMN_ALIASES)

    st.success("File uploaded and processed successfully")
    st.dataframe(df)

    # Keep only relevant columns, create missing ones if absent
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Reorder columns exactly as expected
    X = df[feature_cols]

    # Handle missing numeric values
    for col in numeric_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].fillna(X[col].median())

    # Handle missing categorical values
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].fillna('Unknown')

    # Predictions
    df['Delay_Status'] = clf.predict(X)
    df['Delay_Days'] = reg.predict(X)

    df['Delay_Status'] = df['Delay_Status'].map({1:"Delayed",0:"On Time"})

    def risk(x):
        if x > 20:
            return "🔴 High"
        elif x > 10:
            return "🟡 Medium"
        return "🟢 Low"

    df['Risk_Level'] = df['Delay_Days'].apply(risk)

    st.subheader("📊 Results")
    st.dataframe(df)

    st.subheader("📈 Dashboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Projects", len(df))
    c2.metric("Delayed", (df['Delay_Status']=="Delayed").sum())
    c3.metric("High Risk", (df['Risk_Level']=="🔴 High").sum())

    st.bar_chart(df['Risk_Level'].value_counts())

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Results", csv, "results.csv")
