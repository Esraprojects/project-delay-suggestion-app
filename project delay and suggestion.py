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
# HEADER
# ===============================
st.title("🏗️ Project Delay AI System")
st.write("Smart Prediction • Risk Analysis • Decision Support")
st.markdown("<div style='text-align:center; color:#00ffcc;'>Made by Esra</div>", unsafe_allow_html=True)

# ===============================
# FEATURE DEFINITION (SOURCE OF TRUTH)
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

numeric_cols = [col for col in feature_cols if col not in categorical_cols]

# ===============================
# SAMPLE TEMPLATE (ALIGNED EXACTLY)
# ===============================
st.subheader("📥 Download Correct Template")
sample_data = pd.DataFrame(columns=feature_cols)

st.download_button("Download Template", sample_data.to_csv(index=False), "template.csv")
st.dataframe(sample_data)

# ===============================
# COLUMN NORMALIZATION
# ===============================
def normalize_columns(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

COLUMN_ALIASES = {col.lower(): col for col in feature_cols}

# ===============================
# MODEL TRAINING
# ===============================
@st.cache_resource
def train_models():
    try:
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

        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ], remainder='passthrough')

        clf = Pipeline([('prep', preprocessor), ('model', RandomForestClassifier())])
        reg = Pipeline([('prep', preprocessor), ('model', RandomForestRegressor())])

        clf.fit(X, y_class)
        reg.fit(X, y_reg)

        return clf, reg

    except Exception as e:
        st.error(f"Model training failed: {e}")
        return None, None

clf, reg = train_models()

# ===============================
# FILE UPLOAD
# ===============================
st.subheader("📂 Upload File (CSV or Excel)")
file = st.file_uploader("Upload your dataset", type=["csv","xlsx"])

if file:
    try:
        # File reading
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        if df.empty:
            st.warning("Uploaded file is empty")
            st.stop()

        df = normalize_columns(df)
        df = df.rename(columns=COLUMN_ALIASES)

        st.success("File processed successfully")
        st.dataframe(df)

        # Ensure all required columns exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = np.nan

        X = df[feature_cols].copy()

        # Numeric handling
        for col in numeric_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            if X[col].isnull().all():
                X[col] = 0
            else:
                X[col] = X[col].fillna(X[col].median())

        # Categorical handling
        for col in categorical_cols:
            X[col] = X[col].astype(str).fillna('Unknown')

        # Prediction safety
        if clf is None or reg is None:
            st.error("Model not available")
            st.stop()

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

        st.download_button("Download Results", df.to_csv(index=False), "results.csv")

    except Exception as e:
        st.error("An error occurred while processing the file")
        st.exception(e)
