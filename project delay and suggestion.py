# ===============================
# IMPORTS (SAFE)
# ===============================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Safe report import
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet
    REPORT_AVAILABLE = True
except:
    REPORT_AVAILABLE = False

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
# FEATURES
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
# SAMPLE DATA (REALISTIC 100 ROWS)
# ===============================
st.subheader("📥 Download Realistic Template")

try:
    np.random.seed(2)
    sample_data = pd.DataFrame({
        'Project_Size': np.random.choice(['Small','Medium','Large'],100),
        'Budget': np.random.randint(100000,800000,100),
        'Team_Size': np.random.randint(5,30,100),
        'Planned_Duration': np.random.randint(30,180,100),
        'Weather': np.random.choice(['Good','Bad'],100),
        'Material_Availability': np.random.choice(['Yes','No'],100),
        'Manager_Experience': np.random.randint(1,15,100),
        'Contractor_Experience': np.random.randint(3,20,100),
        'Labor_Availability': np.random.choice(['High','Medium','Low'],100),
        'Equipment_Availability': np.random.choice(['Yes','No'],100),
        'Site_Location': np.random.choice(['Urban','Rural'],100),
        'Permit_Approval': np.random.choice(['Yes','No'],100),
        'Inflation_Rate': np.random.uniform(5,25,100),
        'Supply_Delay': np.random.choice(['Yes','No'],100)
    })

    st.download_button("Download Template", sample_data.to_csv(index=False), "template.csv")
    st.dataframe(sample_data.head(20))

except Exception as e:
    st.error("Error generating sample data")
    st.exception(e)

# ===============================
# MODEL TRAINING
# ===============================
@st.cache_resource
def train_model():
    try:
        df = sample_data.copy()
        delays = []

        for _, r in df.iterrows():
            d = np.random.randint(0, 8)

            if r['Weather']=='Bad': d+=np.random.randint(5,12)
            if r['Material_Availability']=='No': d+=np.random.randint(5,15)
            if r['Labor_Availability']=='Low': d+=np.random.randint(3,10)
            if r['Equipment_Availability']=='No': d+=np.random.randint(3,8)
            if r['Permit_Approval']=='No': d+=np.random.randint(5,12)
            if r['Supply_Delay']=='Yes': d+=np.random.randint(4,10)
            if r['Site_Location']=='Rural': d+=np.random.randint(2,6)
            if r['Manager_Experience']<5: d+=np.random.randint(3,8)

            delays.append(d)

        df['Delay_Days'] = delays
        df['Status'] = (df['Delay_Days'] > 15).astype(int)

        X = df[feature_cols]
        y1 = df['Status']
        y2 = df['Delay_Days']

        pre = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ], remainder='passthrough')

        clf = Pipeline([('p', pre), ('m', RandomForestClassifier())])
        reg = Pipeline([('p', pre), ('m', RandomForestRegressor())])

        clf.fit(X, y1)
        reg.fit(X, y2)

        return clf, reg

    except Exception as e:
        st.error("Model training failed")
        st.exception(e)
        return None, None

clf, reg = train_model()

# ===============================
# FILE UPLOAD
# ===============================
st.subheader("📂 Upload File (CSV or Excel)")
file = st.file_uploader("Upload your dataset", type=["csv","xlsx"])

if file:
    try:
        # Read file
        df = pd.read_csv(file) if file.name.endswith("csv") else pd.read_excel(file)

        if df.empty:
            st.warning("File is empty")
            st.stop()

        # Normalize columns
        df.columns = df.columns.str.strip().str.replace(" ", "_")

        # Ensure required columns exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = np.nan

        X = df[feature_cols].copy()

        # Clean numeric
        for col in numeric_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].fillna(X[col].median())

        # Clean categorical
        for col in categorical_cols:
            X[col] = X[col].astype(str).fillna("Unknown")

        # Predict
        if clf is None or reg is None:
            st.error("Model not available")
            st.stop()

        df['Delay_Status'] = clf.predict(X)
        df['Delay_Days'] = reg.predict(X)
        df['Delay_Status'] = df['Delay_Status'].map({1:"Delayed",0:"On Time"})

        # Risk
        def risk(x):
            return "🔴 High" if x>20 else "🟡 Medium" if x>10 else "🟢 Low"

        df['Risk_Level'] = df['Delay_Days'].apply(risk)

        # ===============================
        # ROOT CAUSES
        # ===============================
        def causes(row):
            c = []
            if row['Weather']=='Bad': c.append('Weather')
            if row['Material_Availability']=='No': c.append('Material')
            if row['Labor_Availability']=='Low': c.append('Labor')
            if row['Equipment_Availability']=='No': c.append('Equipment')
            if row['Permit_Approval']=='No': c.append('Permit')
            if row['Supply_Delay']=='Yes': c.append('Supply')
            return ", ".join(c) if c else "Normal"

        df['Main_Causes'] = df.apply(causes, axis=1)

        # ===============================
        # RESULTS
        # ===============================
        st.subheader("📊 Results")
        st.dataframe(df)

        # ===============================
        # VISUALS
        # ===============================
        st.subheader("📈 Visual Analysis")

        # Risk Chart
        fig1 = plt.figure()
        df['Risk_Level'].value_counts().plot(kind='bar')
        plt.title("Risk Distribution")
        st.pyplot(fig1)
        st.info("Shows distribution of project risks.")

        # Delay Histogram
        fig2 = plt.figure()
        df['Delay_Days'].plot(kind='hist', bins=20)
        plt.title("Delay Distribution")
        st.pyplot(fig2)
        st.info("Shows how delays are spread.")

        # Causes
        fig3 = plt.figure()
        df['Main_Causes'].value_counts().head(5).plot(kind='bar')
        plt.title("Top Causes")
        st.pyplot(fig3)
        st.info("Top reasons causing delays.")

        # ===============================
        # INTERPRETATION
        # ===============================
        st.subheader("🧠 Insights")

        total = len(df)
        high = len(df[df['Risk_Level']=="🔴 High"])

        st.write(f"""
        Total Projects: {total}  
        High Risk: {high}  

        Key Issues:
        - Supply chain delays
        - Material shortages
        - Low experience management

        Fixes:
        - Improve planning
        - Strengthen logistics
        - Assign experienced managers
        """)

    except Exception as e:
        st.error("Error processing file")
        st.exception(e)
