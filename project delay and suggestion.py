import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import os

st.set_page_config(page_title="Project Delay AI", layout="wide")

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
# REALISTIC SAMPLE DATA (100 ROWS)
# ===============================
st.subheader("📥 Download Realistic Template (100 Rows)")
np.random.seed(1)
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

# ===============================
# MODEL
# ===============================
@st.cache_resource
def train():
    df = sample_data.copy()

    delay = []
    for _, r in df.iterrows():
        d = 0
        if r['Weather']=='Bad': d+=10
        if r['Material_Availability']=='No': d+=15
        if r['Labor_Availability']=='Low': d+=10
        if r['Equipment_Availability']=='No': d+=8
        if r['Permit_Approval']=='No': d+=12
        if r['Supply_Delay']=='Yes': d+=10
        if r['Site_Location']=='Rural': d+=5
        if r['Manager_Experience']<5: d+=7
        d += int(r['Inflation_Rate']/5)
        delay.append(d)

    df['Delay_Days']=delay
    df['Status']=(df['Delay_Days']>10).astype(int)

    X=df[feature_cols]
    y1=df['Status']
    y2=df['Delay_Days']

    pre=ColumnTransformer([
        ('cat',OneHotEncoder(handle_unknown='ignore'),categorical_cols)
    ],remainder='passthrough')

    clf=Pipeline([('p',pre),('m',RandomForestClassifier())])
    reg=Pipeline([('p',pre),('m',RandomForestRegressor())])

    clf.fit(X,y1)
    reg.fit(X,y2)

    return clf,reg

clf,reg=train()

# ===============================
# UPLOAD
# ===============================
file=st.file_uploader("Upload CSV/Excel",type=['csv','xlsx'])

if file:
    try:
        df=pd.read_csv(file) if file.name.endswith('csv') else pd.read_excel(file)

        # auto fix columns
        df.columns=df.columns.str.strip().str.replace(' ','_')

        for col in feature_cols:
            if col not in df.columns:
                df[col]=np.nan

        X=df[feature_cols].copy()

        for col in numeric_cols:
            X[col]=pd.to_numeric(X[col],errors='coerce').fillna(X[col].median())

        for col in categorical_cols:
            X[col]=X[col].astype(str).fillna('Unknown')

        df['Delay_Status']=clf.predict(X)
        df['Delay_Days']=reg.predict(X)
        df['Delay_Status']=df['Delay_Status'].map({1:'Delayed',0:'On Time'})

        def risk(x):
            return 'High' if x>20 else 'Medium' if x>10 else 'Low'

        df['Risk_Level']=df['Delay_Days'].apply(risk)

        st.subheader("📊 Results")
        st.dataframe(df)

        # ===============================
        # GRAPHICAL ANALYSIS
        # ===============================
        st.subheader("📈 Visual Analysis")

        fig1=plt.figure()
        df['Risk_Level'].value_counts().plot(kind='bar',color=['green','orange','red'])
        plt.title('Risk Distribution')
        plt.ylabel('Projects')
        plt.savefig('risk.png')
        st.pyplot(fig1)

        fig2=plt.figure()
        df['Delay_Days'].plot(kind='hist',bins=20,color='blue')
        plt.title('Delay Distribution')
        plt.savefig('delay.png')
        st.pyplot(fig2)

        # ===============================
        # INSIGHTS & SUGGESTIONS
        # ===============================
        st.subheader("🧠 AI Suggestions")

        high=df[df['Risk_Level']=='High']

        suggestions="""
        Key Insights:\n
        - High delays are mainly caused by poor material availability, bad weather, and permit issues.\n
        Quick Fixes:\n
        - Improve supply chain logistics\n        - Ensure early permit approvals\n        - Increase experienced managers\n        - Maintain backup equipment\n        - Plan for weather risks\n        """

        st.write(suggestions)

        # ===============================
        # PDF REPORT
        # ===============================
        if st.button("Generate Full Report PDF"):
            try:
                doc=SimpleDocTemplate("report.pdf")
                styles=getSampleStyleSheet()
                elements=[]

                elements.append(Paragraph("Project Delay AI Report",styles['Title']))
                elements.append(Spacer(1,10))
                elements.append(Paragraph(suggestions,styles['Normal']))
                elements.append(Spacer(1,10))

                elements.append(Image('risk.png',width=400,height=200))
                elements.append(Image('delay.png',width=400,height=200))

                doc.build(elements)

                with open("report.pdf","rb") as f:
                    st.download_button("Download Report",f,"report.pdf")

            except Exception as e:
                st.error("Report generation failed")
                st.exception(e)

    except Exception as e:
        st.error("Error processing file")
        st.exception(e)
