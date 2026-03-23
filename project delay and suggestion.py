# ===============================
# IMPORTS
# ===============================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import io
import tempfile
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import warnings
warnings.filterwarnings('ignore')

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Project Delay AI", layout="wide", page_icon="📊")
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🏗️ Project Delay AI System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Smart Prediction • Risk Analysis • Decision Support</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 14px;'>Made by Esrom</p>", unsafe_allow_html=True)
st.markdown("---")

# ===============================
# CONSTANTS
# ===============================
FEATURE_COLS = [
    'Project_Size', 'Budget', 'Team_Size', 'Planned_Duration', 'Weather',
    'Material_Availability', 'Manager_Experience', 'Contractor_Experience',
    'Labor_Availability', 'Equipment_Availability', 'Site_Location',
    'Permit_Approval', 'Inflation_Rate', 'Supply_Delay'
]

CATEGORICAL_COLS = [
    'Project_Size', 'Weather', 'Material_Availability', 'Labor_Availability',
    'Equipment_Availability', 'Site_Location', 'Permit_Approval', 'Supply_Delay'
]

NUMERIC_COLS = [col for col in FEATURE_COLS if col not in CATEGORICAL_COLS]

# ===============================
# DATA GENERATION & MODEL TRAINING
# ===============================
@st.cache_data
def generate_training_data(n=500):
    """Generate a large synthetic dataset for model training."""
    np.random.seed(42)
    data = pd.DataFrame({
        'Project_Size': np.random.choice(['Small', 'Medium', 'Large'], n, p=[0.4, 0.4, 0.2]),
        'Budget': np.random.randint(50000, 2000000, n),
        'Team_Size': np.random.randint(3, 50, n),
        'Planned_Duration': np.random.randint(30, 365, n),
        'Weather': np.random.choice(['Good', 'Bad'], n, p=[0.7, 0.3]),
        'Material_Availability': np.random.choice(['Yes', 'No'], n, p=[0.8, 0.2]),
        'Manager_Experience': np.random.randint(1, 25, n),
        'Contractor_Experience': np.random.randint(1, 30, n),
        'Labor_Availability': np.random.choice(['High', 'Medium', 'Low'], n, p=[0.5, 0.3, 0.2]),
        'Equipment_Availability': np.random.choice(['Yes', 'No'], n, p=[0.9, 0.1]),
        'Site_Location': np.random.choice(['Urban', 'Rural'], n, p=[0.6, 0.4]),
        'Permit_Approval': np.random.choice(['Yes', 'No'], n, p=[0.85, 0.15]),
        'Inflation_Rate': np.random.uniform(2, 15, n),
        'Supply_Delay': np.random.choice(['Yes', 'No'], n, p=[0.7, 0.3])
    })
    
    delays = []
    for _, r in data.iterrows():
        base_delay = np.random.normal(0, 3)
        if r['Weather'] == 'Bad':
            base_delay += np.random.uniform(5, 15)
        if r['Material_Availability'] == 'No':
            base_delay += np.random.uniform(5, 20)
        if r['Labor_Availability'] == 'Low':
            base_delay += np.random.uniform(3, 12)
        if r['Equipment_Availability'] == 'No':
            base_delay += np.random.uniform(2, 10)
        if r['Permit_Approval'] == 'No':
            base_delay += np.random.uniform(5, 18)
        if r['Supply_Delay'] == 'Yes':
            base_delay += np.random.uniform(4, 12)
        if r['Site_Location'] == 'Rural':
            base_delay += np.random.uniform(1, 8)
        if r['Manager_Experience'] < 5:
            base_delay += np.random.uniform(2, 10)
        if r['Contractor_Experience'] < 3:
            base_delay += np.random.uniform(1, 7)
        delays.append(max(0, base_delay))
    
    data['Delay_Days'] = delays
    data['Status'] = (data['Delay_Days'] > 15).astype(int)
    return data

@st.cache_data
def generate_sample_data_for_user(n=50):
    """Generate a separate sample dataset (without target columns) for user download."""
    np.random.seed(42)  # different seed to ensure diversity from training data
    data = pd.DataFrame({
        'Project_Size': np.random.choice(['Small', 'Medium', 'Large'], n, p=[0.4, 0.4, 0.2]),
        'Budget': np.random.randint(50000, 2000000, n),
        'Team_Size': np.random.randint(3, 50, n),
        'Planned_Duration': np.random.randint(30, 365, n),
        'Weather': np.random.choice(['Good', 'Bad'], n, p=[0.7, 0.3]),
        'Material_Availability': np.random.choice(['Yes', 'No'], n, p=[0.8, 0.2]),
        'Manager_Experience': np.random.randint(1, 25, n),
        'Contractor_Experience': np.random.randint(1, 30, n),
        'Labor_Availability': np.random.choice(['High', 'Medium', 'Low'], n, p=[0.5, 0.3, 0.2]),
        'Equipment_Availability': np.random.choice(['Yes', 'No'], n, p=[0.9, 0.1]),
        'Site_Location': np.random.choice(['Urban', 'Rural'], n, p=[0.6, 0.4]),
        'Permit_Approval': np.random.choice(['Yes', 'No'], n, p=[0.85, 0.15]),
        'Inflation_Rate': np.random.uniform(2, 15, n),
        'Supply_Delay': np.random.choice(['Yes', 'No'], n, p=[0.7, 0.3])
    })
    return data

@st.cache_resource
def train_models(training_df, tune=False):
    """Train models on the provided training dataset."""
    X = training_df[FEATURE_COLS]
    y_class = training_df['Status']
    y_reg = training_df['Delay_Days']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_COLS),
            ('cat', categorical_transformer, CATEGORICAL_COLS)
        ])
    
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])
    reg = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    
    if tune:
        param_grid_clf = {'classifier__n_estimators': [100, 200]}
        clf = GridSearchCV(clf, param_grid_clf, cv=3, scoring='f1', n_jobs=-1)
        param_grid_reg = {'regressor__n_estimators': [100, 200]}
        reg = GridSearchCV(reg, param_grid_reg, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        clf.fit(X, y_class)
        reg.fit(X, y_reg)
        return clf.best_estimator_, reg.best_estimator_
    else:
        clf.fit(X, y_class)
        reg.fit(X, y_reg)
        return clf, reg

# ===============================
# PREDICTION & UTILITIES
# ===============================
def preprocess_for_prediction(df):
    """Prepare any dataframe for prediction (fill missing, convert types)."""
    X = df[FEATURE_COLS].copy()
    for col in NUMERIC_COLS:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col] = X[col].fillna(X[col].median())
    for col in CATEGORICAL_COLS:
        X[col] = X[col].astype(str).fillna('Unknown')
    return X

def add_predictions(df, clf, reg):
    """Add predictions and derived columns to a dataframe."""
    X = preprocess_for_prediction(df)
    df['Delay_Status'] = clf.predict(X)
    df['Delay_Days'] = reg.predict(X)
    df['Delay_Status'] = df['Delay_Status'].map({1: "Delayed", 0: "On Time"})
    
    def risk_level(days):
        if days > 20:
            return "🔴 High"
        elif days > 10:
            return "🟡 Medium"
        else:
            return "🟢 Low"
    df['Risk_Level'] = df['Delay_Days'].apply(risk_level)
    
    def identify_causes(row):
        causes = []
        if row.get('Weather') == 'Bad':
            causes.append('Weather')
        if row.get('Material_Availability') == 'No':
            causes.append('Material Shortage')
        if row.get('Labor_Availability') == 'Low':
            causes.append('Labor Shortage')
        if row.get('Equipment_Availability') == 'No':
            causes.append('Equipment Issue')
        if row.get('Permit_Approval') == 'No':
            causes.append('Permit Delay')
        if row.get('Supply_Delay') == 'Yes':
            causes.append('Supply Chain')
        if row.get('Manager_Experience', 10) < 5:
            causes.append('Inexperienced Manager')
        return ', '.join(causes) if causes else 'No clear cause'
    df['Main_Causes'] = df.apply(identify_causes, axis=1)
    return df

def plot_risk_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    df['Risk_Level'].value_counts().plot(kind='bar', color=['#28a745', '#ffc107', '#dc3545'], ax=ax)
    ax.set_title('Risk Distribution', fontsize=14)
    ax.set_xlabel('Risk Level')
    ax.set_ylabel('Number of Projects')
    ax.grid(axis='y', alpha=0.3)
    return fig

def plot_delay_histogram(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    df['Delay_Days'].hist(bins=20, edgecolor='black', ax=ax)
    ax.set_title('Distribution of Predicted Delay Days', fontsize=14)
    ax.set_xlabel('Delay Days')
    ax.set_ylabel('Frequency')
    ax.grid(axis='y', alpha=0.3)
    return fig

def plot_top_causes(df):
    cause_counts = df['Main_Causes'].value_counts().head(5)
    fig, ax = plt.subplots(figsize=(8, 5))
    cause_counts.plot(kind='barh', ax=ax)
    ax.set_title('Top 5 Causes of Delay', fontsize=14)
    ax.set_xlabel('Number of Projects')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    return fig

def generate_pdf_report(df, clf_score, reg_score, risk_fig, delay_fig, cause_fig):
    """Create PDF report with analysis and suggestions."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=20, alignment=1, spaceAfter=12)
    story.append(Paragraph("Project Delay Analysis Report", title_style))
    story.append(Spacer(1, 12))
    
    from datetime import datetime
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Executive summary
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    story.append(Spacer(1, 6))
    total = len(df)
    delayed = len(df[df['Delay_Status'] == 'Delayed'])
    high_risk = len(df[df['Risk_Level'] == '🔴 High'])
    avg_delay = df['Delay_Days'].mean()
    story.append(Paragraph(f"Total projects analyzed: {total}", styles['Normal']))
    story.append(Paragraph(f"Delayed projects: {delayed} ({delayed/total*100:.1f}%)", styles['Normal']))
    story.append(Paragraph(f"High-risk projects: {high_risk} ({high_risk/total*100:.1f}%)", styles['Normal']))
    story.append(Paragraph(f"Average predicted delay: {avg_delay:.1f} days", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Model performance
    story.append(Paragraph("Model Performance", styles['Heading2']))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Classifier F1-score (cross-validation): {clf_score:.3f}", styles['Normal']))
    story.append(Paragraph(f"Regressor MAE (cross-validation): {reg_score:.1f} days", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Deep analysis
    story.append(Paragraph("Deep Analysis of Results", styles['Heading2']))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Risk Distribution:", styles['Heading3']))
    story.append(Paragraph(f"The analysis shows that {high_risk} projects are at high risk of delay (>20 days). "
                           f"Medium risk projects ({len(df[df['Risk_Level'] == '🟡 Medium'])}) require attention, while low-risk projects are generally on track.",
                           styles['Normal']))
    story.append(Spacer(1, 6))
    
    top_cause = df['Main_Causes'].value_counts().index[0] if not df.empty else "N/A"
    story.append(Paragraph("Main Delay Factors:", styles['Heading3']))
    story.append(Paragraph(f"The most frequent cause of delay is '{top_cause}', affecting a significant portion of projects. "
                           f"Other common issues include material shortages, weather, and supply chain disruptions.",
                           styles['Normal']))
    story.append(Spacer(1, 6))
    
    # Suggestions
    story.append(Paragraph("Actionable Recommendations", styles['Heading2']))
    story.append(Spacer(1, 6))
    suggestions = [
        "• For high-risk projects, conduct weekly progress reviews and allocate contingency budgets.",
        "• Address material shortages by diversifying suppliers and maintaining safety stock.",
        "• Improve communication with contractors to mitigate permit and supply delays.",
        "• Assign experienced project managers to complex projects.",
        "• Implement weather contingency plans for sites in rural or adverse weather zones."
    ]
    for s in suggestions:
        story.append(Paragraph(s, styles['Normal']))
        story.append(Spacer(1, 4))
    story.append(Spacer(1, 12))
    
    # Embed charts
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp1:
        risk_fig.savefig(tmp1.name, format='png', dpi=100)
        story.append(Image(tmp1.name, width=6*inch, height=4*inch))
        story.append(Spacer(1, 12))
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp2:
        delay_fig.savefig(tmp2.name, format='png', dpi=100)
        story.append(Image(tmp2.name, width=6*inch, height=4*inch))
        story.append(Spacer(1, 12))
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp3:
        cause_fig.savefig(tmp3.name, format='png', dpi=100)
        story.append(Image(tmp3.name, width=6*inch, height=4*inch))
        story.append(Spacer(1, 12))
    
    story.append(Paragraph("This report was generated automatically by Project Delay AI. "
                           "For further assistance, contact support.", styles['Normal']))
    doc.build(story)
    buffer.seek(0)
    return buffer

# ===============================
# MAIN APP
# ===============================
def main():
    # Sidebar
    st.sidebar.header("📁 Data Source")
    data_source = st.sidebar.radio("Choose data source:", ["Use Sample Data", "Upload Your Data"])
    use_tuning = st.sidebar.checkbox("Enable hyperparameter tuning (slower but potentially better)", value=False)
    
    # --- Train the model (always on a large synthetic dataset) ---
    with st.spinner("Training prediction models..."):
        training_df = generate_training_data(500)   # large dataset for training
        clf, reg = train_models(training_df, tune=use_tuning)
    
    # Cross-validation scores for display
    X_train = training_df[FEATURE_COLS]
    y_class = training_df['Status']
    y_reg = training_df['Delay_Days']
    clf_cv = cross_val_score(clf, X_train, y_class, cv=3, scoring='f1').mean()
    reg_cv = -cross_val_score(reg, X_train, y_reg, cv=3, scoring='neg_mean_absolute_error').mean()
    
    # --- Data acquisition and prediction ---
    if data_source == "Use Sample Data":
        # Generate a fresh sample (input only) for the user to see
        df_input = generate_sample_data_for_user(50)  # only FEATURE_COLS
        st.sidebar.success("Sample data ready (50 projects).")
        # Download sample template
        csv_sample = df_input.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="📥 Download Sample Data (CSV)",
            data=csv_sample,
            file_name="project_sample.csv",
            mime="text/csv",
            help="Download this sample to test or modify."
        )
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_input = pd.read_csv(uploaded_file)
                else:
                    df_input = pd.read_excel(uploaded_file)
                st.sidebar.success(f"Loaded {len(df_input)} rows")
            except Exception as e:
                st.sidebar.error(f"Error reading file: {e}")
                st.stop()
        else:
            st.sidebar.info("Please upload a file to continue.")
            st.stop()
    
    # --- Data preview (show only input columns for sample data) ---
    with st.expander("📋 Data Preview (first 10 rows)"):
        if data_source == "Use Sample Data":
            st.dataframe(df_input.head(10))
            st.caption("Note: This sample contains only the input features required for prediction.")
        else:
            # Show the full uploaded file (may contain extra columns)
            st.dataframe(df_input.head(10))
    
    # --- Predictions ---
    result_df = add_predictions(df_input.copy(), clf, reg)
    
    # --- Display results ---
    st.subheader("📊 Prediction Results")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Projects", len(result_df))
    with col2:
        delayed = len(result_df[result_df['Delay_Status'] == 'Delayed'])
        st.metric("Delayed Projects", delayed, delta=f"{delayed/len(result_df)*100:.1f}%")
    with col3:
        avg_delay = result_df['Delay_Days'].mean()
        st.metric("Avg Delay Days", f"{avg_delay:.1f}")
    with col4:
        high_risk = len(result_df[result_df['Risk_Level'] == '🔴 High'])
        st.metric("High Risk Projects", high_risk)
    
    st.subheader("📄 Detailed Results")
    st.dataframe(result_df, use_container_width=True)
    
    # Download results CSV
    csv_results = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results CSV", csv_results, "delay_results.csv", "text/csv")
    
    # --- Visualizations ---
    st.subheader("📈 Visual Analysis")
    risk_fig = plot_risk_distribution(result_df)
    delay_fig = plot_delay_histogram(result_df)
    cause_fig = plot_top_causes(result_df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(risk_fig)
        st.caption("Risk levels of projects")
    with col2:
        st.pyplot(delay_fig)
        st.caption("Distribution of predicted delay days")
    st.pyplot(cause_fig)
    st.caption("Top root causes contributing to delays")
    
    # --- PDF Report ---
    st.subheader("📄 Generate Comprehensive Report")
    if st.button("Generate PDF Report with Analysis & Suggestions"):
        with st.spinner("Creating report..."):
            pdf_buffer = generate_pdf_report(
                result_df, clf_cv, reg_cv,
                risk_fig, delay_fig, cause_fig
            )
        st.download_button(
            label="⬇️ Download Full Report (PDF)",
            data=pdf_buffer,
            file_name="project_delay_report.pdf",
            mime="application/pdf"
        )
        st.success("Report generated! Click the button above to download.")
    
    # Footer
    st.markdown("---")
    st.markdown("<p style='text-align: center;'>Made by Esrom | Project Delay AI System</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
