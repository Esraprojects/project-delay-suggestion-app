# ... (imports and previous code unchanged until the main function) ...

def main():
    # Sidebar
    st.sidebar.header("📁 Data Source")
    data_source = st.sidebar.radio("Choose data source:", ["Use Sample Data", "Upload Your Data"])
    
    use_tuning = st.sidebar.checkbox("Enable hyperparameter tuning (slower but potentially better)", value=False)
    
    df = None
    if data_source == "Use Sample Data":
        with st.spinner("Generating sample dataset..."):
            df = generate_sample_data(50)
        st.sidebar.success("Sample data ready (50 projects).")
        # Download sample template (only input features)
        csv_sample = df[FEATURE_COLS].to_csv(index=False).encode('utf-8')
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
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.sidebar.success(f"Loaded {len(df)} rows")
            except Exception as e:
                st.sidebar.error(f"Error reading file: {e}")
                st.stop()
        else:
            st.sidebar.info("Please upload a file to continue.")
            st.stop()
    
    # Show data preview - conditionally show only input features for sample data
    with st.expander("📋 Data Preview (first 10 rows)"):
        if data_source == "Use Sample Data":
            # Show only the input features
            st.dataframe(df[FEATURE_COLS].head(10))
            st.caption("Note: The sample data includes only the input features shown above. The target columns (Delay_Days, Status) are generated internally for training.")
        else:
            # Show the entire uploaded file as is
            st.dataframe(df.head(10))
    
    # ... rest of the code unchanged ...
