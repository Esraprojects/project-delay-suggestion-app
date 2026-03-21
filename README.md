PROJECT DELAY AI SYSTEM

An AI-powered Streamlit web application that predicts construction project delays using Machine Learning. The system analyzes uploaded Excel data and provides delay predictions, estimated delay days, and risk levels.

FEATURES

* Upload Excel files (.xlsx)
* Predict delay status (On Time / Delayed)
* Estimate delay duration in days
* Risk level classification (Low / Medium / High)
* Interactive dashboard with KPIs and charts
* Download results as CSV
* Built with Random Forest models
* Clean and modern UI

TECHNOLOGIES USED

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn

PROJECT STRUCTURE
project-delay-ai/

* app.py
* requirements.txt
* README.txt

INPUT DATA FORMAT
The Excel file must contain the following columns:

* Project_Size (Small, Medium, Large)
* Budget
* Team_Size
* Planned_Duration
* Weather (Good, Bad)
* Material_Availability (Yes, No)
* Manager_Experience
* Contractor_Experience
* Labor_Availability (Low, Medium, High)
* Equipment_Availability (Yes, No)
* Site_Location (Urban, Rural)
* Permit_Approval (Yes, No)
* Inflation_Rate
* Supply_Delay (Yes, No)

INSTALLATION AND SETUP

1. Clone the repository:
   git clone [https://github.com/your-username/project-delay-ai.git](https://github.com/your-username/project-delay-ai.git)
   cd project-delay-ai

2. Create a virtual environment (optional):
   python -m venv venv

Activate the environment:
Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt

RUN THE APPLICATION

streamlit run app.py

Then open the URL shown in the terminal (usually [http://localhost:8501](http://localhost:8501))

HOW TO USE

1. Open the application in your browser
2. Download the sample template (optional)
3. Prepare your Excel file using the required format
4. Upload the Excel file
5. View predictions, delay estimates, and risk levels
6. Download the results as a CSV file

REQUIREMENTS

streamlit
pandas
numpy
scikit-learn
openpyxl

NOTES

* Column names must match exactly with the template
* The system automatically encodes categorical values
* Ensure the dataset is clean and properly formatted

AUTHOR
Made by Esra
