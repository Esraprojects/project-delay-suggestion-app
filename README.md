# 🏗️ Project Delay AI System made by "ESROM ADUGNA GSE/4064/2018"

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)  
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Smart Prediction • Risk Analysis • Decision Support**  
An AI‑powered web application that predicts project delays, identifies risk factors, and provides actionable recommendations.

---

## 📖 Table of Contents
- [Problem Statement](#-problem-statement)
- [Features](#-features)
- [How It Works](#-how-it-works)
- [Machine Learning Models](#-machine-learning-models)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Project Structure](#-project-structure)
- [Performance](#-performance)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Problem Statement
Construction and project‑based industries frequently suffer from delays, leading to budget overruns and stakeholder dissatisfaction. Traditional risk assessment is subjective and often reactive. **Project Delay AI** leverages machine learning to provide proactive, data‑driven insights for better planning and mitigation.

---

## ✨ Features
- **Data Upload:** Accepts CSV or Excel files with project attributes.
- **Sample Dataset:** Built‑in 50‑row sample to test the system.
- **Dual Prediction:**
  - **Binary Classification** – Delayed / On Time.
  - **Regression** – Estimated delay days.
- **Risk Level:** Low / Medium / High based on predicted delay.
- **Root Cause Analysis:** Highlights the most likely reasons for delay.
- **Interactive Visualizations:**
  - Risk distribution bar chart.
  - Delay days histogram.
  - Top causes horizontal bar chart.
- **Downloadable Results:** CSV of predictions and analysis.
- **PDF Report:** Comprehensive report with summary, charts, and recommendations.
- **What‑If Analysis:** Modify project parameters and instantly see updated predictions.
- **Made by Esrom** – fully open‑source.

---

## 🔧 How It Works
1. **Data Input:** User uploads a file or selects the sample dataset.
2. **Preprocessing:** Missing values are filled, categorical variables encoded, numeric variables scaled.
3. **Model Inference:** Pre‑trained Random Forest models (classifier and regressor) generate predictions.
4. **Output:** Results table, risk levels, causes, and visualizations.
5. **Reporting:** Option to generate a PDF report with deep analysis and suggestions.

---

## 🤖 Machine Learning Models
We use **two Random Forest models**:

| Model               | Target          | Purpose                                      | Evaluation Metric |
|---------------------|-----------------|----------------------------------------------|-------------------|
| **Classifier**      | `Status` (0/1)  | Determine if delay > 15 days                 | F1‑score: 0.88    |
| **Regressor**       | `Delay_Days`    | Predict exact number of delay days           | MAE: 2.4 days     |

- Both models are trained on a **synthetic dataset** of 500 projects generated with realistic interactions (weather, material shortages, manager experience, etc.).
- A pipeline with `ColumnTransformer` handles categorical and numeric features.
- Cross‑validation ensures generalizability.

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or later
- pip

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/project-delay-ai.git
   cd project-delay-ai
