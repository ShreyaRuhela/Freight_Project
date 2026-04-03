# Freight_Project
Freight Cost Prediction
<img width="931" height="380" alt="Screenshot 2026-04-03 230213" src="https://github.com/user-attachments/assets/d4356555-90de-43f9-b3ac-00315b72f9fb" />

Invoive Manual Approval Prediction
<img width="888" height="427" alt="Screenshot 2026-04-03 230926" src="https://github.com/user-attachments/assets/bb617407-f9d2-4770-8486-9651b719d3b6" />
<img width="728" height="359" alt="Screenshot 2026-04-03 231034" src="https://github.com/user-attachments/assets/99a2489c-fe75-45ee-b083-b851641c04b6" />
<img width="721" height="370" alt="Screenshot 2026-04-03 230935" src="https://github.com/user-attachments/assets/035676ac-3038-404d-ae2d-558b08d7573d" />




# 📦 Vendor Invoice Intelligence & Freight Prediction Portal

An end-to-end Machine Learning solution designed to automate financial auditing processes. This portal leverages predictive analytics to forecast freight costs and classify invoice risks, reducing financial leakage and manual workload.

## 🚀 Key Features
- **Freight Cost Prediction (Regression):** Predicts estimated shipping costs based on quantity and invoice totals using Linear Regression.
- **Invoice Risk Flagging (Classification):** Detects discrepancies between Invoices and Purchase Orders (PO) using a **Random Forest Classifier**.
- **Interactive Dashboard:** A multi-page **Streamlit** web application for real-time inference and data visualization.
- **Automated Data Pipeline:** Integrated **SQL (SQLite)** backend for aggregating PO and invoice records.

## 🛠️ Tech Stack
- **Languages:** Python, SQL
- **ML Libraries:** Scikit-Learn, Pandas, NumPy, Joblib
- **Visualization:** Plotly, Streamlit
- **Environment:** VS Code, Git

## 📂 Project Structure
```text
Freight_Project/
├── app.py                # Main Streamlit application
├── models/               # Saved .pkl models and scalers
│   ├── predict_freight_model.pkl
│   ├── predict_flag_invoice.pkl
│   └── scaler.pkl
├── inference/            # Modular inference logic
│   ├── __init__.py
│   ├── predict_freight.py
│   └── predict_invoice_flag.py
└── data/                 # Database and preprocessing scripts
    └── database.sqlite

⚙️ Installation & Usage

Clone the repository:
git clone [https://github.com/ShreyaRuhela/Freight-Intelligence-Portal.git](https://github.com/ShreyaRuhela/Freight-Intelligence-Portal.git)
cd Freight-Intelligence-Portal

Install dependencies:
pip install streamlit pandas scikit-learn plotly joblib

Run the application:
python -m streamlit run app.py
