# ABC Ltd — Loan Default Risk Advisor

**Business Analytics · Assignment 2: Predictive Analytics & Managerial AI Adoption**
Institute of Rural Management Anand

A predictive tool that helps credit officers at ABC Ltd see **how likely a loan applicant is to default, why, and what terms would make the loan safer**.

| | |
|---|---|
| **Business problem** | Loan default (30% of past loans defaulted) |
| **Model 1: Linear regression** | Typical loan amount for a borrower profile (R² ≈ 0.53) |
| **Model 2: Logistic regression** | Probability of default (AUC ≈ 0.81, catches about 77% of defaulters) |
| **App** | Streamlit, for non-technical credit managers |
| **User study** | 36-respondent credit-manager survey, analysed quantitatively and qualitatively |

> **Data:** Statlog German Credit data (1,000 loans), from the UCI Machine Learning Repository, also on Kaggle. The lender is called ABC Ltd throughout.

## Repository structure
```
abc-loan-risk-advisor/
├── app.py                     ← Streamlit app (entry point)
├── model_utils.py             ← model training + explanations used by the app
├── requirements.txt
├── data/credit_data.csv
├── models/                    ← models saved from Colab (evidence)
├── notebooks/
│   ├── 01_ABC_Loan_Default_Models.ipynb          ← EDA + linear & logistic regression
│   └── 02_Credit_Manager_Survey_Analysis.ipynb   ← user-study analysis
└── survey/
    ├── questionnaire.md          ← questions for the Google Form
    ├── survey_responses.csv      ← responses (currently SYNTHETIC placeholder)
    ├── generate_synthetic_survey.py
    └── survey_analysis.xlsx      ← analysis tables
```

## Steps
1. **Colab:** upload `notebooks/01_ABC_Loan_Default_Models.ipynb` and click **Runtime → Run all**. The data loads from a public URL.
2. **GitHub:** create a **public** repo called `abc-loan-risk-advisor`. Upload the files, then upload the folders `data`, `models`, `notebooks` and `survey` by dragging each *folder* (not its contents). Commit.
3. **Streamlit:** go to share.streamlit.io, then **Create app**. Choose the repo, branch `main`, and main file `app.py`, then click **Deploy**. Copy the URL; it is the submission link.
4. **Survey:** build a Google Form from `survey/questionnaire.md` and paste its link into `FEEDBACK_URL` at the top of `app.py`.
5. **Survey analysis:** open `notebooks/02_Credit_Manager_Survey_Analysis.ipynb` in Colab. Upload `survey/survey_responses.csv` to the Files panel, then click **Run all**.

The app's cover (names, roll numbers, and "Submitted to") is set at the top of `app.py` and can be edited on GitHub with the ✏️ pencil icon.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
