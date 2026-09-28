"""
Model code shared by the Streamlit app (and mirrored in the Colab notebook).

ABC Ltd - Loan Default Risk Advisor
  * Logistic regression -> probability that a borrower defaults
  * Linear regression   -> typical loan amount for a borrower profile

Sex / marital status and nationality are deliberately left out of both
models so the tool cannot treat applicants differently on those grounds.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_HERE = Path(__file__).parent
PUBLIC_URL = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"
DATA_PATH = next(
    (p for p in [_HERE / "data" / "credit_data.csv", _HERE / "credit_data.csv"] if p.exists()),
    PUBLIC_URL,
)
RANDOM_STATE = 42

# ---------------------------------------------------------------- plain-English labels
RELABEL = {
    "status": {
        "no checking account": "No current account",
        "... < 100 DM": "Low / overdrawn balance",
        "0 <= ... < 200 DM": "Moderate balance",
        "... >= 200 DM / salary for at least 1 year": "High balance / salary account",
    },
    "credit_history": {
        "existing credits paid back duly till now": "Current loans paid on time",
        "critical account/other credits existing": "Has loans at other lenders",
        "delay in paying off in the past": "Delays in the past",
        "all credits at this bank paid back duly": "All loans with us repaid",
        "no credits taken/all credits paid back duly": "No loans / all repaid",
    },
    "savings": {
        "... < 100 DM": "Very low (< 100)",
        "unknown/no savings account": "None / unknown",
        "100 <= ... < 500 DM": "Low (100–500)",
        "500 <= ... < 1000 DM": "Medium (500–1,000)",
        "... >= 1000 DM": "High (1,000+)",
    },
    "employment_duration": {
        "unemployed": "Unemployed",
        "... < 1 year": "Less than 1 year",
        "1 <= ... < 4 years": "1–4 years",
        "4 <= ... < 7 years": "4–7 years",
        "... >= 7 years": "7+ years",
    },
    "other_debtors": {"none": "None", "guarantor": "Guarantor", "co-applicant": "Co-applicant"},
    "property": {
        "real estate": "Real estate",
        "building society savings agreement/life insurance": "Savings plan / life insurance",
        "car or other": "Vehicle or other",
        "unknown/no property": "No property",
    },
    "other_installment_plans": {"none": "None", "bank": "With another bank", "stores": "With stores"},
    "housing": {"own": "Own", "rent": "Rent", "for free": "Lives free (family)"},
    "job": {
        "unemployed/unskilled - non-resident": "Unemployed / unskilled (non-resident)",
        "unskilled - resident": "Unskilled",
        "skilled employee/official": "Skilled employee",
        "management/self-employed/highly qualified employee/officer": "Manager / self-employed / professional",
    },
    "purpose": {
        "car (new)": "New vehicle", "car (used)": "Used vehicle",
        "furniture/equipment": "Furniture / equipment", "radio/television": "Electronics",
        "domestic appliances": "Home appliances", "repairs": "Repairs", "education": "Education",
        "retraining": "Skill training", "business": "Business", "others": "Other",
    },
    "telephone": {"no": "No", "yes": "Yes"},
}

# Order of categories for dropdowns (low risk -> high risk where it makes sense)
ORDER = {
    "status": ["No current account", "High balance / salary account", "Moderate balance", "Low / overdrawn balance"],
    "savings": ["None / unknown", "Very low (< 100)", "Low (100–500)", "Medium (500–1,000)", "High (1,000+)"],
    "employment_duration": ["Unemployed", "Less than 1 year", "1–4 years", "4–7 years", "7+ years"],
}

# ---------------------------------------------------------------- features
DEF_NUM = ["duration", "amount", "installment_rate", "age", "number_credits",
           "present_residence", "people_liable"]
DEF_CAT = ["status", "credit_history", "purpose", "savings", "employment_duration",
           "other_debtors", "property", "other_installment_plans", "housing", "job", "telephone"]

AMT_NUM = ["duration", "installment_rate", "age", "number_credits"]
AMT_CAT = ["purpose", "job", "property", "housing", "savings", "telephone"]

LABELS = {
    "duration": "Loan tenure (months)", "amount": "Loan amount",
    "installment_rate": "EMI burden (% of income band)", "age": "Age",
    "number_credits": "Existing loans with us", "present_residence": "Years at current address",
    "people_liable": "Dependants", "status": "Current-account balance",
    "credit_history": "Repayment history", "purpose": "Loan purpose", "savings": "Savings",
    "employment_duration": "Time in current job", "other_debtors": "Guarantor / co-applicant",
    "property": "Collateral / property", "other_installment_plans": "Other EMIs running",
    "housing": "Housing", "job": "Job type", "telephone": "Registered phone",
}

ACTIONS = {
    "duration": "Offer a shorter tenure to reduce exposure time.",
    "amount": "Sanction a smaller amount, or split into tranches.",
    "installment_rate": "Restructure the EMI so it takes a smaller share of income.",
    "status": "Ask for 6 months of bank statements; consider salary-account linkage.",
    "credit_history": "Pull a fresh credit bureau report and verify past delays.",
    "savings": "Ask for a margin deposit or recurring-deposit linkage.",
    "employment_duration": "Verify employment; ask for a co-applicant with stable income.",
    "other_debtors": "Ask for a guarantor or co-applicant.",
    "property": "Ask for collateral or hypothecation of the asset.",
    "other_installment_plans": "Check total EMI load across lenders before sanction.",
    "housing": "Verify address and residence stability.",
    "job": "Verify income documents.",
    "purpose": "Check end-use of funds and ask for quotations/invoices.",
    "age": "-", "number_credits": "Review exposure across existing loans.",
    "present_residence": "-", "people_liable": "-", "telephone": "Verify contact details.",
}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, m in RELABEL.items():
        if col in df:
            df[col] = df[col].replace(m)
    if "credit_risk" in df and "default" not in df:
        df["default"] = 1 - df["credit_risk"]          # 1 = borrower defaulted
    return df


def load_data(path=DATA_PATH) -> pd.DataFrame:
    return clean(pd.read_csv(path))


def _pre(num, cat):
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat),
    ])


def train_models(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df = load_data()

    # ---- Logistic regression: default
    X, y = df[DEF_NUM + DEF_CAT], df["default"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    logit = Pipeline([("prep", _pre(DEF_NUM, DEF_CAT)),
                      ("model", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.3))]).fit(Xtr, ytr)
    p = logit.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    logit_metrics = {
        "accuracy": accuracy_score(yte, pred), "precision": precision_score(yte, pred),
        "recall": recall_score(yte, pred), "f1": f1_score(yte, pred),
        "roc_auc": roc_auc_score(yte, p),
        "confusion_matrix": confusion_matrix(yte, pred).tolist(), "n_test": int(len(yte)),
    }

    # ---- Linear regression: typical loan amount
    Xa, ya = df[AMT_NUM + AMT_CAT], df["amount"]
    Xatr, Xate, yatr, yate = train_test_split(Xa, ya, test_size=0.25, random_state=RANDOM_STATE)
    linreg = Pipeline([("prep", _pre(AMT_NUM, AMT_CAT)), ("model", LinearRegression())]).fit(Xatr, yatr)
    pa = linreg.predict(Xate)
    lin_metrics = {"r2": r2_score(yate, pa), "mae": mean_absolute_error(yate, pa),
                   "rmse": float(np.sqrt(np.mean((yate - pa) ** 2))), "n_test": int(len(yate))}

    baseline = np.asarray(logit.named_steps["prep"].transform(Xtr).mean(axis=0)).ravel()
    return {"logit": logit, "linreg": linreg, "logit_metrics": logit_metrics,
            "lin_metrics": lin_metrics, "baseline": baseline, "base_rate": float(y.mean())}


def risk_band(p: float) -> str:
    if p < 0.35:
        return "Low"
    if p < 0.60:
        return "Medium"
    return "High"


DECISION = {
    "Low": "Approve — standard terms",
    "Medium": "Review — approve with conditions",
    "High": "Refer to credit committee / ask for mitigation",
}


def explain(models: dict, row: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """Contribution of each input to the log-odds of default versus an average borrower."""
    logit = models["logit"]
    prep = logit.named_steps["prep"]
    coefs = logit.named_steps["model"].coef_.ravel()
    x = np.asarray(prep.transform(row[DEF_NUM + DEF_CAT])).ravel()
    contrib = coefs * (x - models["baseline"])
    agg = {}
    for n, c in zip(prep.get_feature_names_out(), contrib):
        base = n.split("__", 1)[1]
        orig = next((f for f in DEF_CAT if base.startswith(f + "_")), base)
        agg[orig] = agg.get(orig, 0.0) + c
    out = (pd.DataFrame({"feature": list(agg), "impact": list(agg.values())})
           .assign(a=lambda d: d.impact.abs()).sort_values("a", ascending=False).head(top_n))
    out["factor"] = out.feature.map(LABELS)
    out["direction"] = np.where(out.impact > 0, "Raises risk", "Lowers risk")
    out["value"] = [row.iloc[0][f] for f in out.feature]
    out["suggested_action"] = [ACTIONS.get(f, "-") if i > 0 else "-" for f, i in zip(out.feature, out.impact)]
    return out.drop(columns="a").reset_index(drop=True)


def coefficient_table(models: dict) -> pd.DataFrame:
    logit = models["logit"]
    names = logit.named_steps["prep"].get_feature_names_out()
    df = pd.DataFrame({"term": [n.split("__", 1)[1] for n in names],
                       "coef": logit.named_steps["model"].coef_.ravel()})
    df["odds_ratio"] = np.exp(df.coef)
    return df.sort_values("coef", ascending=False).reset_index(drop=True)
