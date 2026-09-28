"""
ABC Ltd - Loan Default Risk Advisor
Streamlit app for non-technical credit managers and branch officers.

Run locally:   streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model_utils import (AMT_CAT, AMT_NUM, DECISION, DEF_CAT, DEF_NUM, ORDER,
                         coefficient_table, explain, load_data, risk_band,
                         train_models)

# Paste your Google Form link here after you create the survey
FEEDBACK_URL = "https://forms.gle/your-form-id"

# ---- Cover details (edit here if anything changes) ----
UNIVERSITY = "“Tribhuvan” Sahkari University"
INSTITUTE = "Institute of Rural Management Anand"
COURSE = "Business Analytics"
ASSIGNMENT = "Assignment 2: Predictive Analytics & Managerial AI Adoption"
SUBMITTED_TO = ""          # e.g. "Prof. Janak Suthar" — leave "" to hide the line
PREPARED_BY = [  # (Name, Roll No.)
    ("Ankit Yadav", "abm01003"),
    ("Animesh Sharma", "p46153"),
    ("Ishita Jain", "p46245"),
    ("Kanika Chaudhary", "p46247"),
    ("Prakhar Ganvir", "p46116"),
    ("Palash Dubey", "p46041"),
    ("Veeresh", "cbf01014"),
]

st.set_page_config(page_title="ABC Ltd Loan Risk Advisor", page_icon="🏦", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_models():
    return train_models(get_data())


df = get_data()
models = get_models()
logit, linreg = models["logit"], models["linreg"]
BAND_COLOR = {"Low": "#2e7d32", "Medium": "#ef8f00", "High": "#c62828"}


def score(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["Default risk (%)"] = (logit.predict_proba(out[DEF_NUM + DEF_CAT])[:, 1] * 100).round(1)
    out["Risk band"] = out["Default risk (%)"].apply(lambda p: risk_band(p / 100))
    out["Suggested decision"] = out["Risk band"].map(DECISION)
    out["Typical amount"] = linreg.predict(out[AMT_NUM + AMT_CAT]).clip(250).round(0)
    out["Amount vs typical (%)"] = ((out["amount"] - out["Typical amount"]) / out["Typical amount"] * 100).round(0)
    return out


def gauge(p: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100,
        number={"suffix": "%", "font": {"size": 44}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#263238"},
               "steps": [{"range": [0, 35], "color": "#c8e6c9"},
                         {"range": [35, 60], "color": "#ffe0b2"},
                         {"range": [60, 100], "color": "#ffcdd2"}],
               "threshold": {"line": {"color": "#1565c0", "width": 3},
                             "value": models["base_rate"] * 100}}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=0))
    return fig


def opts(col):
    vals = list(df[col].unique())
    return ORDER.get(col, sorted(vals))


# ---------------------------------------------------------------- cover
members = "".join(f"<div>{n} [{r}]</div>" for n, r in PREPARED_BY)
sub_to = f'<div class="sub"><b>Submitted to:</b> {SUBMITTED_TO}</div>' if SUBMITTED_TO else ""
st.markdown(f"""
<style>
.cover {{text-align:center;padding:18px 16px 12px;border:1px solid rgba(128,128,128,.3);
         border-radius:12px;margin-bottom:18px}}
.cover .uni {{font-size:1.3rem}}
.cover .inst {{font-size:1.2rem;margin-top:2px}}
.cover .course {{font-size:1.15rem;font-weight:700;margin-top:12px}}
.cover .asg {{font-size:1rem;font-weight:700;margin-bottom:10px}}
.cover .sub {{font-size:1rem;margin:2px 0}}
.cover .prep {{font-weight:700;margin-top:10px}}
.cover .names {{font-size:.95rem;line-height:1.5}}
</style>
<div class="cover">
  <div class="uni">{UNIVERSITY}</div>
  <div class="inst">{INSTITUTE}</div>
  <div class="course">{COURSE}</div>
  <div class="asg">{ASSIGNMENT}</div>
  {sub_to}
  <div class="prep">Prepared by:</div>
  <div class="names">{members}</div>
</div>
""", unsafe_allow_html=True)

st.title("🏦 ABC Ltd — Loan Default Risk Advisor")
st.caption(
    "Estimates how likely a loan applicant is to default, explains why, suggests how to reduce the risk, "
    "and checks whether the amount asked for is in line with similar borrowers. "
    "It supports the credit officer's judgement — it does not replace it.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["👤 Assess an applicant", "📋 Score a loan batch", "🔍 How the tool works", "📝 Give feedback"])

# ================================================================ TAB 1
with tab1:
    with st.form("applicant"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**The loan**")
            amount = st.number_input("Loan amount requested", 250, 20000, 3000, step=250)
            duration = st.slider("Tenure (months)", 4, 72, 24)
            purpose = st.selectbox("Purpose", opts("purpose"), index=opts("purpose").index("New vehicle"))
            inst = st.select_slider("EMI burden (1 = light, 4 = heavy share of income)", [1, 2, 3, 4], value=3)
            debtors = st.selectbox("Guarantor / co-applicant", ["None", "Guarantor", "Co-applicant"])
            prop = st.selectbox("Collateral / property", opts("property"), index=opts("property").index("Vehicle or other"))
        with c2:
            st.markdown("**The applicant**")
            age = st.slider("Age", 19, 75, 33)
            job = st.selectbox("Job type", opts("job"), index=opts("job").index("Skilled employee"))
            emp = st.selectbox("Time in current job", opts("employment_duration"), index=2)
            housing = st.selectbox("Housing", opts("housing"), index=opts("housing").index("Own"))
            resid = st.select_slider("Years at current address (1 = <1 yr, 4 = 7+ yrs)", [1, 2, 3, 4], value=3)
            liable = st.radio("Dependants", [1, 2], horizontal=True, format_func=lambda v: "0–2" if v == 1 else "3 or more")
            phone = st.radio("Registered phone", ["Yes", "No"], horizontal=True)
        with c3:
            st.markdown("**Money & credit record**")
            status = st.selectbox("Current-account balance", opts("status"), index=2)
            savings = st.selectbox("Savings", opts("savings"), index=2)
            history = st.selectbox("Repayment history", opts("credit_history"),
                                   index=opts("credit_history").index("Current loans paid on time"))
            other = st.selectbox("Other EMIs running", ["None", "With another bank", "With stores"])
            ncred = st.select_slider("Existing loans with ABC Ltd", [1, 2, 3, 4], value=1)
        st.form_submit_button("Assess risk", type="primary", width="stretch")

    app_row = pd.DataFrame([{
        "duration": duration, "amount": amount, "installment_rate": inst, "age": age,
        "number_credits": ncred, "present_residence": resid, "people_liable": liable,
        "status": status, "credit_history": history, "purpose": purpose, "savings": savings,
        "employment_duration": emp, "other_debtors": debtors, "property": prop,
        "other_installment_plans": other, "housing": housing, "job": job, "telephone": phone,
    }])
    s = score(app_row).iloc[0]
    p = s["Default risk (%)"] / 100
    band = s["Risk band"]

    st.divider()
    r1, r2, r3 = st.columns([1.1, 1, 1])
    with r1:
        st.subheader("Chance of default")
        st.plotly_chart(gauge(p), width="stretch")
        st.markdown(f"<div style='text-align:center;font-size:1.15rem'>Risk level: "
                    f"<b style='color:{BAND_COLOR[band]}'>{band}</b><br>{DECISION[band]}</div>",
                    unsafe_allow_html=True)
        st.caption(f"Blue line = ABC Ltd average ({models['base_rate']*100:.0f}% of past loans defaulted).")
    with r2:
        st.subheader("Amount check")
        st.metric("Typical amount for this profile", f"{s['Typical amount']:,.0f}")
        st.metric("Amount requested", f"{amount:,.0f}", delta=f"{s['Amount vs typical (%)']:+.0f}% vs typical",
                  delta_color="inverse")
        if s["Amount vs typical (%)"] > 50:
            st.warning("Much larger than similar borrowers usually take — check need and repayment capacity.")
        else:
            st.success("In the usual range for similar borrowers.")
        st.caption("Typical amount comes from the linear regression model "
                   "(tenure, purpose, job, collateral, savings, age).")
    with r3:
        st.subheader("What if we change the terms?")
        st.caption("See how the risk moves with common mitigations.")
        w_dur = st.slider("Tenure (months)", 4, 72, duration, key="w_dur")
        w_amt = st.slider("Amount", 250, 20000, int(amount), step=250, key="w_amt")
        w_guar = st.checkbox("Add a guarantor", value=False, disabled=debtors != "None")
        w_prop = st.checkbox("Take real-estate collateral", value=False, disabled=prop == "Real estate")
        what = app_row.copy()
        what["duration"], what["amount"] = w_dur, w_amt
        if w_guar and debtors == "None":
            what["other_debtors"] = "Guarantor"
        if w_prop:
            what["property"] = "Real estate"
        p2 = logit.predict_proba(what[DEF_NUM + DEF_CAT])[0, 1]
        st.metric("Risk with new terms", f"{p2*100:.1f}%", delta=f"{(p2-p)*100:+.1f} pts", delta_color="inverse")
        st.caption(f"Band: **{risk_band(p2)}** — {DECISION[risk_band(p2)]}")

    st.subheader("Why the tool thinks so")
    st.caption("The factors that move this applicant's risk the most, compared with an average ABC Ltd borrower.")
    for _, r in explain(models, app_row).iterrows():
        icon = "🔺" if r.impact > 0 else "🔻"
        act = f" — *{r.suggested_action}*" if r.suggested_action != "-" else ""
        st.markdown(f"{icon} **{r.factor}** ({r.value}) · {r.direction.lower()}{act}")

    lm = models["logit_metrics"]
    st.info(f"**Use this as one input.** On past loans the tool caught about "
            f"{lm['recall']*100:.0f}% of defaulters, but roughly half of the loans it flags still get repaid. "
            "Verify documents and talk to the applicant before deciding.")

# ================================================================ TAB 2
with tab2:
    st.subheader("Rank a batch of applications by default risk")
    st.caption("Upload a CSV with the same columns as the template, or try a sample of past ABC Ltd loans.")
    st.download_button("Download CSV template", df[DEF_NUM + DEF_CAT].head(3).to_csv(index=False),
                       "loan_template.csv", "text/csv")
    up = st.file_uploader("Upload applications CSV", type="csv")
    use_sample = st.toggle("Use a sample of 40 past ABC Ltd loans", value=up is None)

    batch = None
    if up is not None:
        batch = pd.read_csv(up)
        missing = set(DEF_NUM + DEF_CAT) - set(batch.columns)
        if missing:
            st.error(f"Missing columns: {', '.join(sorted(missing))}")
            batch = None
    elif use_sample:
        batch = df.sample(40, random_state=11).reset_index(drop=True)
        batch.insert(0, "Application", [f"LN-{2400+i}" for i in range(len(batch))])

    if batch is not None:
        sc = score(batch).sort_values("Default risk (%)", ascending=False)
        a, b, c = st.columns(3)
        a.metric("Applications", len(sc))
        b.metric("High risk", int((sc["Risk band"] == "High").sum()))
        c.metric("Exposure in high-risk loans", f"{sc.loc[sc['Risk band']=='High','amount'].sum():,.0f}")
        show = [x for x in ["Application"] if x in sc] + [
            "purpose", "amount", "duration", "Typical amount", "Default risk (%)", "Risk band", "Suggested decision"]
        st.dataframe(sc[show].rename(columns={"purpose": "Purpose", "amount": "Amount", "duration": "Tenure"})
                     .style.map(lambda v: f"color:{BAND_COLOR.get(v,'inherit')};font-weight:600", subset=["Risk band"]),
                     width="stretch", hide_index=True)
        st.download_button("Download results", sc.to_csv(index=False), "loan_risk_scores.csv", "text/csv")

# ================================================================ TAB 3
with tab3:
    lm, im = models["logit_metrics"], models["lin_metrics"]
    st.subheader("Two simple, explainable models")
    st.markdown(
        "- **Logistic regression** estimates the chance that a borrower defaults (yes/no outcome).\n"
        "- **Linear regression** estimates the loan amount similar borrowers typically take, "
        "so unusually large requests stand out.\n\n"
        "Both were trained on 1,000 past ABC Ltd loans; 25% were held back to test them. "
        "Sex, marital status and nationality are deliberately excluded so the tool cannot treat "
        "applicants differently on those grounds.")
    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f"{lm['accuracy']*100:.0f}%")
    b.metric("Defaulters caught (recall)", f"{lm['recall']*100:.0f}%")
    c.metric("Ranking quality (AUC)", f"{lm['roc_auc']:.2f}")
    d.metric("Amount model fit (R²)", f"{im['r2']:.2f}")
    tn, fp, fn, tp = np.array(lm["confusion_matrix"]).ravel()
    st.markdown(
        f"On {lm['n_test']} test loans the tool flagged {tp+fp} as likely to default: **{tp} actually defaulted** "
        f"and {fp} were repaid (false alarms). It missed {fn} defaulters. "
        f"The amount model is off by about **{im['mae']:,.0f}** on average.")
    st.subheader("What drives default at ABC Ltd")
    st.caption("Odds ratio > 1 raises the chance of default; < 1 lowers it. Numeric factors are per one standard deviation.")
    ct = coefficient_table(models)
    ct = pd.concat([ct.head(8), ct.tail(8)])
    ct["label"] = ct.term.str.replace("_", ": ", n=1)
    fig = go.Figure(go.Bar(x=ct.odds_ratio, y=ct.label, orientation="h",
                           marker_color=np.where(ct.odds_ratio > 1, "#c62828", "#2e7d32")))
    fig.add_vline(x=1, line_dash="dash")
    fig.update_layout(height=520, yaxis=dict(autorange="reversed"), xaxis_title="Odds ratio",
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.subheader("Limits")
    st.markdown(
        "- Learns from past approved loans only; it has never seen applicants who were rejected.\n"
        "- Some patterns reflect how the bank lent in the past (e.g. applicants with loans elsewhere were "
        "screened harder, so they look safer). Treat odd-looking drivers with care.\n"
        "- It cannot see income documents, bureau scores or local knowledge of the borrower.\n"
        "- A 'High' score is a prompt for closer checks, not an automatic rejection.")

# ================================================================ TAB 4
with tab4:
    st.subheader("Help us improve this tool")
    st.markdown("We are studying how credit managers decide whether to trust AI predictions. "
                "After trying the tool, please fill in a short survey (about 8 minutes).")
    st.link_button("Open the feedback survey", FEEDBACK_URL, type="primary")
    st.caption("Responses are anonymous and used only for an academic study at IRMA.")
