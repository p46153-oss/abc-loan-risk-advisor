"""
Generates the SYNTHETIC credit-manager survey used as placeholder data for the
qualitative study. Replace survey_responses.csv with real Google Form
responses (same column names) and re-run the analysis notebook.

python survey/generate_synthetic_survey.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(4153)
OUT = Path(__file__).parent / "survey_responses.csv"

# ------------------------------------------------------------ profile
roles = (["Branch Manager"] * 8 + ["Credit / Loan Officer"] * 9 + ["Relationship Manager"] * 5 +
         ["Credit / Risk Head"] * 3 + ["Cooperative / Credit Society Manager"] * 5 +
         ["Entrepreneur (MFI / NBFC / business)"] * 3 + ["Working Professional"] * 3)
rng.shuffle(roles)
N = len(roles)
exp_base = {"Branch Manager": (14, 4), "Credit / Loan Officer": (6, 2.5), "Relationship Manager": (5, 2),
            "Credit / Risk Head": (19, 4), "Cooperative / Credit Society Manager": (15, 5),
            "Entrepreneur (MFI / NBFC / business)": (10, 4), "Working Professional": (4, 2)}
orgs_of = {
    "Branch Manager": ["Public sector bank", "Public sector bank", "Private bank", "Cooperative bank / credit society"],
    "Credit / Loan Officer": ["Public sector bank", "Private bank", "NBFC", "Microfinance"],
    "Relationship Manager": ["Private bank", "Private bank", "NBFC", "Fintech"],
    "Credit / Risk Head": ["Private bank", "NBFC", "Public sector bank"],
    "Cooperative / Credit Society Manager": ["Cooperative bank / credit society"],
    "Entrepreneur (MFI / NBFC / business)": ["Microfinance", "NBFC", "Fintech"],
    "Working Professional": ["Fintech", "Private bank", "Other"],
}
volume_of = {
    "Branch Manager": ["51–200", "51–200", "200+"], "Credit / Loan Officer": ["11–50", "51–200", "51–200"],
    "Relationship Manager": ["1–10", "11–50"], "Credit / Risk Head": ["200+"],
    "Cooperative / Credit Society Manager": ["11–50", "51–200"],
    "Entrepreneur (MFI / NBFC / business)": ["51–200", "200+"], "Working Professional": ["None", "1–10"],
}
ai_freq = ["Never", "Rarely", "Monthly", "Weekly", "Daily"]

rows = []
for i, role in enumerate(roles):
    mu, sd = exp_base[role]
    exp = int(np.clip(round(rng.normal(mu, sd)), 2, 32))
    org = rng.choice(orgs_of[role])
    tech = rng.normal(0, 1)
    tech += {"Fintech": .6, "Private bank": .3, "NBFC": .2, "Cooperative bank / credit society": -.5,
             "Public sector bank": -.2}.get(org, 0)
    if exp > 15:
        tech -= 0.3
    freq = ai_freq[int(np.clip(round(2 + tech * 1.1 + rng.normal(0, .5)), 0, 4))]

    PU = 3.8 + 0.35 * tech + rng.normal(0, .45)
    EU = 4.0 + 0.3 * tech + rng.normal(0, .5)
    TR = 3.0 + 0.4 * tech - 0.03 * (exp - 10) + rng.normal(0, .65)
    EX = 4.4 + 0.1 * tech + rng.normal(0, .5)
    FR = 3.7 + 0.25 * (role in ("Credit / Loan Officer", "Branch Manager")) + rng.normal(0, .6)
    IN = 3.2 + 0.05 * (exp - 10) - 0.25 * tech + 0.4 * (org == "Cooperative bank / credit society") + rng.normal(0, .5)
    OS = 3.0 + 0.4 * (org in ("Fintech", "Private bank")) - 0.5 * (org == "Cooperative bank / credit society") + rng.normal(0, .6)
    BI = 0.45 * PU + 0.35 * TR + 0.2 * OS - 0.2 * IN + 0.9 + rng.normal(0, .35)

    item = lambda base, sd=.38: int(np.clip(round(base + rng.normal(0, sd)), 1, 5))
    r = {
        "RespondentID": f"R{i+1:02d}", "Role": role, "ExperienceYears": exp,
        "Organisation": org, "LoanDecisionsPerMonth": rng.choice(volume_of[role]),
        "AIUseFrequency": freq,
        "ReviewMode": "Used it myself" if rng.random() < .65 else "Watched a demo",
        "PU1": item(PU + .2), "PU2": item(PU - .35), "PU3": item(PU),
        "EU1": item(EU), "EU2": item(EU + .15),
        "TR1": item(TR), "TR2": item(TR - .25), "TR3": item(TR),
        "EX1": item(EX + .1), "EX2": item(EX - .1),
        "FR1": item(FR + .15), "FR2": item(FR - .15),
        "IN1": item(IN), "IN2": item(IN + .1),
        "OS1": item(OS - .3), "OS2": item(OS + .1),
        "BI1": item(BI), "BI2": item(BI - .1),
    }
    logits = np.array([0.6 * (TR - 3) - 0.8, 0.8 * (IN - 3) - 0.6, 1.0, 0.7 + .3 * (FR - 3)])
    pr = np.exp(logits) / np.exp(logits).sum()
    r["C1_Scenario"] = rng.choice([
        "Follow the tool: reject or refer the application",
        "Follow my judgement: approve on normal terms",
        "Approve with conditions (guarantor, collateral, shorter tenure)",
        "Verify more first (statements, bureau report, field visit)"], p=pr)
    r["_tier"] = "pos" if BI >= 3.9 else ("neu" if BI >= 3.2 else "neg")
    rows.append(r)

df = pd.DataFrame(rows)

# ------------------------------------------------------------ open-ended text
T = {
 "Q1": {
  "pos": ["Very useful for screening. We get many small-ticket applications and can't check every file in depth.",
          "Useful as a first filter before the file goes to the sanctioning authority.",
          "It gives new loan officers a structured way to look at risk instead of guessing.",
          "The batch ranking is useful for our monthly portfolio review.",
          "Useful because it shows how to reduce risk, not just whether to reject.",
          "Helpful for branches without a dedicated credit analyst.",
          "Very useful in peak season when sanction targets push us to move fast.",
          "It puts savings, tenure and collateral on one screen, which saves time."],
  "neu": ["Moderately useful. It helps, but the bureau score already tells us a lot.",
          "Useful for retail loans, less so for agri or MSME loans where cash flows are seasonal.",
          "Somewhat useful. I liked the what-if part more than the score itself.",
          "Useful for credit heads reviewing portfolios; branch officers may find it extra work.",
          "It depends on how often the model is updated with our own repayment data.",
          "Useful as a second opinion on borderline files."],
  "neg": ["Limited. In rural lending, character and local reputation matter more than these variables.",
          "Not very useful for cooperative societies — our members are known to us personally.",
          "The data it uses is not what we collect, so we can't feed it properly.",
          "We already have a bank scorecard; I don't see what this adds."]},
 "Q2": {
  "pos": ["Yes, at the pre-sanction stage, to decide which files need extra documents.",
          "Yes, especially to decide conditions like a guarantor or shorter tenure.",
          "Yes, if it is linked to our loan origination system so no double entry.",
          "Yes. It would help me justify conditions to the customer with clear reasons.",
          "Yes, for training new officers on what drives default.",
          "I would, for personal loans and vehicle loans where volumes are high."],
  "neu": ["Maybe, as a second opinion, but not as the basis for rejecting anyone.",
          "Possibly, if the credit policy allows it and audit accepts it.",
          "Only if it is validated on our own portfolio first.",
          "I'd use the what-if feature to structure loans; not sure about the score.",
          "Perhaps for portfolio monitoring rather than individual sanction.",
          "If head office mandates it, yes. Otherwise officers won't adopt it."],
  "neg": ["Probably not. The RBI and internal audit expect decisions we can defend line by line.",
          "Not yet. It was trained on a different market and different borrowers.",
          "No, I'm worried it will reject genuine borrowers from poorer backgrounds.",
          "Not in its current form; it misses income, bureau score and bank statement data."]},
 "Q3": {
  "pos": ["The suggested actions — guarantor, collateral, shorter tenure — are practical.",
          "The what-if panel. Seeing risk fall with a shorter tenure is very convincing.",
          "The three bands with a suggested decision are easy to act on.",
          "It shows the reasons behind every score.",
          "Simple screens; no jargon.",
          "The batch view with exposure in high-risk loans."],
  "neu": ["It is honest about its error rate.",
          "Good that sex and nationality are excluded.",
          "The gauge with the average line gives context.",
          "Clean and quick to use.",
          "The drivers list matches what we see in practice for savings and tenure.",
          "The template download for batch scoring."],
  "neg": ["The interface is simple, I'll give it that.",
          "The what-if feature is the only part I found directly useful.",
          "It is easy to use.",
          "The design is clean."]},
 "Q4": {
  "pos": ["No field for monthly income or bureau score — those are the first things we check.",
          "I'd like an EMI-to-income calculation built in.",
          "Needs a confidence range, not just one number.",
          "Should allow uploading bank statements directly.",
          "A mobile version for field officers would help."],
  "neu": ["Too many false alarms — nearly half of flagged loans were actually repaid.",
          "The amount check didn't seem to predict default, so I'm not sure what to do with it.",
          "Some drivers look odd, like loans at other lenders lowering risk.",
          "No agri or seasonal cash-flow inputs.",
          "Currency units are generic; should match our loan products.",
          "No option to record the final decision and outcome for learning."],
  "neg": ["It could lead officers to reject people mechanically.",
          "It misses about a quarter of defaulters, which is too many for us.",
          "Doesn't say who is accountable if the model is wrong.",
          "Built on foreign data — our borrowers are different."]},
 "Q5": {
  "pos": ["Mostly, because the main drivers — savings, tenure, collateral — match my experience.",
          "Yes as a direction. The reasons make sense for most files I tested.",
          "Fairly, since it was tested on past loans and the accuracy is shown openly.",
          "Yes for ranking applicants; less for the exact percentage."],
  "neu": ["Partly. I trust the ranking more than the percentage.",
          "I trust it for patterns across the portfolio, not for one borrower.",
          "Somewhat. The false alarms bother me — I would lose good customers.",
          "Depends on data quality. Self-declared information can be wrong.",
          "Half-half. A few drivers look like quirks of old lending policy."],
  "neg": ["Not fully. It hasn't been tested on our own customers.",
          "No. Default depends on events like crop failure or illness that no model sees.",
          "Not yet; it only learned from loans that were approved, so it has a blind spot.",
          "Not really; missing income data is a big gap."]},
 "Q6": {
  "pos": ["Back-testing on our own portfolio for at least one year.",
          "Showing its past predictions next to what actually happened.",
          "Integration with the bureau score and bank statement analysis.",
          "Regular updates with our latest repayment data."],
  "neu": ["Approval from the credit policy committee and audit.",
          "A clear note on what the model can't see.",
          "Being able to override it with a recorded reason.",
          "An independent validation for bias against weaker sections.",
          "A pilot in one region with results shared openly.",
          "Other branch managers vouching for it after using it."],
  "neg": ["Testing it on our members first.",
          "Clarity on who is responsible if a flagged loan is approved and fails.",
          "Local variables like crop cycle and group guarantees.",
          "Transparency on the data used and privacy of customer information."]},
 "Q7": {
  "pos": ["I'd verify more — bank statements, bureau report, maybe a field visit — before deciding.",
          "Approve with conditions. The tool tells me where to be careful; I decide how.",
          "Check which driver is raising the score and verify that specific point.",
          "Use the what-if panel to find terms that bring the risk down."],
  "neu": ["Go with my judgement but add a guarantor to be safe.",
          "Refer it to the sanctioning authority with both views noted.",
          "Probably approve, but with a shorter tenure.",
          "Discuss with the credit head before sanction.",
          "Ask for a margin deposit and then approve.",
          "Do a field visit and meet the family again."],
  "neg": ["My judgement. I know my borrowers personally.",
          "Experience first — the officer is accountable, not the software.",
          "I'd approve if I know the family. Local knowledge beats data here."]},
 "Q8": ["Officers are accountable for the loan, so they feel safer relying on their own assessment.",
        "Local knowledge — family, reputation, crop situation — isn't in the data.",
        "Habit. Most senior bankers learned credit appraisal by experience.",
        "The model feels like a black box, and audit asks for reasons.",
        "Past scorecards gave wrong signals, so people are sceptical.",
        "Relationship lending depends on trust built over years.",
        "Intuition is quick; the tool needs data entry.",
        "Admitting a model knows better can feel like a threat to expertise.",
        "In cooperatives, members expect a human decision.",
        "Sanction targets reward speed, and intuition is faster.",
        "They remember cases where the scorecard rejected someone who repaid well.",
        "Hierarchy — seniors' judgement carries more weight than a score."],
 "Q9": ["Yes, a lot. I have to write a credit note explaining every sanction.",
        "Definitely. Without reasons, audit and the committee won't accept it.",
        "Yes — if I can see why, I can verify that point myself.",
        "Yes, explanations matter more to me than the accuracy number.",
        "Somewhat. The reasons need to be correct, not just present.",
        "Yes. A black-box rejection would also be unfair to the customer.",
        "It matters because customers ask why they got stricter terms.",
        "Yes, but keep it short — three reasons are enough.",
        "Not much for me; I care whether it has been right on our loans.",
        "Yes, regulators increasingly expect explainable credit decisions."],
 "Q10": ["Yes. If a loan goes bad, the officer faces staff accountability, not the tool.",
         "Yes, vigilance and audit fear make officers very cautious.",
         "A little. But ignoring a warning that turns out right is also a risk.",
         "Yes, which is why I'd use it to decide conditions, not to reject.",
         "Not much — using a documented tool actually protects me in an audit.",
         "Yes. In banks, one bad loan is remembered more than a hundred good ones.",
         "It does; officers prefer to be wrong with their own judgement.",
         "Somewhat. A clear policy on how to use the tool would reduce the fear.",
         "No. A structured tool reduces my personal risk.",
         "Yes, especially after NPA reviews where every sanction is questioned."],
 "Q11": ["Incomplete or outdated customer data in core banking systems.",
         "No training on analytics for branch staff.",
         "Senior management and audit are not comfortable with AI decisions.",
         "Customer privacy and data protection concerns.",
         "Workload — officers are already handling targets and recovery.",
         "Fear that AI will reduce the role of the credit officer.",
         "Cost of new systems for small cooperatives.",
         "Resistance from senior staff used to manual appraisal.",
         "No clear accountability rule when AI is wrong.",
         "Tools built by IT without asking credit officers.",
         "Low digital literacy in rural branches.",
         "Too many disconnected systems — LOS, CBS, bureau portal — one more is a burden.",
         "Unclear regulatory guidance on AI in lending."],
}


def pick(pool, k, used):
    fresh = [a for a in pool if a not in used[k]] or pool
    a = rng.choice(fresh)
    used[k].add(a)
    return a


used = {k: set() for k in T}
for k, pools in T.items():
    col, seen = [], set()
    for tier in df["_tier"]:
        pool = pools[tier] if isinstance(pools, dict) else pools
        for _ in range(40):
            ans = pick(pool, k, used)
            if rng.random() < .25 or ans in seen:
                neg = ans.startswith(("No", "Not"))
                same = [a for a in pool if a != ans and a.startswith(("No", "Not")) == neg]
                if not same:
                    continue
                ans = f"{ans} {rng.choice(same)}"
            if ans not in seen:
                break
        seen.add(ans)
        col.append(ans)
    df[k] = col

df = df.drop(columns="_tier")
df.to_csv(OUT, index=False)
print(f"Wrote {len(df)} synthetic responses to {OUT}")
