# [Project Name: e.g., Retail Sales Forecasting]

**Author:** Group 5  
**Course:** MLOps: Master in Business Analytics and Data Sciense   
**Status:** In progress

---

## 1. Business Objective
Identifying undervalued NHL players through different regression models.

* **The Goal:** 
Evaluate and predict the offensive contribution (Points) of NHL players based on underlying on-ice performance metrics (e.g., Faceoff Win %, Takeaways, Icetime). This creates business value by helping teams identify undervalued talent, optimize salary cap space, and make data-driven trade decisions.

* **The User:** 
General Managers, Scouting Departments, and Coaching Staffs. They will consume these predictions via automated reports to compare expected player output against actual contract costs.

---

## 2. Success Metrics
*How do we know if the project is successful?*

* **Business KPI (The "Why"):**
Improve draft and trade ROI by identifying at least 3-5 undervalued players per season whose underlying metrics suggest high point-production potential.

* **Technical Metric (The "How"):**
Identify 3-5 undervalued players per season whose predicted points are at least 5 points above their actual point production.

* **Acceptance Criteria:**
The modular regression model must successfully execute end-to-end without data leakage, pass all strict validation gates (no NaNs, correct schema), and outperform a naive baseline (such as a simple moving average of previous seasons' points).

---

## 3. The Data

* **Source:** Historical NHL player performance dataset (CSV format, from MoneyPuck.com).
* **Target Variable:** Points (A composite target computed during preprocessing as Goals + Primary_Assists + Secondary_Assists).
* **Sensitive Info:** This dataset contains publicly available sports statistics. There are no emails, financial records, or Personally Identifiable Information (PII) to protect. (Note: The raw data folder is still ignored by Git to prevent committing large files).
---

## 4. Repository Structure

This project follows a strict separation between "Sandbox" (Notebooks) and "Production" (Src).

```text
.
├── README.md                # This file (Project definition)
├── environment.yml          # Dependencies (Conda/Pip)
├── config.yaml              # Global configuration (paths, params)
├── .env                     # Secrets placeholder
│
├── notebooks/               # Experimental sandbox
│   └── yourbaseline.ipynb   # From previous work
│
├── src/                     # Production code (The "Factory")
│   ├── __init__.py          # Python package
│   ├── load_data.py         # Ingest raw data
│   ├── clean_data.py        # Preprocessing & cleaning
│   ├── validate.py          # Data quality checks
│   ├── train.py             # Model training & saving
│   ├── evaluate.py          # Metrics & plotting
│   ├── infer.py             # Inference logic
│   └── main.py              # Pipeline orchestrator
│
├── data/                    # Local storage (IGNORED by Git)
│   ├── raw/                 # Immutable input data
│   └── processed/           # Cleaned data ready for training
│
├── models/                  # Serialized artifacts (IGNORED by Git)
│
├── reports/                 # Generated metrics, plots, and figures
│
└── tests/                   # Automated tests
```

## 5. Execution Model

The full machine learning pipeline will eventually be executable through:

`python src/main.py`



