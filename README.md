# 📊 Loan Portfolio Credit Risk Analysis

A comprehensive SQL-based credit risk analytics project simulating a real-world retail lending portfolio. Built using **SQLite**, this project covers everything from portfolio-level summaries to advanced risk metrics — NPA analysis, borrower segmentation, vintage analysis, expected loss estimation, and early warning watchlists.

---

## 🗂️ Repository Structure

```
Risk/
├── borrowers.csv              # Borrower demographic & credit data (5,000 records)
├── loans.csv                  # Loan-level data with status and financials
├── repayments.csv             # Repayment transaction records (174,632 rows)
└── loan_portfolio_analysis.sql # 23 analytical SQL queries across 10 sections
```

---

## 🧱 Dataset Overview

### Table 1: `borrowers`
| Column | Description |
|---|---|
| `customer_id` | Unique borrower identifier |
| `name` | Borrower name |
| `age` | Age of borrower |
| `annual_income` | Annual income (INR) |
| `credit_score` | CIBIL-style credit score |
| `employment_type` | Salaried / Self-Employed / etc. |
| `state` | Indian state of residence |
| `years_employed` | Employment tenure |
| `existing_loans_count` | Number of existing loans at origination |

### Table 2: `loans`
| Column | Description |
|---|---|
| `loan_id` | Unique loan identifier |
| `customer_id` | FK to borrowers |
| `loan_purpose` | Personal / Home / Auto / Education / Gold Loan etc. |
| `loan_amount` | Disbursed amount (INR) |
| `outstanding_balance` | Current outstanding amount |
| `interest_rate` | Annual interest rate (%) |
| `tenure_months` | Loan tenure |
| `emi_amount` | Monthly EMI |
| `disbursal_date` | Date of disbursement |
| `loan_status` | Active / 30 DPD / 60 DPD / 90+ DPD / NPA / Written-Off |

### Table 3: `repayments`
| Column | Description |
|---|---|
| `repayment_id` | Unique payment record |
| `loan_id` | FK to loans |
| `customer_id` | FK to borrowers |
| `payment_date` | Date of payment |
| `amount_paid` | Amount actually paid |
| `emi_due` | EMI amount due |
| `days_late` | Days delayed (0 = on-time) |
| `payment_month` | Payment month number in loan cycle |

---

## 🔍 Analysis Sections

### 1. Portfolio Summary (Q1–Q3)
- Total loans disbursed, unique borrowers, average loan size, and credit score
- Loan book breakdown by **purpose** (Personal, Home, Auto, Education, Gold, Business)
- Loan book by **status** (Active → 30 DPD → 60 DPD → 90+ DPD → NPA → Written-Off)

### 2. NPA & Delinquency Analysis (Q4–Q6)
- **Gross NPA %** by loan purpose (count-based and value-based)
- **Delinquency roll-rate matrix** — DPD bucket distribution across product types
- **Credit score band analysis** — NPA rates segmented into Poor / Fair / Good / Very Good / Excellent buckets

### 3. Borrower Risk Segmentation (Q7–Q8)
- **P1–P4 risk model** — Prime, Near-Prime, Sub-Prime, Deep Sub-Prime classification using credit score + income thresholds
- **Debt-to-Income (DTI) ratio** bands with NPA rate and credit score correlation

### 4. Repayment Behaviour Analysis (Q9–Q12)
- **Monthly collection trend** (2020–2023) — payments made, demand vs. collection, efficiency %
- **Running cumulative collection per loan** using `SUM() OVER()` window functions
- **Consecutive late payment detection** — flags borrowers with 3+ straight late payments using `LAG()`
- **Partial payment / shortfall analysis** by product type

### 5. Geographic & Demographic Analysis (Q13–Q14)
- **State-wise portfolio concentration** with NPA rates and portfolio rank using `RANK() OVER()`
- **Age-group default propensity** — NPA rates across <30, 30–39, 40–49, 50+ cohorts

### 6. Portfolio Concentration Risk (Q15–Q16)
- **Top 20 borrowers by exposure** — identifies single-borrower concentration with portfolio % share
- **Herfindahl-Hirschman Index (HHI)** by loan purpose to quantify portfolio concentration

### 7. Vintage Analysis (Q17–Q18)
- **Vintage NPA rates by disbursement year** — tracks credit quality evolution over time
- **Early Default Detection** — identifies loans defaulting within 0–6 months vs. 7–12 months

### 8. Trend Analysis (Q19)
- **Month-over-Month NPA trend** using `LAG()` window function on disbursement cohorts

### 9. Advanced Ranking & Percentile (Q20–Q21)
- **Percentile ranking of borrowers** using `NTILE()`, `PERCENT_RANK()`, and `DENSE_RANK()`
- **3-month rolling average EMI collection** per product using `ROWS BETWEEN` window frame

### 10. Provision & Expected Loss Estimation (Q22–Q23)
- **Expected Loss (EL) calculation** — `EL = PD × LGD × EAD` framework with differentiated LGD assumptions (60% unsecured, 40% secured)
- **Watchlist generation** — flags HIGH RISK and MODERATE RISK loans based on credit score, DPD bucket, and existing loan count

---

## ⚙️ How to Run

**Requirements:** SQLite3 (or any SQLite-compatible tool like DB Browser for SQLite)

```bash
# 1. Clone the repository
git clone https://github.com/mankoyyy/Risk.git
cd Risk

# 2. Launch SQLite and create the database
sqlite3 credit_risk.db

# 3. Import CSVs
.mode csv
.import borrowers.csv borrowers
.import loans.csv loans
.import repayments.csv repayments

# 4. Run the analysis
.read loan_portfolio_analysis.sql
```

---

## 🛠️ SQL Concepts Used

- **Aggregate functions** — `COUNT`, `SUM`, `AVG`, `ROUND`
- **Window functions** — `SUM() OVER()`, `AVG() OVER()`, `LAG()`, `RANK()`, `DENSE_RANK()`, `NTILE()`, `PERCENT_RANK()`
- **CTEs** — `WITH` clause for multi-step logic (consecutive late payments, early default detection, HHI)
- **CASE WHEN** — custom segmentation (risk bands, DPD buckets, age groups, DTI bands)
- **Subqueries & JOINs** — multi-table joins across borrowers, loans, and repayments
- **NULLIF / GREATEST** — safe division and shortfall calculations

---

## 📌 Key Business Metrics Computed

| Metric | Description |
|---|---|
| Gross NPA % | Count of NPA+Written-Off loans / Total loans |
| NPA by Value % | Outstanding balance in NPA / Total outstanding |
| Collection Efficiency % | Amount paid / EMI due |
| DTI Ratio | Annual EMI / Annual income |
| Expected Loss (EL) | PD × LGD × EAD |
| HHI Index | Σ(share²) — portfolio concentration measure |
| Early Default Rate | Loans defaulting within 12 months of disbursal |

---

## 👤 Author

**Mayank Sharma** 
🔗 [GitHub](https://github.com/mankoyyy)

---

> *This project is part of a portfolio demonstrating applied credit risk analytics skills using SQL, built to simulate real-world retail lending analysis.*
