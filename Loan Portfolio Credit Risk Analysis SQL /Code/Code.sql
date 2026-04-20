-- ============================================================
-- LOAN PORTFOLIO CREDIT RISK ANALYSIS
-- SQL Project by Mayank Sharma
-- Dataset: Synthetic Indian Banking Loan Portfolio
-- Tables: borrowers, loans, repayments
-- ============================================================

-- ============================================================
-- SECTION 1: DATABASE SCHEMA & OVERVIEW
-- ============================================================

-- Table 1: borrowers
-- customer_id, name, age, annual_income, credit_score,
-- employment_type, state, years_employed, existing_loans_count

-- Table 2: loans
-- loan_id, customer_id, loan_purpose, loan_amount, outstanding_balance,
-- interest_rate, tenure_months, emi_amount, disbursal_date, loan_status

-- Table 3: repayments
-- repayment_id, loan_id, customer_id, payment_date, amount_paid,
-- emi_due, days_late, payment_month

-- ============================================================
-- SECTION 2: PORTFOLIO OVERVIEW
-- ============================================================

-- Q1: High-level portfolio summary
SELECT
    COUNT(*)                                        AS total_loans,
    COUNT(DISTINCT l.customer_id)                   AS unique_borrowers,
    ROUND(SUM(l.loan_amount) / 1e7, 2)              AS total_disbursed_cr,
    ROUND(SUM(l.outstanding_balance) / 1e7, 2)      AS total_outstanding_cr,
    ROUND(AVG(l.loan_amount), 0)                    AS avg_loan_amount,
    ROUND(AVG(b.credit_score), 1)                   AS avg_credit_score,
    ROUND(AVG(l.interest_rate), 2)                  AS avg_interest_rate
FROM loans l
JOIN borrowers b ON l.customer_id = b.customer_id;


-- Q2: Loan book breakdown by purpose
SELECT
    loan_purpose,
    COUNT(*)                                        AS loan_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS pct_of_portfolio,
    ROUND(SUM(loan_amount) / 1e7, 2)                AS disbursed_cr,
    ROUND(AVG(loan_amount), 0)                      AS avg_ticket_size,
    ROUND(AVG(interest_rate), 2)                    AS avg_rate
FROM loans
GROUP BY loan_purpose
ORDER BY disbursed_cr DESC;


-- Q3: Loan book by status
SELECT
    loan_status,
    COUNT(*)                                         AS count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct,
    ROUND(SUM(outstanding_balance) / 1e7, 2)         AS outstanding_cr,
    ROUND(AVG(outstanding_balance), 0)               AS avg_outstanding
FROM loans
GROUP BY loan_status
ORDER BY
    CASE loan_status
        WHEN 'Active' THEN 1
        WHEN '30 DPD' THEN 2
        WHEN '60 DPD' THEN 3
        WHEN '90+ DPD' THEN 4
        WHEN 'NPA' THEN 5
        WHEN 'Written-Off' THEN 6
    END;


-- ============================================================
-- SECTION 3: NPA & DELINQUENCY ANALYSIS
-- ============================================================

-- Q4: NPA Rate by loan purpose (Gross NPA %)
SELECT
    loan_purpose,
    COUNT(*)                                                AS total_loans,
    SUM(CASE WHEN loan_status IN ('NPA','Written-Off') THEN 1 ELSE 0 END) AS npa_count,
    ROUND(
        SUM(CASE WHEN loan_status IN ('NPA','Written-Off') THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                       AS gross_npa_pct,
    ROUND(
        SUM(CASE WHEN loan_status IN ('NPA','Written-Off') THEN outstanding_balance ELSE 0 END) /
        NULLIF(SUM(outstanding_balance), 0) * 100, 2
    )                                                       AS npa_by_value_pct
FROM loans
GROUP BY loan_purpose
ORDER BY gross_npa_pct DESC;


-- Q5: Delinquency roll-rate matrix — count of loans at each DPD bucket
SELECT
    loan_purpose,
    SUM(CASE WHEN loan_status = 'Active'     THEN 1 ELSE 0 END) AS active,
    SUM(CASE WHEN loan_status = '30 DPD'     THEN 1 ELSE 0 END) AS dpd_30,
    SUM(CASE WHEN loan_status = '60 DPD'     THEN 1 ELSE 0 END) AS dpd_60,
    SUM(CASE WHEN loan_status = '90+ DPD'    THEN 1 ELSE 0 END) AS dpd_90_plus,
    SUM(CASE WHEN loan_status = 'NPA'        THEN 1 ELSE 0 END) AS npa,
    SUM(CASE WHEN loan_status = 'Written-Off' THEN 1 ELSE 0 END) AS written_off
FROM loans
GROUP BY loan_purpose
ORDER BY loan_purpose;


-- Q6: Credit score band vs NPA rate
SELECT
    CASE
        WHEN b.credit_score < 500          THEN '300-499 (Poor)'
        WHEN b.credit_score BETWEEN 500 AND 599 THEN '500-599 (Fair)'
        WHEN b.credit_score BETWEEN 600 AND 699 THEN '600-699 (Good)'
        WHEN b.credit_score BETWEEN 700 AND 749 THEN '700-749 (Very Good)'
        ELSE                                    '750+ (Excellent)'
    END                                                     AS credit_band,
    COUNT(*)                                                AS loans,
    ROUND(AVG(b.credit_score), 0)                           AS avg_score,
    ROUND(
        SUM(CASE WHEN l.loan_status IN ('NPA','Written-Off') THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                       AS npa_rate_pct,
    ROUND(AVG(l.loan_amount), 0)                            AS avg_loan_amount
FROM loans l
JOIN borrowers b ON l.customer_id = b.customer_id
GROUP BY credit_band
ORDER BY avg_score;


-- ============================================================
-- SECTION 4: BORROWER RISK SEGMENTATION
-- ============================================================

-- Q7: Risk segmentation using credit score + income (P1-P4 model)
-- P1 = Prime, P2 = Near-Prime, P3 = Sub-Prime, P4 = Deep Sub-Prime
SELECT
    risk_segment,
    COUNT(*)                                                    AS borrowers,
    ROUND(AVG(b.credit_score), 0)                               AS avg_credit_score,
    ROUND(AVG(b.annual_income), 0)                              AS avg_income,
    ROUND(
        SUM(CASE WHEN l.loan_status IN ('NPA','Written-Off') THEN 1 ELSE 0 END) * 100.0
        / COUNT(*), 2
    )                                                           AS npa_rate_pct,
    ROUND(SUM(l.outstanding_balance) / 1e7, 2)                  AS outstanding_cr
FROM loans l
JOIN borrowers b ON l.customer_id = b.customer_id
JOIN (
    SELECT
        customer_id,
        CASE
            WHEN credit_score >= 720 AND annual_income >= 800000  THEN 'P1 - Prime'
            WHEN credit_score >= 650 AND annual_income >= 400000  THEN 'P2 - Near-Prime'
            WHEN credit_score >= 550 AND annual_income >= 200000  THEN 'P3 - Sub-Prime'
            ELSE                                                      'P4 - Deep Sub-Prime'
        END AS risk_segment
    FROM borrowers
) seg ON b.customer_id = seg.customer_id
GROUP BY risk_segment
ORDER BY risk_segment;


-- Q8: Debt-to-Income (DTI) ratio analysis
SELECT
    dti_band,
    COUNT(*)                                                    AS loan_count,
    ROUND(
        SUM(CASE WHEN l.loan_status IN ('NPA','Written-Off') THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                           AS npa_rate_pct,
    ROUND(AVG(l.loan_amount), 0)                                AS avg_loan_amount,
    ROUND(AVG(b.credit_score), 0)                               AS avg_credit_score
FROM loans l
JOIN borrowers b ON l.customer_id = b.customer_id
JOIN (
    SELECT
        l2.loan_id,
        CASE
            WHEN (l2.emi_amount * 12) / NULLIF(b2.annual_income, 0) < 0.20 THEN '< 20% (Low Risk)'
            WHEN (l2.emi_amount * 12) / NULLIF(b2.annual_income, 0) < 0.35 THEN '20-35% (Moderate)'
            WHEN (l2.emi_amount * 12) / NULLIF(b2.annual_income, 0) < 0.50 THEN '35-50% (High)'
            ELSE '> 50% (Very High)'
        END AS dti_band
    FROM loans l2
    JOIN borrowers b2 ON l2.customer_id = b2.customer_id
) dti ON l.loan_id = dti.loan_id
GROUP BY dti_band
ORDER BY npa_rate_pct;


-- ============================================================
-- SECTION 5: REPAYMENT BEHAVIOUR ANALYSIS
-- ============================================================

-- Q9: Monthly repayment collection trend (2020-2023)
SELECT
    SUBSTR(payment_date, 1, 7)                          AS month,
    COUNT(*)                                            AS payments_made,
    ROUND(SUM(amount_paid) / 1e7, 3)                    AS collection_cr,
    ROUND(SUM(emi_due) / 1e7, 3)                        AS demand_cr,
    ROUND(SUM(amount_paid) / NULLIF(SUM(emi_due), 0) * 100, 2) AS collection_efficiency_pct,
    SUM(CASE WHEN days_late > 0 THEN 1 ELSE 0 END)      AS late_payments
FROM repayments
WHERE payment_date >= '2020-01-01' AND payment_date <= '2023-12-31'
GROUP BY month
ORDER BY month;


-- Q10: Running cumulative collection per loan (Window Function)
SELECT
    loan_id,
    payment_month,
    payment_date,
    amount_paid,
    SUM(amount_paid) OVER (
        PARTITION BY loan_id ORDER BY payment_month
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )                                                   AS cumulative_collected,
    AVG(amount_paid) OVER (
        PARTITION BY loan_id ORDER BY payment_month
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    )                                                   AS rolling_3m_avg_payment
FROM repayments
WHERE loan_id IN ('LN00001','LN00002','LN00003','LN00004','LN00005')
ORDER BY loan_id, payment_month;


-- Q11: Borrower payment consistency score
-- Flag borrowers with 3+ consecutive late payments
WITH late_flags AS (
    SELECT
        r.customer_id,
        r.loan_id,
        r.payment_month,
        r.days_late,
        CASE WHEN r.days_late > 0 THEN 1 ELSE 0 END    AS is_late,
        LAG(CASE WHEN r.days_late > 0 THEN 1 ELSE 0 END, 1, 0) OVER (
            PARTITION BY r.loan_id ORDER BY r.payment_month
        )                                               AS prev1_late,
        LAG(CASE WHEN r.days_late > 0 THEN 1 ELSE 0 END, 2, 0) OVER (
            PARTITION BY r.loan_id ORDER BY r.payment_month
        )                                               AS prev2_late
    FROM repayments r
),
consec_late AS (
    SELECT
        customer_id,
        loan_id,
        MAX(CASE WHEN is_late = 1 AND prev1_late = 1 AND prev2_late = 1 THEN 1 ELSE 0 END) AS has_3_consec_late
    FROM late_flags
    GROUP BY customer_id, loan_id
)
SELECT
    b.employment_type,
    COUNT(*)                                            AS total_loans,
    SUM(cl.has_3_consec_late)                           AS loans_with_consec_late,
    ROUND(SUM(cl.has_3_consec_late) * 100.0 / COUNT(*), 2) AS pct_with_consec_late
FROM consec_late cl
JOIN borrowers b ON cl.customer_id = b.customer_id
GROUP BY b.employment_type
ORDER BY pct_with_consec_late DESC;


-- Q12: Payment shortfall analysis (partial payment detection)
SELECT
    l.loan_purpose,
    COUNT(r.repayment_id)                               AS total_payments,
    SUM(CASE WHEN r.amount_paid < r.emi_due * 0.95 THEN 1 ELSE 0 END) AS partial_payments,
    ROUND(
        SUM(CASE WHEN r.amount_paid < r.emi_due * 0.95 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                   AS partial_pct,
    ROUND(SUM(GREATEST(r.emi_due - r.amount_paid, 0)) / 1e5, 2) AS total_shortfall_lacs
FROM repayments r
JOIN loans l ON r.loan_id = l.loan_id
GROUP BY l.loan_purpose
ORDER BY partial_pct DESC;


-- ============================================================
-- SECTION 6: GEOGRAPHIC & DEMOGRAPHIC ANALYSIS
-- ============================================================

-- Q13: State-wise portfolio concentration & NPA
SELECT
    b.state,
    COUNT(l.loan_id)                                    AS total_loans,
    ROUND(SUM(l.loan_amount) / 1e7, 2)                  AS disbursed_cr,
    ROUND(
        SUM(CASE WHEN l.loan_status IN ('NPA','Written-Off') THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                   AS npa_rate_pct,
    ROUND(AVG(b.credit_score), 0)                       AS avg_credit_score,
    ROUND(AVG(b.annual_income / 100000.0), 2)           AS avg_income_lacs,
    RANK() OVER (ORDER BY SUM(l.loan_amount) DESC)      AS portfolio_rank
FROM loans l
JOIN borrowers b ON l.customer_id = b.customer_id
GROUP BY b.state
ORDER BY disbursed_cr DESC;


-- Q14: Age-group vs default propensity
SELECT
    CASE
        WHEN b.age < 30 THEN '< 30'
        WHEN b.age BETWEEN 30 AND 39 THEN '30-39'
        WHEN b.age BETWEEN 40 AND 49 THEN '40-49'
        ELSE '50+'
    END                                                 AS age_group,
    COUNT(*)                                            AS loans,
    ROUND(AVG(l.loan_amount), 0)                        AS avg_loan,
    ROUND(AVG(b.credit_score), 0)                       AS avg_score,
    ROUND(
        SUM(CASE WHEN l.loan_status IN ('NPA','Written-Off') THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                   AS npa_rate_pct
FROM loans l
JOIN borrowers b ON l.customer_id = b.customer_id
GROUP BY age_group
ORDER BY age_group;


-- ============================================================
-- SECTION 7: PORTFOLIO CONCENTRATION RISK
-- ============================================================

-- Q15: Top 20 borrowers by outstanding exposure (Concentration Risk)
SELECT
    b.customer_id,
    b.name,
    b.state,
    b.credit_score,
    COUNT(l.loan_id)                                    AS num_loans,
    ROUND(SUM(l.outstanding_balance) / 1e5, 2)         AS total_exposure_lacs,
    ROUND(
        SUM(l.outstanding_balance) / SUM(SUM(l.outstanding_balance)) OVER () * 100, 3
    )                                                   AS pct_of_portfolio,
    GROUP_CONCAT(l.loan_purpose, ' | ')                AS loan_types
FROM loans l
JOIN borrowers b ON l.customer_id = b.customer_id
GROUP BY b.customer_id
ORDER BY total_exposure_lacs DESC
LIMIT 20;


-- Q16: Herfindahl-Hirschman Index (HHI) by loan purpose
-- HHI measures concentration; higher = more concentrated
WITH purpose_share AS (
    SELECT
        loan_purpose,
        SUM(outstanding_balance) / (SELECT SUM(outstanding_balance) FROM loans) * 100 AS share_pct
    FROM loans
    GROUP BY loan_purpose
)
SELECT
    loan_purpose,
    ROUND(share_pct, 2)                                 AS share_pct,
    ROUND(share_pct * share_pct, 4)                    AS hhi_contribution
FROM purpose_share
UNION ALL
SELECT
    'TOTAL HHI INDEX' AS loan_purpose,
    NULL,
    ROUND(SUM(share_pct * share_pct), 2)
FROM purpose_share
ORDER BY hhi_contribution DESC;


-- ============================================================
-- SECTION 8: VINTAGE ANALYSIS
-- ============================================================

-- Q17: Vintage analysis — NPA rates by disbursement year
SELECT
    SUBSTR(l.disbursal_date, 1, 4)                      AS vintage_year,
    COUNT(*)                                            AS loans_disbursed,
    ROUND(SUM(l.loan_amount) / 1e7, 2)                  AS disbursed_cr,
    SUM(CASE WHEN l.loan_status IN ('NPA','Written-Off') THEN 1 ELSE 0 END) AS npas,
    ROUND(
        SUM(CASE WHEN l.loan_status IN ('NPA','Written-Off') THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                   AS npa_rate_pct,
    ROUND(AVG(b.credit_score), 0)                       AS avg_credit_score_at_origination
FROM loans l
JOIN borrowers b ON l.customer_id = b.customer_id
GROUP BY vintage_year
ORDER BY vintage_year;


-- Q18: Early Default Detection (loans defaulting within 12 months)
WITH first_default AS (
    SELECT
        r.loan_id,
        MIN(CASE WHEN r.days_late > 30 THEN r.payment_month ELSE NULL END) AS first_default_month
    FROM repayments r
    GROUP BY r.loan_id
)
SELECT
    CASE
        WHEN fd.first_default_month <= 6  THEN '0-6 months (Early Default)'
        WHEN fd.first_default_month <= 12 THEN '7-12 months'
        WHEN fd.first_default_month <= 24 THEN '13-24 months'
        ELSE '> 24 months'
    END                                                 AS default_timing,
    COUNT(*)                                            AS loan_count,
    ROUND(AVG(b.credit_score), 0)                       AS avg_credit_score,
    ROUND(AVG(b.annual_income), 0)                      AS avg_income,
    ROUND(AVG(l.loan_amount), 0)                        AS avg_loan_amount
FROM first_default fd
JOIN loans l ON fd.loan_id = l.loan_id
JOIN borrowers b ON l.customer_id = b.customer_id
WHERE fd.first_default_month IS NOT NULL
GROUP BY default_timing
ORDER BY default_timing;


-- ============================================================
-- SECTION 9: ADVANCED WINDOW FUNCTIONS
-- ============================================================

-- Q19: Month-over-Month NPA trend using LAG
WITH monthly_npa AS (
    SELECT
        SUBSTR(l.disbursal_date, 1, 7)                  AS cohort_month,
        COUNT(*)                                        AS total_loans,
        SUM(CASE WHEN l.loan_status IN ('NPA','Written-Off') THEN 1 ELSE 0 END) AS npa_loans
    FROM loans l
    GROUP BY cohort_month
)
SELECT
    cohort_month,
    total_loans,
    npa_loans,
    ROUND(npa_loans * 100.0 / total_loans, 2)           AS npa_rate_pct,
    LAG(ROUND(npa_loans * 100.0 / total_loans, 2), 1) OVER (
        ORDER BY cohort_month
    )                                                   AS prev_month_npa_rate,
    ROUND(
        ROUND(npa_loans * 100.0 / total_loans, 2) -
        LAG(ROUND(npa_loans * 100.0 / total_loans, 2), 1) OVER (ORDER BY cohort_month), 2
    )                                                   AS mom_change
FROM monthly_npa
ORDER BY cohort_month;


-- Q20: Percentile ranking of borrowers by risk exposure
SELECT
    b.customer_id,
    b.name,
    b.credit_score,
    b.annual_income,
    l.loan_amount,
    l.loan_status,
    NTILE(4) OVER (ORDER BY b.credit_score DESC)        AS credit_quartile,
    NTILE(4) OVER (ORDER BY l.loan_amount DESC)         AS loan_size_quartile,
    PERCENT_RANK() OVER (ORDER BY b.credit_score DESC)  AS credit_score_percentile,
    DENSE_RANK() OVER (
        PARTITION BY l.loan_purpose ORDER BY l.loan_amount DESC
    )                                                   AS rank_within_product
FROM loans l
JOIN borrowers b ON l.customer_id = b.customer_id
ORDER BY credit_quartile, loan_size_quartile
LIMIT 50;


-- Q21: 3-month rolling average EMI collection per product
SELECT
    loan_purpose,
    payment_month_str,
    monthly_collection,
    ROUND(AVG(monthly_collection) OVER (
        PARTITION BY loan_purpose
        ORDER BY payment_month_str
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 2)                                               AS rolling_3m_avg
FROM (
    SELECT
        l.loan_purpose,
        SUBSTR(r.payment_date, 1, 7)                    AS payment_month_str,
        ROUND(SUM(r.amount_paid) / 1e5, 2)              AS monthly_collection
    FROM repayments r
    JOIN loans l ON r.loan_id = l.loan_id
    GROUP BY l.loan_purpose, payment_month_str
) monthly_data
ORDER BY loan_purpose, payment_month_str;


-- ============================================================
-- SECTION 10: PROVISION & EXPECTED LOSS ESTIMATION
-- ============================================================

-- Q22: Simplified Expected Loss calculation
-- EL = PD * LGD * EAD
-- Using proxy PD from historical NPA rates, LGD = 60% for unsecured, 40% secured
SELECT
    l.loan_purpose,
    COUNT(*)                                                AS loans,
    ROUND(SUM(l.outstanding_balance) / 1e7, 2)             AS ead_cr,
    ROUND(
        SUM(CASE WHEN l.loan_status IN ('NPA','Written-Off') THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4
    )                                                       AS pd_estimate,
    CASE
        WHEN l.loan_purpose IN ('Personal Loan','Education Loan','Gold Loan') THEN 0.60
        ELSE 0.40
    END                                                     AS lgd_assumption,
    ROUND(
        SUM(l.outstanding_balance) *
        (SUM(CASE WHEN l.loan_status IN ('NPA','Written-Off') THEN 1.0 ELSE 0 END) / COUNT(*)) *
        CASE WHEN l.loan_purpose IN ('Personal Loan','Education Loan','Gold Loan') THEN 0.60 ELSE 0.40 END
        / 1e7, 4
    )                                                       AS expected_loss_cr
FROM loans l
GROUP BY l.loan_purpose
ORDER BY expected_loss_cr DESC;


-- Q23: Watch-list loans — high risk early warning signals
SELECT
    l.loan_id,
    b.name,
    b.credit_score,
    l.loan_purpose,
    l.outstanding_balance,
    l.loan_status,
    b.existing_loans_count,
    ROUND((l.emi_amount * 12) / NULLIF(b.annual_income, 0) * 100, 1) AS dti_pct,
    CASE
        WHEN b.credit_score < 550
         AND l.loan_status IN ('30 DPD','60 DPD')
         AND b.existing_loans_count >= 2               THEN 'HIGH RISK - Watch Immediately'
        WHEN b.credit_score < 620
         AND l.loan_status IN ('30 DPD','60 DPD')     THEN 'MODERATE RISK - Monitor Closely'
        ELSE 'STANDARD MONITORING'
    END                                                 AS watch_flag
FROM loans l
JOIN borrowers b ON l.customer_id = b.customer_id
WHERE l.loan_status IN ('30 DPD','60 DPD')
  AND b.credit_score < 650
ORDER BY watch_flag, b.credit_score ASC
LIMIT 30;


-- ============================================================
-- END OF PROJECT
-- ============================================================
