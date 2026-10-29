# Data Dictionary — UCI Default of Credit Card Clients

## Provenance

| Item | Value |
|---|---|
| Dataset | Default of Credit Card Clients |
| Repository | UCI Machine Learning Repository, dataset ID 350 |
| URL | https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients |
| DOI | 10.24432/C55S3H |
| Creator | I-Cheng Yeh |
| Donated to UCI | 2016-01-25 |
| Licence | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Introductory paper | Yeh, I.-C., & Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473–2480. |
| File used | `data/raw/default of credit card clients.xls` (5,539,328 bytes) |
| SHA-256 | `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933` |
| Retrieved | The project owner downloaded the file; its local modification time is 2026-10-29 16:34 UTC. The exact download date is not otherwise recorded. |
| Shape | 30,000 rows × 25 columns (ID, 23 explanatory variables, target) |

The raw file is not modified and is not committed to Git (CC BY 4.0 would allow redistribution with attribution, but the project downloads it from the source instead).

## Variables

Amounts are in New Taiwan dollars (NT$). Monthly index 1 = September 2005, 6 = April 2005.
"UCI description" means the variable information published on the dataset page. "Observed" means what the file actually contains (see `reports/data_quality_report.json`).

| Column (project name) | Raw name | UCI description | Observed values / notes | Role |
|---|---|---|---|---|
| `ID` | ID | Client identifier | 1–30,000, unique | **Excluded** from features |
| `LIMIT_BAL` | X1 | Amount of given credit, including individual consumer credit and family (supplementary) credit | 10,000 – 1,000,000 | Feature |
| `SEX` | X2 | 1 = male, 2 = female | Only documented codes | Feature (protected attribute; see Limitations) |
| `EDUCATION` | X3 | 1 = graduate school, 2 = university, 3 = high school, 4 = others | Also 0 (14), 5 (280), 6 (51): **undocumented** | Feature (categorical) |
| `MARRIAGE` | X4 | 1 = married, 2 = single, 3 = others | Also 0 (54): **undocumented** | Feature (protected attribute; categorical) |
| `AGE` | X5 | Age in years | 21 – 79 | Feature (protected attribute) |
| `PAY_1` … `PAY_6` | X6–X11 (`PAY_0`, `PAY_2`–`PAY_6`) | Repayment status, Sep → Apr 2005: −1 = paid duly; 1 = payment delay of one month; … ; 9 = delay of nine months or more | −2 and 0 occur in most rows: **undocumented**; maximum observed is 8. Raw `PAY_0` renamed to `PAY_1` | Feature (ordinal) |
| `BILL_AMT1` … `BILL_AMT6` | X12–X17 | Amount of bill statement, Sep → Apr 2005 | Negative in ~2% of rows per month (credit balance); some exceed `LIMIT_BAL` | Feature |
| `PAY_AMT1` … `PAY_AMT6` | X18–X23 | Amount of previous payment, Sep → Apr 2005 | 0 – 1,684,259; never negative | Feature |
| `default_next_month` | Y (`default payment next month`) | Default payment: 1 = yes, 0 = no | 6,636 defaults (22.12%) | **Target** |

### Interpreting undocumented `PAY_*` codes

Secondary sources often read −2 as "no consumption" and 0 as "use of revolving credit", but the official UCI description doesn't confirm this. The project treats `PAY_*` as an ordinal delay scale. Values ≤ 0 mean "no recorded delay", and the models may still use the distinct values. Any reading of the −2/0 codes is an **assumption**, not a documented fact.

### Timing of `PAY_AMT` relative to `BILL_AMT`

The UCI page calls `PAY_AMTk` the "amount of previous payment" in month *k*. It doesn't say which bill statement each payment settles. Engineered ratios such as "payment / previous bill" therefore rest on an assumed alignment, and they are documented as such where they are used.

## Data-quality summary

Produced by `scripts/audit_data.py` → `reports/data_quality_report.json`.

- No missing values; IDs unique; target binary.
- 399 rows have an undocumented `EDUCATION` or `MARRIAGE` code.
- 795 inactive accounts: zero bills and zero payments for all six months.
- 3,931 rows where at least one monthly bill exceeds `LIMIT_BAL`.
- 56 rows repeat another row's feature vector (ID excluded). 21 of these duplicate groups contain both outcomes.

## Limitations

- **One cross-section**, from Taiwan in 2005. It can't establish temporal stability or represent a current portfolio.
- **Default definition** is the UCI flag. There is no days-past-due rule or materiality threshold, so it is not equivalent to the regulatory definition of default (e.g. CRR Article 178 / EBA guidelines).
- **No EAD, LGD or loss information.** Expected loss, IFRS 9 provisions and capital cannot be computed from this data.
- **Protected attributes** (`SEX`, `MARRIAGE`, `AGE`) are used as model inputs at the project owner's request, to maximise predictive performance. The project reports fairness diagnostics. Using them in a real lending decision would need legal and fair-lending review.
