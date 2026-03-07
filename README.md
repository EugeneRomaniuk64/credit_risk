# Credit Risk Analysis

A machine learning project for predicting loan defaults and estimating expected credit loss. The project covers the full pipeline from exploratory data analysis through to model comparison and financial risk quantification.

---

## Project Structure

```
credit_risk/
├── data/
│   └── credit_risk_dataset.csv
├── src/
│   ├── exploratory_data_analysis.ipynb
│   └── main.py
├── .gitignore
├── LICENSE
└── README.md
```

---

## Overview

This project builds and compares three classification models to predict whether a borrower will default on a loan. Beyond accuracy metrics, the project calculates **expected credit loss (EL)** for each model using the formula:

```
Expected Loss = PD × LGD × EAD
```

Where:
- **PD** (Probability of Default) — predicted by the model
- **LGD** (Loss Given Default) — fixed at 45%
- **EAD** (Exposure at Default) — the loan amount

---

## Workflow

### 1. Exploratory Data Analysis (`exploratory_data_analysis.ipynb`)

- Loads and inspects the dataset (shape, types, summary statistics)
- Identifies data quality issues — outliers in `person_age` and `person_emp_length`
- Checks for missing values and duplicates
- **Univariate analysis** — distribution plots with skewness for all numerical features, class balance of loan status
- **Bivariate analysis** — pair plot across all features
- **Correlation matrix** — heatmap of numerical features
- Deeper exploration of `loan_int_rate` and `loan_percent_income` vs loan status due to high correlation

### 2. Modelling (`main.py`)

**Data Cleaning**
- Imputes missing `person_emp_length` with the median
- Drops rows with missing `loan_int_rate`

**Preprocessing**
- One-hot encodes categorical features
- Applies `StandardScaler` for Logistic Regression

**Models Trained**
| Model | Notes |
|---|---|
| Logistic Regression | Scaled features, default params |
| Random Forest | `class_weight="balanced"`, `random_state=123` |
| Gradient Boosting (XGBoost) | `random_state=123` |

**Threshold Optimisation**
- For each model, the classification threshold is optimised using 5-fold cross-validation out-of-fold predictions, maximising the F1 score

**Evaluation Metrics**
- Accuracy, ROC-AUC, F1, Precision, Recall, Brier Score
- Expected Credit Loss (USD)
- ROC curves plotted for all three models side by side

---

## Requirements

Install dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

---

## Usage

**Run EDA:**

Open `src/exploratory_data_analysis.ipynb` in VS Code or Jupyter and run all cells.

**Run modelling pipeline:**

```bash
python src/main.py
```

This will print a comparison table of all three models and display ROC curves.

---

## Dataset

The project uses `credit_risk_dataset.csv`, which should be placed in the `data/` directory. The dataset contains borrower attributes such as age, income, employment length, loan amount, interest rate, and loan status (default/non-default).

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.