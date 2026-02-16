import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)

df = pd.read_csv('data/credit_risk_dataset.csv')

#DATA CLEANING

df['person_emp_length'] = df['person_emp_length'].replace(np.nan, df['person_emp_length'].median()) #replacing missing employement length values with the median

df = df.dropna(subset=['loan_int_rate']) #dropping all rows without an interest rate since it is an integral part of any loan

#PREPROCESSING

df_num = df.select_dtypes(exclude=['object'])
df_str = df.select_dtypes(include=['object'])
df_str_onehot = pd.get_dummies(df_str)

df = pd.concat([df_num, df_str_onehot], axis=1)

X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

#LOGISTIC REGRESSION

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)

#RANDOM FOREST

rf_model = RandomForestClassifier(random_state=123, class_weight="balanced")
rf_model.fit(X_train, y_train)

#GRADIENT BOOSTING

gb_model = XGBClassifier(random_state=123, class_weight="balanced")
gb_model.fit(X_train, y_train)

#COMPARISON AND EVALUATION

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob),
        'F1': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred)
    }

results = {}

results['Logistic Regression'] = evaluate_model(log_model, X_test_scaled, y_test)
results['Random Forest'] = evaluate_model(rf_model, X_test, y_test)
results['Gradient Boosting'] = evaluate_model(gb_model, X_test, y_test)

results_df = pd.DataFrame(results).T

print(results_df)