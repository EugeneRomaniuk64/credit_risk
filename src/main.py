import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    brier_score_loss
)

LGD = 0.45

def find_optimal_threshold(model, X_train, y_train):
    
    oof_probs = cross_val_predict(model, X_train, y_train, cv=5, method='predict_proba')[:, 1]

    thresholds = np.linspace(0, 1, 100)
    scores = [f1_score(y_train, oof_probs >= t) for t in thresholds]

    optimal_t = thresholds[np.argmax(scores)]
    
    return optimal_t

def evaluate_model(model, X_test, y_test, threshold, EAD_series):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    expected_loss = y_prob * LGD * EAD_series

    total_exprected_loss = np.sum(expected_loss)

    fpr, tpr, thresholds_roc = roc_curve(y_test, y_prob)

    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob),
        'F1': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'Brier Score': brier_score_loss(y_test, y_prob),
        'Expected Loss': f'${total_exprected_loss:,.0f}',
        'FPR': fpr,
        'TPR': tpr,
        'y_prob': y_prob
    }



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

gb_model = XGBClassifier(random_state=123)
gb_model.fit(X_train, y_train)

#OPTIMIZING THRESHOLDS 

threshold_log = find_optimal_threshold(log_model, X_train_scaled, y_train)
threshold_rf = find_optimal_threshold(rf_model, X_train, y_train)
threshold_gb = find_optimal_threshold(gb_model, X_train, y_train)

#COMPARISON AND EVALUATION

results = {}

results['Logistic Regression'] = evaluate_model(log_model, X_test_scaled, y_test, threshold_log, X_test['loan_amnt'])
results['Random Forest'] = evaluate_model(rf_model, X_test, y_test, threshold_rf, X_test['loan_amnt'])
results['Gradient Boosting'] = evaluate_model(gb_model, X_test, y_test, threshold_gb, X_test['loan_amnt'])

results_df = pd.DataFrame(results).T

print(results_df.drop(['FPR', 'TPR', 'y_prob'], axis=1))

fig, ax = plt.subplots(1, 3, figsize=(12, 6))

for model_name, i in zip(results_df.index.to_list(), range(3)):
    ax[i].plot(
        results_df.at[model_name, 'FPR'],
        results_df.at[model_name, 'TPR'],
        label=f'ROC Curve (AUC = {results_df.at[model_name, 'ROC-AUC']:.2f})'
    )
    ax[i].plot([0, 1], [0, 1], linestyle="--")
    ax[i].set_title(f'ROC Curve for {model_name}')
    ax[i].set_xlabel('False Positive Rate'); ax[i].set_ylabel('True Positive Rate')

plt.show()