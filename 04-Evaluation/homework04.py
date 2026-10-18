
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score

# 1. Load data
url = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv"
df = pd.read_csv(url)

print("Dataset shape:", df.shape)
print("\nMissing values before filling:")
print(df.isnull().sum())

# 2. Data preparation
# Fill missing values
categorical = ['lead_source', 'industry', 'employment_status', 'location']
numerical = ['annual_income', 'number_of_courses_viewed', 'interaction_count', 'lead_score']

for col in categorical:
    df[col] = df[col].fillna('NA')

for col in numerical:
    df[col] = df[col].fillna(0.0)

print("\nMissing values after filling:")
print(df.isnull().sum())

# 3. Train / Validation / Test split
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_full_train = df_full_train.reset_index(drop=True)

y_train = df_train.converted.values
y_val = df_val.converted.values
y_test = df_test.converted.values
y_full_train = df_full_train.converted.values

del df_train['converted']
del df_val['converted']
del df_test['converted']
del df_full_train['converted']

print(f"\nTrain size: {len(df_train)}")
print(f"Val size:   {len(df_val)}")
print(f"Test size:  {len(df_test)}")

# Question 1: ROC AUC feature importance
print("\n" + "="*60)
print("QUESTION 1: ROC AUC feature importance")
print("="*60)

auc_scores = {}
for col in numerical:
    auc = roc_auc_score(y_train, df_train[col])
    if auc < 0.5:
        auc = roc_auc_score(y_train, -df_train[col])
    auc_scores[col] = auc
    print(f"{col:30s}: {auc:.4f}")

best_feature = max(auc_scores, key=auc_scores.get)
print(f"\n→ Highest AUC feature: {best_feature}")

# Question 2: Train logistic regression + AUC on validation
print("\n" + "="*60)
print("QUESTION 2: Model AUC on validation set")
print("="*60)

dv = DictVectorizer(sparse=False)

train_dicts = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dicts)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
print(f"Validation ROC AUC: {auc:.3f}")

# Question 3 & 4: Precision, Recall and F1 across thresholds
print("\n" + "="*60)
print("QUESTION 3 & 4: Precision / Recall / F1")
print("="*60)

thresholds = np.linspace(0, 1, 101)

scores = []
for t in thresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)

    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)

    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()
    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    scores.append((t, precision, recall, f1, abs(precision - recall)))

df_scores = pd.DataFrame(scores, columns=['threshold', 'precision', 'recall', 'f1', 'diff'])

# Q3: intersection (smallest |precision - recall| where both > 0)
df_valid = df_scores[(df_scores.precision > 0) & (df_scores.recall > 0)]
best_intersect = df_valid.loc[df_valid['diff'].idxmin()]
print(f"Q3 - Precision & Recall intersect at threshold: {best_intersect.threshold:.2f}")
print(f"     Precision: {best_intersect.precision:.3f}, Recall: {best_intersect.recall:.3f}")

# Q4: maximum F1
best_f1 = df_scores.loc[df_scores.f1.idxmax()]
print(f"Q4 - Maximum F1 = {best_f1.f1:.4f} at threshold: {best_f1.threshold:.2f}")

# Question 5: 5-Fold Cross-Validation
print("\n" + "="*60)
print("QUESTION 5: 5-Fold CV standard deviation")
print("="*60)

def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    model.fit(X_train, y_train)
    return dv, model

def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred

kfold = KFold(n_splits=5, shuffle=True, random_state=1)
scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_tr = df_full_train.iloc[train_idx]
    df_vl = df_full_train.iloc[val_idx]
    y_tr = y_full_train[train_idx]
    y_vl = y_full_train[val_idx]

    dv, model = train(df_tr, y_tr, C=1.0)
    y_pred = predict(df_vl, dv, model)
    auc = roc_auc_score(y_vl, y_pred)
    scores.append(auc)

print(f"Fold AUCs: {[round(s, 4) for s in scores]}")
print(f"Mean AUC:  {np.mean(scores):.3f}")
print(f"Std:       {np.std(scores):.3f}")

# Question 6: Hyperparameter tuning for C
print("\n" + "="*60)
print("QUESTION 6: Best C")
print("="*60)

for C in [0.000001, 0.001, 1]:
    scores = []
    for train_idx, val_idx in kfold.split(df_full_train):
        df_tr = df_full_train.iloc[train_idx]
        df_vl = df_full_train.iloc[val_idx]
        y_tr = y_full_train[train_idx]
        y_vl = y_full_train[val_idx]

        dv, model = train(df_tr, y_tr, C=C)
        y_pred = predict(df_vl, dv, model)
        auc = roc_auc_score(y_vl, y_pred)
        scores.append(auc)

    print(f"C = {C:8}: mean = {np.mean(scores):.3f}, std = {np.std(scores):.3f}")
