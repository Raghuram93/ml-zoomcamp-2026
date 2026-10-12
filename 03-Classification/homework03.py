import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

# ============================================================
# 1. Load data
# ============================================================
url = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv"
df = pd.read_csv(url)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# ============================================================
# Data preparation (as required)
# ============================================================
# Identify categorical and numerical columns
categorical = ['lead_source', 'industry', 'employment_status', 'location']
numerical = ['annual_income', 'number_of_courses_viewed', 'interaction_count', 'lead_score']

# Fill missing values
for col in categorical:
    df[col] = df[col].fillna('NA')

for col in numerical:
    df[col] = df[col].fillna(0.0)

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# Question 1: Mode of industry

mode_industry = df['industry'].mode()[0]
print("\n=== Question 1 ===")
print("Most frequent observation (mode) for industry:", mode_industry)
# Options: NA / technology / healthcare / retail


# Question 2: Biggest correlation among numerical features

corr = df[numerical].corr()
print("\n=== Question 2 ===")
print("Correlation matrix:")
print(corr.round(3))

# Check the specific pairs
pairs = [
    ('interaction_count', 'lead_score'),
    ('number_of_courses_viewed', 'lead_score'),
    ('number_of_courses_viewed', 'interaction_count'),
    ('annual_income', 'interaction_count')
]

print("\nCorrelations of the given pairs:")
for a, b in pairs:
    print(f"{a} – {b}: {corr.loc[a, b]:.4f}")


# Split the data

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

# Separate target
y_train = df_train['converted'].values
y_val = df_val['converted'].values
y_test = df_test['converted'].values

# Remove target from dataframes
del df_train['converted']
del df_val['converted']
del df_test['converted']
del df_full_train['converted']

print("\nTrain size:", len(df_train))
print("Val size:", len(df_val))
print("Test size:", len(df_test))


# Question 3: Mutual information (training set only)

def calculate_mi(series):
    return mutual_info_score(series, y_train)

mi_scores = df_train[categorical].apply(calculate_mi)
mi_scores = mi_scores.round(2).sort_values(ascending=False)

print("\n=== Question 3 ===")
print("Mutual information scores (rounded to 2 decimals):")
print(mi_scores)
# Options: industry / location / lead_source / employment_status


# Question 4: Logistic Regression accuracy
 
# One-hot encoding with DictVectorizer
train_dicts = df_train[categorical + numerical].to_dict(orient='records')
val_dicts = df_val[categorical + numerical].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print("\n=== Question 4 ===")
print(f"Accuracy on validation set (rounded to 2 decimals): {acc:.2f}")# Options: 0.55 / 0.65 / 0.75 / 0.85


# Question 5: Feature elimination (least useful feature)
 
features = categorical + numerical
original_acc = acc   # without rounding

diffs = {}
for feat in ['lead_source', 'number_of_courses_viewed', 'interaction_count']:
    # Remove the feature
    feats_without = [f for f in features if f != feat]
    
    train_dicts_wo = df_train[feats_without].to_dict(orient='records')
    val_dicts_wo = df_val[feats_without].to_dict(orient='records')
    
    dv_wo = DictVectorizer(sparse=False)
    X_train_wo = dv_wo.fit_transform(train_dicts_wo)
    X_val_wo = dv_wo.transform(val_dicts_wo)
    
    model_wo = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
    model_wo.fit(X_train_wo, y_train)
    
    y_pred_wo = model_wo.predict(X_val_wo)
    acc_wo = accuracy_score(y_val, y_pred_wo)
    
    diff = original_acc - acc_wo
    diffs[feat] = diff
    print(f"Without '{feat}': accuracy = {acc_wo:.6f}, difference = {diff:.6f}")

print("\n=== Question 5 ===")
print("Differences:", diffs)
smallest = min(diffs, key=lambda k: abs(diffs[k]))  # smallest absolute difference
print("Feature with the smallest difference:", smallest)
# Note: difference does not have to be positive


# Question 6: Regularized logistic regression (different C)

C_values = [0.000001, 0.00001, 0.0001, 0.001]
results = {}

for C in C_values:
    model_c = LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=42)
    model_c.fit(X_train, y_train)
    y_pred_c = model_c.predict(X_val)
    acc_c = accuracy_score(y_val, y_pred_c)
    results[C] = round(acc_c, 3)
    print(f"C={C}: accuracy = {results[C]}")

print("\n=== Question 6 ===")
best_C = min([c for c, a in results.items() if a == max(results.values())])  # smallest C if ties
print("Best C (smallest if ties):", best_C)
# Options: 0.000001 / 0.00001 / 0.0001 / 0.001
