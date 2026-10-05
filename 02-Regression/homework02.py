import pandas as pd
import numpy as np

# --------------------------------------------------
# Load data
# --------------------------------------------------
url = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv"
df = pd.read_csv(url)

# Keep only required columns
cols = ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year', 'fuel_efficiency_mpg']
df = df[cols].copy()

#Question 1
missing_col = df.columns[df.isnull().sum() > 0][0]
print("Q1:", missing_col)

#Question 2
print("Q2:", df['horsepower'].median())

#helper functions
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX_inv = np.linalg.inv(X.T @ X)
    w = XTX_inv @ X.T @ y
    return w[0], w[1:]

def train_linear_regression_reg(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T @ X
    XTX = XTX + r * np.eye(XTX.shape[0])
    w = np.linalg.inv(XTX) @ X.T @ y
    return w[0], w[1:]

def rmse(y, y_pred):
    return np.sqrt(((y - y_pred) ** 2).mean())

def prepare_X(df, fill_value=0):
    return df.fillna(fill_value).values

def split_data(df, seed=42):
    n = len(df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test

    np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:n_train]].reset_index(drop=True)
    df_val   = df.iloc[idx[n_train:n_train+n_val]].reset_index(drop=True)
    df_test  = df.iloc[idx[n_train+n_val:]].reset_index(drop=True)

    y_train = df_train['fuel_efficiency_mpg'].values
    y_val   = df_val['fuel_efficiency_mpg'].values
    y_test  = df_test['fuel_efficiency_mpg'].values

    df_train = df_train.drop(columns=['fuel_efficiency_mpg'])
    df_val   = df_val.drop(columns=['fuel_efficiency_mpg'])
    df_test  = df_test.drop(columns=['fuel_efficiency_mpg'])

    return df_train, df_val, df_test, y_train, y_val, y_test

#Question 3
df_train, df_val, _, y_train, y_val, _ = split_data(df, seed=42)

# fill with 0
X_train = prepare_X(df_train, 0)
w0, w = train_linear_regression(X_train, y_train)
y_pred = w0 + prepare_X(df_val, 0) @ w
rmse_0 = round(rmse(y_val, y_pred), 3)

# fill with mean
mean_hp = df_train['horsepower'].mean()
X_train = prepare_X(df_train, mean_hp)
w0, w = train_linear_regression(X_train, y_train)
y_pred = w0 + prepare_X(df_val, mean_hp) @ w
rmse_mean = round(rmse(y_val, y_pred), 3)

print(f"Q3: 0 → {rmse_0} | mean → {rmse_mean}")
if rmse_0 < rmse_mean:
    print("   → Better: With 0")
elif rmse_mean < rmse_0:
    print("   → Better: With mean")
else:
    print("   → Both are equally good")

#Question 4
X_train = prepare_X(df_train, 0)
X_val   = prepare_X(df_val, 0)

best_r, best_score = None, float('inf')
print("Q4:")
for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
    w0, w = train_linear_regression_reg(X_train, y_train, r=r)
    score = round(rmse(y_val, w0 + X_val @ w), 4)
    print(f"  r={r:<6} → {score}")
    if score < best_score:
        best_score = score
        best_r = r
print(f"   → Best r: {best_r}")

#Question 5
scores = []
for seed in range(10):
    df_tr, df_v, _, y_tr, y_v, _ = split_data(df, seed=seed)
    w0, w = train_linear_regression(prepare_X(df_tr, 0), y_tr)
    scores.append(rmse(y_v, w0 + prepare_X(df_v, 0) @ w))

print("Q5:", round(np.std(scores), 3))

#Question 6
df_train, df_val, df_test, y_train, y_val, y_test = split_data(df, seed=9)

df_full = pd.concat([df_train, df_val])
y_full  = np.concatenate([y_train, y_val])

w0, w = train_linear_regression_reg(prepare_X(df_full, 0), y_full, r=0.001)
test_rmse = rmse(y_test, w0 + prepare_X(df_test, 0) @ w)

print("Q6:", round(test_rmse, 3))
