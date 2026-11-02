import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# -------------------------------------------------
# 1. Load & prepare data
# -------------------------------------------------
url = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv"
df = pd.read_csv(url)

# Fill missing values with 0
df = df.fillna(0)

# Exact split from the homework
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val   = df_val.reset_index(drop=True)
df_test  = df_test.reset_index(drop=True)

y_train = df_train.fuel_efficiency_mpg.values
y_val   = df_val.fuel_efficiency_mpg.values
y_test  = df_test.fuel_efficiency_mpg.values

del df_train['fuel_efficiency_mpg']
del df_val['fuel_efficiency_mpg']
del df_test['fuel_efficiency_mpg']

# DictVectorizer
train_dicts = df_train.to_dict(orient='records')
dv = DictVectorizer(sparse=True)
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val.to_dict(orient='records')
X_val = dv.transform(val_dicts)

feature_names = dv.get_feature_names_out()

# -------------------------------------------------
# Question 1 – Decision tree (max_depth=1)
# -------------------------------------------------
dt = DecisionTreeRegressor(max_depth=1)
dt.fit(X_train, y_train)

print("=== Question 1 ===")
print(export_text(dt, feature_names=list(feature_names)))
# Look at the first split – map one-hot features back to original column
# (e.g. origin=USA → 'origin')

# -------------------------------------------------
# Question 2 – Random Forest baseline
# -------------------------------------------------
rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("\n=== Question 2 ===")
print(f"RMSE = {rmse:.4f}")

# -------------------------------------------------
# Question 3 – n_estimators tuning
# -------------------------------------------------
print("\n=== Question 3 ===")
scores = {}
for n in [10, 50, 100, 150]:
    rf = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    scores[n] = round(rmse, 3)
    print(f"n_estimators={n:3d}  RMSE={rmse:.4f}")

best_n = min(scores, key=scores.get)
print(f"Best n_estimators = {best_n}")

# -------------------------------------------------
# Question 4 – max_depth tuning (mean RMSE)
# -------------------------------------------------
print("\n=== Question 4 ===")
mean_scores = {}
for d in [10, 15, 20, 25]:
    rmses = []
    for n in [10, 50, 100, 150]:
        rf = RandomForestRegressor(n_estimators=n, max_depth=d,
                                   random_state=1, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        rmses.append(np.sqrt(mean_squared_error(y_val, y_pred)))
    mean_rmse = np.mean(rmses)
    mean_scores[d] = mean_rmse
    print(f"max_depth={d}  mean RMSE={mean_rmse:.4f}")

best_depth = min(mean_scores, key=mean_scores.get)
print(f"Best max_depth = {best_depth}")

# -------------------------------------------------
# Question 5 – Feature importance
# -------------------------------------------------
print("\n=== Question 5 ===")
rf = RandomForestRegressor(n_estimators=10, max_depth=20,
                           random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)

importances = pd.Series(rf.feature_importances_, index=feature_names)
# Keep only the 4 features asked in the question
candidates = ['vehicle_weight', 'horsepower', 'acceleration', 'engine_displacement']
# Some features may be one-hot encoded – take the max importance that belongs to each original column
imp = {}
for c in candidates:
    matching = [f for f in feature_names if f.startswith(c) or f == c]
    imp[c] = importances[matching].max() if matching else 0.0

print(pd.Series(imp).sort_values(ascending=False))
print("Most important =", max(imp, key=imp.get))

# -------------------------------------------------
# Question 6 – XGBoost eta
# -------------------------------------------------
print("\n=== Question 6 ===")
features = list(feature_names)
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=features)
watchlist = [(dtrain, 'train'), (dval, 'val')]

def train_xgb(eta):
    params = {
        'eta': eta,
        'max_depth': 6,
        'min_child_weight': 1,
        'objective': 'reg:squarederror',
        'nthread': 8,
        'seed': 1,
        'verbosity': 1,
    }
    model = xgb.train(params, dtrain, num_boost_round=100,
                      evals=watchlist, verbose_eval=False)
    y_pred = model.predict(dval)
    return np.sqrt(mean_squared_error(y_val, y_pred))

rmse_03 = train_xgb(0.3)
rmse_01 = train_xgb(0.1)
print(f"eta=0.3  RMSE={rmse_03:.4f}")
print(f"eta=0.1  RMSE={rmse_01:.4f}")

if abs(rmse_03 - rmse_01) < 1e-4:
    print("Both give equal value")
elif rmse_03 < rmse_01:
    print("Better eta = 0.3")
else:
    print("Better eta = 0.1")
