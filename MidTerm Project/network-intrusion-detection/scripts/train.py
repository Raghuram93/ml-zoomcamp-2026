from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# -------------------------------------------------
# Paths (works from any location when run in VS Code)
# -------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

# -------------------------------------------------
# 1. Load data
# -------------------------------------------------
columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]

df_train = pd.read_csv(DATA_DIR / "KDDTrain+.txt", names=columns)
df_test  = pd.read_csv(DATA_DIR / "KDDTest+.txt",  names=columns)

print(f"Train shape: {df_train.shape}")
print(f"Test  shape: {df_test.shape}")

# -------------------------------------------------
# 2. Create binary target
# -------------------------------------------------
df_train["target"] = (df_train["label"] != "normal").astype(int)
df_test["target"]  = (df_test["label"]  != "normal").astype(int)

df_train = df_train.drop(columns=["label", "difficulty"])
df_test  = df_test.drop(columns=["label", "difficulty"])

# -------------------------------------------------
# 3. Feature lists
# -------------------------------------------------
categorical = ["protocol_type", "service", "flag"]
numerical = [c for c in df_train.columns if c not in categorical + ["target"]]

def prepare(df):
    df = df.copy()
    df[categorical] = df[categorical].astype(str)
    return df

df_train = prepare(df_train)
df_test  = prepare(df_test)

# -------------------------------------------------
# 4. Split
# -------------------------------------------------
df_full_train, df_val = train_test_split(
    df_train, test_size=0.2, random_state=42, stratify=df_train["target"]
)

y_train = df_full_train["target"].values
y_val   = df_val["target"].values
y_test  = df_test["target"].values

# -------------------------------------------------
# 5. Vectorizer
# -------------------------------------------------
dv = DictVectorizer(sparse=False)

X_train = dv.fit_transform(df_full_train[categorical + numerical].to_dict(orient="records"))
X_val   = dv.transform(df_val[categorical + numerical].to_dict(orient="records"))
X_test  = dv.transform(df_test[categorical + numerical].to_dict(orient="records"))

# -------------------------------------------------
# 6. Models
# -------------------------------------------------
models = {
    "logistic": LogisticRegression(max_iter=1000, random_state=42),
    "decision_tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
    "xgboost": XGBClassifier(
        n_estimators=120, max_depth=6, learning_rate=0.1,
        random_state=42, n_jobs=-1, eval_metric="logloss"
    )
}

print("\n===== Model Comparison =====")
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    acc = accuracy_score(y_val, y_pred >= 0.5)
    results[name] = {"model": model, "auc": auc}
    print(f"{name:15} | AUC: {auc:.4f} | Accuracy: {acc:.4f}")

# -------------------------------------------------
# 7. Best model + Test evaluation
# -------------------------------------------------
best_name = max(results, key=lambda k: results[k]["auc"])
best_model = results[best_name]["model"]
print(f"\nBest model → {best_name}")

y_pred_test = best_model.predict_proba(X_test)[:, 1]
print(f"Test AUC  → {roc_auc_score(y_test, y_pred_test):.4f}")
print("\nClassification Report (Test set):")
print(classification_report(y_test, (y_pred_test >= 0.5).astype(int)))

# -------------------------------------------------
# 8. Save
# -------------------------------------------------
joblib.dump({
    "model": best_model,
    "dv": dv,
    "categorical": categorical,
    "numerical": numerical
}, MODEL_DIR / "model.bin")

print(f"\nModel saved to → {MODEL_DIR / 'model.bin'}")