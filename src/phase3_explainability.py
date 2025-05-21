# src/phase3_explainability.py

import os
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ─── Paths ───────────────────────────────────────────────────────────────
MATCHES_CSV   = os.path.join("data", "processed", "pubg_phase1_matches.csv")
ROLES_CSV     = os.path.join("data", "processed", "pubg_phase2_match_roles.csv")
EMBEDS_CSV    = os.path.join("data", "processed", "pubg_phase2_player_embeddings.csv")
SHAP_PLOT_OUT = os.path.join("outputs", "shap_summary.png")

# ─── 1. Load & merge features ───────────────────────────────────────────────
df_match = pd.read_csv(MATCHES_CSV)
df_roles = pd.read_csv(ROLES_CSV)[["Id", "sessionId", "role_cluster"]]
df_emb   = pd.read_csv(EMBEDS_CSV)

df = (
    df_match
      .merge(df_roles, on=["Id", "sessionId"], how="left")
      .merge(df_emb,   on="Id",             how="left")
)

# ─── 2. Prepare categorical & drop IDs ────────────────────────────────────
# Cast game_phase and role_cluster to categorical
if "game_phase" in df.columns:
    df["game_phase"] = df["game_phase"].astype("category")
df["role_cluster"] = df["role_cluster"].astype("category")

# Drop only pure identifiers; keep winPlacePerc as our target
drop_cols = ["Id", "groupId", "matchType", "matchId", "sessionId"]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# ─── 3. Drop rows missing target ───────────────────────────────────────────
df.dropna(subset=["winPlacePerc"], inplace=True)

# ─── 4. Split features & target ────────────────────────────────────────────
X = df.drop(columns=["winPlacePerc"])
y = df["winPlacePerc"]

# ─── 5. Train/test split ───────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─── 6. Create DMatrix with categorical support ────────────────────────────
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dtest  = xgb.DMatrix(X_test,  label=y_test,  enable_categorical=True)

# ─── 7. Train XGBoost regressor ────────────────────────────────────────────
params = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "eval_metric": "rmse"
}
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, "train"), (dtest, "test")],
    verbose_eval=False
)

# ─── 8. SHAP explanation ──────────────────────────────────────────────────
explainer  = shap.TreeExplainer(bst)
shap_values = explainer.shap_values(dtest)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
os.makedirs(os.path.dirname(SHAP_PLOT_OUT), exist_ok=True)
plt.savefig(SHAP_PLOT_OUT, bbox_inches="tight")
plt.close()

print(f"✔ SHAP summary plot saved to {SHAP_PLOT_OUT}")
