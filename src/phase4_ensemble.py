# Phase 4: Ensemble & Calibration with GNN Integration
# File: src/phase4_ensemble.py

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import root_mean_squared_error


# ─── Paths ───────────────────────────────────────────────────────────────────
MATCHES_CSV   = os.path.join("data", "processed", "pubg_phase1_matches.csv")
ROLES_CSV     = os.path.join("data", "processed", "pubg_phase2_match_roles.csv")
EMBEDS_CSV    = os.path.join("data", "processed", "pubg_phase2_player_embeddings.csv")
GNN_PRED_CSV  = os.path.join("data", "processed", "pubg_phase5_gnn_preds.csv")
ENSEMBLE_OUT  = os.path.join("data", "processed", "pubg_phase4_ensemble_with_gnn.csv")
PERF_OUT      = os.path.join("outputs", "ensemble_performance_with_gnn.txt")
BOOST_DIR     = os.path.join("models", "boosters")
META_PATH     = os.path.join("models", "meta_model.pkl")
ISO_PATH      = os.path.join("models", "iso_reg.pkl")

# Ensure model dirs exist
os.makedirs(BOOST_DIR, exist_ok=True)
os.makedirs(os.path.dirname(META_PATH), exist_ok=True)

# ─── 1. Load & merge data ─────────────────────────────────────────────────────
df_matches = pd.read_csv(MATCHES_CSV)
df_roles   = pd.read_csv(ROLES_CSV)[["Id", "sessionId", "role_cluster"]]
df_embeds  = pd.read_csv(EMBEDS_CSV)

# Save identifiers for output
id_series      = df_matches["Id"]
match_series   = df_matches["matchId"]
session_series = df_matches["sessionId"]

# Merge features
df = (
    df_matches
    .merge(df_roles,  on=["Id", "sessionId"], how="left")
    .merge(df_embeds, on="Id",              how="left")
)

# ─── 2. Drop identifiers and unsupported columns ─────────────────────────────
for c in ["Id", "matchId", "sessionId", "groupId"]:
    if c in df.columns:
        df.drop(columns=c, inplace=True)

# ─── 3. Cast categorical features ───────────────────────────────────────────
if "game_phase" in df.columns:
    df["game_phase"] = df["game_phase"].astype("category")

# ─── 4. Filter out any invalid targets ────────────────────────────────────────
df = df[df["winPlacePerc"].notna() & np.isfinite(df["winPlacePerc"])]
df = df.reset_index(drop=True)

# ─── 5. Prepare feature matrix & target ─────────────────────────────────────
TARGET = "winPlacePerc"
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(float).values

# ─── 6. Sub-models per role_cluster with 5-fold OOF ──────────────────────────
group_key = "role_cluster"
modes     = X[group_key].unique()
oof_preds = pd.DataFrame(index=df.index, columns=modes, dtype=float)

def train_submodels():
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for mode in modes:
        mask   = X[group_key] == mode
        X_mode = X.loc[mask].drop(columns=[group_key])
        y_mode = y[mask]
        oof    = np.zeros(len(X_mode), dtype=float)

        for train_idx, val_idx in kf.split(X_mode):
            X_tr, X_val = X_mode.iloc[train_idx], X_mode.iloc[val_idx]
            y_tr, y_val = y_mode[train_idx], y_mode[val_idx]

            valid = np.isfinite(y_tr)
            X_tr, y_tr = X_tr.iloc[valid], y_tr[valid]

            dtr  = xgb.DMatrix(X_tr,  label=y_tr,  enable_categorical=True)
            dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
            bst  = xgb.train(
                {"objective": "reg:squarederror", "tree_method": "hist"},
                dtr, num_boost_round=100, verbose_eval=False
            )
            bst.save_model(os.path.join(BOOST_DIR, f"boost_{mode}.json"))
            oof[val_idx] = bst.predict(dval)

        oof_preds.loc[mask, mode] = oof
        rmse_mode = mean_squared_error(y_mode, oof, squared=False)
        print(f"Submodel {mode} OOF RMSE: {rmse_mode:.4f}")

train_submodels()

# ─── 7. Load GNN predictions and augment meta-features ───────────────────────
gnn_df  = pd.read_csv(GNN_PRED_CSV)[["Id", "gnn_pred"]]
gnn_map = gnn_df.set_index("Id")["gnn_pred"]
gnn_col = df_matches.loc[df.index, "Id"].map(gnn_map).fillna(0).values
meta_df = oof_preds.fillna(0).copy()
meta_df["gnn_pred"] = gnn_col

# ─── 8. Train Ridge meta-model ───────────────────────────────────────────────
meta_matrix = meta_df.values
meta_model  = Ridge(alpha=1.0)
meta_model.fit(meta_matrix, y)
joblib.dump(meta_model, META_PATH)
ensemble_oof = meta_model.predict(meta_matrix)

# ─── 9. Isotonic calibration ─────────────────────────────────────────────────
iso_reg        = IsotonicRegression(out_of_bounds="clip")
calibrated_pts = iso_reg.fit_transform(ensemble_oof, y)
joblib.dump(iso_reg, ISO_PATH)

# ─── 10. Evaluate performance ───────────────────────────────────────────────
rmse_raw = root_mean_squared_error(y, ensemble_oof)
rmse_cal = root_mean_squared_error(y, calibrated_pts)
print(f"Raw Ensemble+GNN OOF RMSE: {rmse_raw:.4f}")
print(f"Calibrated Ensemble+GNN RMSE: {rmse_cal:.4f}")

# ─── 11. Save outputs ───────────────────────────────────────────────────────
os.makedirs(os.path.dirname(PERF_OUT), exist_ok=True)
with open(PERF_OUT, "w") as f:
    f.write(f"Raw OOF RMSE: {rmse_raw:.4f}\n")
    f.write(f"Calibrated OOF RMSE: {rmse_cal:.4f}\n")

os.makedirs(os.path.dirname(ENSEMBLE_OUT), exist_ok=True)
out_df = df_matches.loc[df.index, ["Id", "matchId", "sessionId"]].copy()
out_df["pred_raw"] = ensemble_oof
out_df["pred_calibrated"] = calibrated_pts

out_df.to_csv(ENSEMBLE_OUT, index=False)
print(f"\u2714 Augmented ensemble predictions saved to {ENSEMBLE_OUT}")