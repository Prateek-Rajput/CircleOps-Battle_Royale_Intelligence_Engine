# File: src/phase6_score_test_debug.py

import os
import sys
import joblib
import xgboost as xgb
import pandas as pd
import polars as pl
import numpy as np

# ─── Paths ─────────────────────────────────────────────────────────────────
RAW_TEST       = os.path.join('data','raw','test_v2.csv')
RAW_TEST_TMP   = os.path.join('data','raw','test_v2_with_wpp.csv')
TEST_INTERIM   = os.path.join('data','processed','test_phase1_interim.csv')
MATCHES_TEST   = os.path.join('data','processed','test_phase1_matches.csv')
ROLES_TEST     = os.path.join('data','processed','test_phase2_roles.csv')
EMBEDS_TEST    = os.path.join('data','processed','test_phase2_embeds.csv')
BOOST_DIR      = os.path.join('models','boosters')
META_PATH      = os.path.join('models','meta_model.pkl')
ISO_PATH       = os.path.join('models','iso_reg.pkl')
GNN_PRED_CSV   = os.path.join('data','processed','pubg_phase5_gnn_preds.csv')
OUTPUT_CSV     = os.path.join('data','processed','pubg_phase6_test_predictions.csv')

# ─── 1. Check model artifacts exist ──────────────────────────────────────────
if not os.path.isdir(BOOST_DIR) or not os.listdir(BOOST_DIR):
    raise FileNotFoundError(f"No boosters found in '{BOOST_DIR}'. Run Phase 4 first.")
if not os.path.exists(META_PATH) or not os.path.exists(ISO_PATH):
    raise FileNotFoundError(f"Meta-model or calibrator missing. Ensure they exist.")

# ─── 2. Prepare raw test for Phase 1 ─────────────────────────────────────────
tmp = pl.read_csv(RAW_TEST)
tmp = tmp.with_columns(pl.lit(0.5).alias('winPlacePerc'))
tmp.write_csv(RAW_TEST_TMP)

# ─── 3. Run Phase 1 & 2 pipelines on test set ───────────────────────────────
dirpath = os.path.dirname(__file__)
sys.path.append(dirpath)
import phase1_pipeline as p1
import phase2_role_embedding as p2

p1.RAW_PATH     = RAW_TEST_TMP
p1.INTERIM_PATH = TEST_INTERIM
p1.MATCHES_OUT  = MATCHES_TEST
p1.SESSIONS_OUT = os.path.join('data','processed','test_phase1_sessions.csv')
p1.run_phase1_polars()

p2.MATCHES_IN    = MATCHES_TEST
p2.ROLES_OUT     = ROLES_TEST
p2.PLAYER_EMBEDS = EMBEDS_TEST
p2.run_phase2_polars()

# ─── 4. Load and merge test features ────────────────────────────────────────
df_test = pd.read_csv(MATCHES_TEST)
roles   = pd.read_csv(ROLES_TEST)[['Id','sessionId','role_cluster']]
embeds  = pd.read_csv(EMBEDS_TEST)

# Debug: Show unique role clusters in test
print("\n🧠 Test role_cluster distribution:")
print(roles['role_cluster'].value_counts())

# Merge and drop placeholders
df = df_test.merge(roles, on=['Id','sessionId'], how='left')
df = df.merge(embeds, on='Id', how='left')
df = df.drop(columns=['winPlacePerc'], errors='ignore')
for c in ['matchId','sessionId','groupId']:
    df.drop(columns=c, inplace=True, errors='ignore')

# Prepare feature DataFrame
feature_cols = [c for c in df.columns if c not in ['Id','role_cluster']]
X_test = df[feature_cols].copy()
if 'game_phase' in X_test:
    X_test['game_phase'] = X_test['game_phase'].astype('category')
cluster_key = 'role_cluster'

# ─── 5. Generate sub-model predictions ──────────────────────────────────────
modes       = sorted([f for f in os.listdir(BOOST_DIR) if f.startswith('boost_')])
pred_matrix = np.zeros((len(df), len(modes)))

print("\n🚀 Scoring per booster:")
for i, fname in enumerate(modes):
    mode = int(fname.replace('boost_','').replace('.json',''))
    bst  = xgb.Booster()
    bst.load_model(os.path.join(BOOST_DIR, fname))
    mask = df[cluster_key] == mode
    print(f"  - Booster: {mode}, Players matched: {mask.sum()}")

    if mask.sum() > 0:
        X_sub = X_test.loc[mask]
        dmat  = xgb.DMatrix(X_sub, enable_categorical=True)
        pred_matrix[mask, i] = bst.predict(dmat)

# ─── 6. Append GNN predictions ──────────────────────────────────────────────
gnn_df   = pd.read_csv(GNN_PRED_CSV)[['Id','gnn_pred']]
gnn_map  = gnn_df.set_index('Id')['gnn_pred']
gnn_col  = df['Id'].map(gnn_map).fillna(0).values.reshape(-1,1)
pred_matrix = np.hstack([pred_matrix, gnn_col])

# ─── 7. Stack & calibrate ───────────────────────────────────────────────────
meta_model = joblib.load(META_PATH)
iso_reg    = joblib.load(ISO_PATH)
raw_preds  = meta_model.predict(pred_matrix)
cal_preds  = iso_reg.transform(raw_preds)

# ─── 8. Save predictions ───────────────────────────────────────────────────
out = pd.DataFrame({
    'Id': df['Id'],
    'pred_raw': raw_preds,
    'pred_calibrated': cal_preds
})
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
out.to_csv(OUTPUT_CSV, index=False)
print(f"\n✔ Test-set predictions saved to {OUTPUT_CSV}")
