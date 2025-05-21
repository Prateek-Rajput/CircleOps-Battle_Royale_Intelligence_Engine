# File: notebooks/identify_predicted_winners.py

import pandas as pd



# ─── Load Data ─────────────────────────────────────────────────────────────
PRED_PATH   = "data/processed/pubg_phase6_test_predictions.csv"
MATCH_PATH  = "data/processed/test_phase1_matches.csv"

preds   = pd.read_csv(PRED_PATH)
matches = pd.read_csv(MATCH_PATH)[["Id", "matchId"]]

# ─── Merge matchId into predictions ─────────────────────────────────────────
df = preds.merge(matches, on="Id", how="left")

# ─── Sort by predicted win placement per match ─────────────────────────────
df_sorted = df.sort_values(["matchId", "pred_calibrated"], ascending=[True, False])

# ─── Get top predicted winner per match ─────────────────────────────────────
winners = df_sorted.groupby("matchId").first().reset_index()

# ─── Output Sample ─────────────────────────────────────────────────────────
print("\n🏆 Top Predicted Players per Match:")
print(winners[["matchId", "Id", "pred_calibrated"]].head())

# ─── (Optional) Save to CSV ─────────────────────────────────────────────────
WINNER_CSV = "data/processed/predicted_match_winners.csv"
winners.to_csv(WINNER_CSV, index=False)
print(f"\n✔ Predicted winners saved to: {WINNER_CSV}")
