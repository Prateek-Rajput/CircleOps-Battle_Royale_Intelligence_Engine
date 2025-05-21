# File: analyze_phase4_output.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# ─── Load Ensemble Output ─────────────────────────────────────────────────
ENSEMBLE_CSV = "data/processed/pubg_phase4_ensemble_with_gnn.csv"
df = pd.read_csv(ENSEMBLE_CSV)

# ─── Summary Stats ───────────────────────────────────────────────────────
print("\n📊 Prediction Summary:")
print(df[["pred_raw", "pred_calibrated"]].describe())
print("\n🔗 Correlation:", df["pred_raw"].corr(df["pred_calibrated"]))

# ─── Histograms ──────────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
plt.hist(df["pred_raw"], bins=50, alpha=0.5, label="Raw")
plt.hist(df["pred_calibrated"], bins=50, alpha=0.5, label="Calibrated")
plt.title("Prediction Distribution: Raw vs Calibrated")
plt.xlabel("winPlacePerc")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ─── Outliers ────────────────────────────────────────────────────────────
print("\n⚠️  High Predictions:")
print(df[df["pred_calibrated"] > 1.0].head())

print("\n⚠️  Low Predictions:")
print(df[df["pred_calibrated"] < 0.1].head())

# ─── Optional: Merge Ground Truth and Compute RMSE ───────────────────────
try:
    df_truth = pd.read_csv("data/processed/pubg_phase1_matches.csv", usecols=["Id", "winPlacePerc"])
    merged = df.merge(df_truth, on="Id", how="left")
    rmse = mean_squared_error(merged["winPlacePerc"], merged["pred_calibrated"], squared=False)
    print(f"\n✅ Calibrated RMSE vs Ground Truth: {rmse:.4f}")
except Exception as e:
    print("\n⚠️  Skipping RMSE check (ground truth not found or mismatch):", str(e))
