# File: analyze_phase6_output.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# ─── Load Phase 6 Output ─────────────────────────────────────────────────
PHASE6_CSV = "data/processed/pubg_phase6_test_predictions.csv"
df = pd.read_csv(PHASE6_CSV)

# ─── Summary Stats ──────────────────────────────────────────────────────
print("\n📊 Prediction Summary:")
print(df[["pred_raw", "pred_calibrated"]].describe())
print("\n🔗 Correlation:", df["pred_raw"].corr(df["pred_calibrated"]))

# ─── Histograms ─────────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
plt.hist(df["pred_raw"], bins=50, alpha=0.5, label="Raw", color="blue")
plt.hist(df["pred_calibrated"], bins=50, alpha=0.5, label="Calibrated", color="orange")
plt.title("Phase 6 Test Prediction Distribution: Raw vs Calibrated")
plt.xlabel("winPlacePerc")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ─── Outliers ───────────────────────────────────────────────────────────
print("\n⚠️  High Calibrated Predictions (> 1.0):")
print(df[df["pred_calibrated"] > 1.0].head())

print("\n⚠️  Low Calibrated Predictions (< 0.1):")
print(df[df["pred_calibrated"] < 0.1].head())
