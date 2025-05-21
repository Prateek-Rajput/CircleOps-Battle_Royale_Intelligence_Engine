# PUBG Intelligence Engine 🎯  
End-to-End Player Modeling, Role Classification, and Squad-Level Prediction Pipeline

---

## 🚀 Project Overview

The PUBG Intelligence Engine is a full-stack machine learning pipeline built to analyze competitive player behavior, model squad dynamics, and predict win placements using gameplay telemetry. It combines role discovery, graph learning, ensemble modeling, and explainability into one cohesive system.

---

## 🔍 Analysis & Insights

### 🎯 Match Winner Role Distribution
- **Passive Campers (Cluster 0)** win most matches — they focus on survival and positioning.
- **Midgame Scouts (Cluster 3)** often rotate early and survive long enough to place well.
- **Clutch Gods (Cluster 4)** dominate when present, but are rare.
- **Aggressive Slayers (Cluster 1)** deal high damage but win fewer matches — likely due to risky combat.
- **Support Combatants (Cluster 2)** had the lowest win rate — they help squads but don’t close out games.

### 📊 Strategic Insight
> The best match predictors were survival time, mobility, and kill efficiency — not raw aggression.  
> **Careful movement and zone control beat pure combat.**

---

## 🧠 How Were Player Roles (Clusters) Formed?

- A subset of **combat, mobility, and support stats** (kills, damage, heals, distance) was used to cluster players.
- Clustering was performed using **k-Means (k=5)** after standardization.
- These clusters were then assigned behavioral labels using feature averages and SHAP interpretation.

| Cluster | Role Name           | Description |
|---------|---------------------|-------------|
| 0       | Passive Camper      | Plays safe, low combat |
| 1       | Aggressive Slayer   | High kill volume, fast play |
| 2       | Support Combatant   | High revives, low aggression |
| 3       | Midgame Scout       | Moderate stats, rotates early |
| 4       | Clutch God          | Explosive stats, rare but dominant |

---

## 📁 Project Structure

```
pubg-intelligence-engine/
├── data/
│   ├── raw/                  ← train_v2.csv, test_v2.csv
│   ├── processed/            ← phase outputs (matches, roles, predictions)
│   └── interim/              ← cleaned interim files
├── notebooks/
│   ├── analyze_predicted_winners.py
│   ├── identify_pred_winners.py
│   ├── profile_role_clusters.py
├── outputs/
│   ├── shap_summary.png
│   ├── ensemble_performance_with_gnn.txt
│   └── role_cluster_profiles.csv
├── models/                   ← saved models and boosters
├── src/
│   ├── phase1_pipeline.py
│   ├── phase2_role_embedding.py
│   ├── phase3_explainability.py
│   ├── phase4_ensemble.py
│   ├── phase5_squad_gnn.py
│   ├── phase6_score_test.py
│   └── utils/
│       └── map_role_names.py
├── .gitignore
├── requirements.txt
└── README.md
```


---

## ⚙️ Technical Stack

- Python 3.11+
- `polars` for high-speed ETL
- `scikit-learn`, `xgboost`, `shap`, `gensim`, `torch`, `torch-geometric`

---

## 📊 Modeling Pipeline (6 Phases)

| Phase | Module                    | Output                             |
|-------|---------------------------|-------------------------------------|
| 1️⃣    | Session Modeling           | Cleaned gameplay + session metrics |
| 2️⃣    | Role Clustering & Embedding | Role assignments + Word2Vec vectors |
| 3️⃣    | Explainability            | SHAP-based feature insights         |
| 4️⃣    | Ensemble Modeling         | Role-based XGBoost + Ridge stack    |
| 5️⃣    | Squad Graph Modeling      | PyG-based squad synergy predictions |
| 6️⃣    | Test Set Scoring          | Calibrated match outcome predictions|

---

## 📈 Key Results

- 🧠 SHAP: `killPlace`, `walkDistance`, and `momentum_score` are top predictors
- 📊 Clustering + role names enabled deeper interpretation of squad behaviors
- 📈 Ridge + GNN ensemble improved RMSE over single models
- ✅ Calibration ensured valid probability outputs between [0, 1]

---

## 📦 How to Run

```bash
# Phase 1–6 pipeline
python src/phase1_pipeline.py
python src/phase2_role_embedding.py
python src/phase3_explainability.py
python src/phase4_ensemble.py
python src/phase5_squad_gnn.py
python src/phase6_score_test.py
```

```bash
# Final analysis
python notebooks/identify_pred_winners.py
python notebooks/profile_role_clusters.py
python notebooks/analyze_predicted_winners.py
```

---

## 📬 Author

**Prateek Rajput**  
Data Scientist | Gaming & ML Enthusiast  
🔗 [LinkedIn](https://www.linkedin.com/in/prateek-rajput-b802b0169/)  
🕹️ Built with passion for competitive gaming data

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
