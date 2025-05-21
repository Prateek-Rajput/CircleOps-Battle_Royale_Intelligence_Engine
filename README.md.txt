# PUBG Intelligence Engine 🎯  
End-to-End Player Modeling, Role Classification, and Squad-Level Prediction Pipeline

---

## 🚀 Project Overview

The PUBG Intelligence Engine is a comprehensive machine learning system that processes raw player match data and delivers strategic insights across multiple layers:  
- **Player behavior modeling (via role clustering and embeddings)**  
- **Win prediction using XGBoost, Ridge, and Graph Neural Networks**  
- **Squad synergy modeling through PyTorch Geometric**  
- **Explainability with SHAP, role-based analytics, and calibrated scoring**

Built using **Polars**, **XGBoost**, **GNNs**, **SHAP**, and **Word2Vec**, this pipeline simulates real-world challenges in competitive multiplayer analytics.

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
├── LICENSE
└── README.md
```

---

## ⚙️ Technical Stack

- Python 3.11+
- `polars` for high-speed ETL
- `scikit-learn`, `xgboost`, `joblib`, `shap`, `gensim`, `torch`, `torch-geometric`

---

## 📊 Modeling Pipeline (6 Phases)

| Phase | Module                       | Output                                       |
|-------|------------------------------|----------------------------------------------|
| 1️⃣    | Session Modeling              | Cleaned match/session data                   |
| 2️⃣    | Role Clustering & Embedding  | Player roles + Word2Vec embeddings           |
| 3️⃣    | Explainability               | SHAP feature importance visualization        |
| 4️⃣    | Role-based Ensemble          | XGBoost submodels + Ridge meta-model         |
| 5️⃣    | Squad Graph Model            | GNN predictions via PyTorch Geometric        |
| 6️⃣    | Scoring API                  | Calibrated predictions on test set           |

---

## 🧠 Player Roles (Cluster Mapping)

| Cluster | Role Name          |
|---------|--------------------|
| 0       | Passive Camper     |
| 1       | Aggressive Slayer  |
| 2       | Support Combatant  |
| 3       | Midgame Scout      |
| 4       | Clutch God         |

---

## 📈 Results

- 📊 **SHAP** reveals killPlace, distance, and momentum drive win outcomes
- 🧠 **Role Cluster 0 (Passive Campers)** win most matches — smart survival > brute force
- 🔁 **GNN + Ridge Ensemble** improves generalization after calibration

---

## 📦 How to Run

```bash
# Phase 1: Preprocess raw match data
python src/phase1_pipeline.py

# Phase 2: Role clustering + Word2Vec embedding
python src/phase2_role_embedding.py

# (Optional) Phase 3: SHAP explainability
python src/phase3_explainability.py

# Phase 4: Train ensemble model
python src/phase4_ensemble.py

# Phase 5: Train squad-based GNN model
python src/phase5_squad_gnn.py

# Phase 6: Score test set
python src/phase6_score_test.py
```

---

## 🧪 Post-Modeling Analysis

```bash
python notebooks/identify_pred_winners.py
python notebooks/profile_role_clusters.py
python notebooks/analyze_predicted_winners.py
```

---

## 💡 Key Takeaways

- Multi-stage pipelines reflect real competitive game data workflows
- Role-aware modeling leads to more interpretable predictions
- Squad-based GNNs enhance performance prediction via teammate dynamics

---

## 📬 Author

**Prateek Rajput**  
Data Scientist | Gaming & ML Enthusiast  
🔗 [LinkedIn](https://www.linkedin.com/in/your-profile/)  
🕹️ Built with passion for competitive gaming data

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
