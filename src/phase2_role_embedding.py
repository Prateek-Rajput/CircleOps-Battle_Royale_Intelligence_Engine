# src/phase2_role_embedding_polars.py

import os
import numpy as np
import polars as pl
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from gensim.models import Word2Vec

# ─── Paths ───────────────────────────────────────────────────────────────
MATCHES_IN    = os.path.join("data", "processed", "pubg_phase1_matches.csv")
ROLES_OUT     = os.path.join("data", "processed", "pubg_phase2_match_roles.csv")
PLAYER_EMBEDS = os.path.join("data", "processed", "pubg_phase2_player_embeddings.csv")

def run_phase2_polars(n_clusters: int = 5, w2v_size: int = 16, w2v_window: int = 3):
    # 1. Load match-level features
    df = pl.read_csv(MATCHES_IN)

    # 2. Ensure distance_covered
    if "distance_covered" not in df.columns:
        df = df.with_columns(
            (pl.col("walkDistance") + pl.col("rideDistance") + pl.col("swimDistance"))
            .alias("distance_covered")
        )

    # 3. Features for clustering
    feat_cols = ["kills","damageDealt","assists","revives",
                 "DBNOs","heals","boosts","distance_covered",
                 "engagement_density","momentum_score"]
    type_cols = [c for c in df.columns if c.startswith("matchType_")]
    all_feats = feat_cols + type_cols

    # 4. Build feature matrix
    X = df.select(all_feats).fill_null(0).to_numpy()

    # 5. Scale & cluster
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(Xs).astype(str)

    # 6. Attach and save match roles
    df = df.with_columns(pl.Series("role_cluster", labels))
    os.makedirs(os.path.dirname(ROLES_OUT), exist_ok=True)
    df.write_csv(ROLES_OUT)
    print(f"✔ Match roles saved: {ROLES_OUT}")

    # 7. Get sequences via pandas
    pdf = df.select(["Id","sessionId","role_cluster"]).to_pandas()
    pdf.sort_values(["Id","sessionId"], inplace=True)
    seqs = pdf.groupby("Id")["role_cluster"].apply(list).reset_index(name="roles")

    # 8. Train Word2Vec
    w2v = Word2Vec(
        sentences=seqs["roles"].tolist(),
        vector_size=w2v_size,
        window=w2v_window,
        min_count=1,
        workers=4,
        seed=42
    )

    # 9. Compute embeddings
    emb_dict = {"Id": [], **{f"emb_{i}": [] for i in range(w2v_size)}}
    for pid, roles in zip(seqs["Id"], seqs["roles"]):
        vecs = np.vstack([w2v.wv[r] for r in roles])
        mean_vec = vecs.mean(axis=0)
        emb_dict["Id"].append(pid)
        for i in range(w2v_size):
            emb_dict[f"emb_{i}"].append(mean_vec[i])

    emb_df = pd.DataFrame(emb_dict)

    # 10. Save player embeddings
    emb_df.to_csv(PLAYER_EMBEDS, index=False)
    print(f"✔ Player embeddings saved: {PLAYER_EMBEDS}")

if __name__ == "__main__":
    run_phase2_polars()
