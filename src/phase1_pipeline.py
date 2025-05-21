# src/phase1_pipeline_polars.py

import polars as pl
import os

# ─── Paths ─────────────────────────────────────────────────────────────────
RAW_PATH     = os.path.join("data", "raw",     "train_v2.csv")
INTERIM_PATH = os.path.join("data", "interim", "pubg_cleaned_phase1.csv")
MATCHES_OUT  = os.path.join("data", "processed","pubg_phase1_matches.csv")
SESSIONS_OUT = os.path.join("data", "processed","pubg_phase1_sessions.csv")

def run_phase1_polars():
    # 1. Load & drop NAs
    df = pl.read_csv(RAW_PATH)
    df = df.drop_nulls(subset=["matchDuration","Id","matchId","matchType"])
    print(f"✔ After dropna: {df.shape}")

    # 2. Save interim snapshot
    os.makedirs(os.path.dirname(INTERIM_PATH), exist_ok=True)
    df.write_csv(INTERIM_PATH)
    print(f"✔ Interim saved: {INTERIM_PATH}")

    # 3. Vectorized game_phase inference
    df = df.with_columns(
        pl.when((pl.col("walkDistance") < 100) & (pl.col("kills") == 0))
          .then(pl.lit("early_loot"))
          .when((pl.col("kills") > 0) & (pl.col("damageDealt") < 100))
          .then(pl.lit("rotate_aggressive"))
          .when(pl.col("damageDealt") >= 300)
          .then(pl.lit("end_clutch"))
          .otherwise(pl.lit("midgame_safe"))
          .alias("game_phase")
    )

    # 4. Core match-level features
    df = df.with_columns([
        (pl.col("heals") + pl.col("boosts")).alias("health_used"),
        (pl.col("walkDistance") + pl.col("rideDistance") + pl.col("swimDistance"))
            .alias("distance_covered"),
        ((pl.col("damageDealt") + pl.col("kills") * 100)
          / (pl.col("matchDuration") / 60)).alias("engagement_density"),
        (pl.col("killStreaks") * 2 + pl.col("damageDealt") * 0.05)
            .alias("momentum_score")
    ])

    # 5. One-hot encode matchType via to_dummies
    df = df.to_dummies(["matchType"])

    # 6. Rolling stats (damage & kills) per player
    df = df.sort(["Id", "matchId"])
    df = df.with_columns([
        pl.col("damageDealt")
          .rolling_mean(window_size=5, min_samples=1)
          .over("Id")
          .alias("rolling_damage"),
        pl.col("kills")
          .rolling_mean(window_size=5, min_samples=1)
          .over("Id")
          .alias("rolling_kills")
    ])

    # 7. Session fallback: one session per match
    df = df.with_columns(pl.col("matchId").alias("sessionId"))

    # 8. Session-level summary (one row per match)
    session_stats = (
        df
        .group_by(["Id", "sessionId"])
        .agg([
            pl.col("matchId").n_unique().alias("matchesInSession"),
            pl.col("kills").mean().alias("avg_kills"),
            pl.col("damageDealt").mean().alias("avg_damage")
        ])
    )

    # 9. Save outputs
    os.makedirs(os.path.dirname(MATCHES_OUT), exist_ok=True)
    df.write_csv(MATCHES_OUT)
    os.makedirs(os.path.dirname(SESSIONS_OUT), exist_ok=True)
    session_stats.write_csv(SESSIONS_OUT)
    print("✔ Phase 1 complete with Polars")

if __name__ == "__main__":
    run_phase1_polars()
