# File: notebooks/analyze_predicted_winners.py

import pandas as pd
import sys
sys.path.append("src")  # adjust if you're running from notebooks
from utils.map_role_names import assign_role_name
from utils.map_role_names import ROLE_NAME_MAP
import matplotlib.pyplot as plt

# ─── Load Data ─────────────────────────────────────────────────────────────
winner_path = "data/processed/predicted_match_winners.csv"
roles_path  = "data/processed/test_phase2_roles.csv"
match_path  = "data/processed/test_phase1_matches.csv"
train_path  = "data/processed/pubg_phase1_matches.csv"

winners = pd.read_csv(winner_path)
roles   = pd.read_csv(roles_path)[["Id", "role_cluster"]]
match   = pd.read_csv(match_path)[["Id", "groupId", "matchId"]]
train   = pd.read_csv(train_path)[["Id", "matchId", "winPlacePerc"]]

# ─── Merge Roles and Squad Info ────────────────────────────────────────────
winners = winners.merge(roles, on="Id", how="left")
winners["role_name"] = assign_role_name(winners["role_cluster"])
winners = winners.merge(match, on="Id", how="left", suffixes=("", "_dup"))
winners = winners.drop(columns=[col for col in winners.columns if col.endswith("_dup")])



# ─── Role Distribution of Winners ──────────────────────────────────────────
role_counts = winners["role_cluster"].value_counts().sort_index()
role_counts_named = role_counts.rename(index=ROLE_NAME_MAP)
plt.figure(figsize=(8, 5))
role_counts_named.plot(kind="bar", color="skyblue")
plt.title("🏆 Match Wins per Player Role")
plt.xlabel("Role Cluster")
plt.ylabel("Number of Wins")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# ─── Squad Analysis: Top Winning Squads ────────────────────────────────────
squad_wins = winners.groupby("groupId").size().reset_index(name="wins")
top_squads = squad_wins.sort_values("wins", ascending=False).head()
print("\n🎯 Top Winning Squads:")
print(top_squads)

# ─── Validation: Ground Truth Winners (if available) ───────────────────────
true_winners = train[train["winPlacePerc"] == 1.0]
true_counts = true_winners["matchId"].value_counts().reset_index()
true_counts.columns = ["matchId", "true_winner_count"]
print("\n✅ Sample True Winners from Train Set:")
print(true_winners.head())
