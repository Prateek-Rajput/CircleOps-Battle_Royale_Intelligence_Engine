# File: notebooks/profile_role_clusters.py

import pandas as pd
import sys
sys.path.append("src")  # So it can find utils/map_role_names.py
from utils.map_role_names import assign_role_name

# Load match features with role_cluster info
matches = pd.read_csv("data/processed/pubg_phase2_match_roles.csv")

# Select key behavioral stats
features = [
    "kills", "damageDealt", "assists", "revives", "DBNOs",
    "heals", "boosts", "distance_covered", "engagement_density", "momentum_score"
]

# Group by role_cluster and calculate mean stats
role_profiles = (
    matches
    .groupby("role_cluster")[features]
    .mean()
    .round(2)
    .sort_index()
)

# Display role profiles
print("\n📊 Average Behavior per Role Cluster:")
print(role_profiles)
role_profiles["role_name"] = assign_role_name(role_profiles.index)

# Optional: Save for manual annotation or dashboard
role_profiles.to_csv("outputs/role_cluster_profiles.csv")
print("\n✔ Role profiles saved to outputs/role_cluster_profiles.csv")