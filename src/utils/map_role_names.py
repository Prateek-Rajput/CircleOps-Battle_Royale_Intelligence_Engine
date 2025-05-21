# File: utils/map_role_names.py

# Dictionary to map numeric clusters to human-readable role names
ROLE_NAME_MAP = {
    0: "Passive Camper",
    1: "Aggressive Slayer",
    2: "Support Combatant",
    3: "Midgame Scout",
    4: "Clutch God"
}

def assign_role_name(role_cluster_col):
    """
    Maps a column of role_cluster integers to their corresponding role names.

    Parameters:
    role_cluster_col (pd.Series): Series of integer cluster labels

    Returns:
    pd.Series: Series of descriptive role names
    """
    return role_cluster_col.map(ROLE_NAME_MAP)

# Example usage:
# df["role_name"] = assign_role_name(df["role_cluster"])
