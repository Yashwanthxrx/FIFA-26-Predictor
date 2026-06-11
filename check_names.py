import pandas as pd
import os

# Define Paths
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
MATCHES_FILE = os.path.join(DATA_DIR, "matches_1930_2022.csv")
ELO_FILE = os.path.join(DATA_DIR, "elo_ratings_wc2026.csv")

matches_df = pd.read_csv(MATCHES_FILE)
elo_df = pd.read_csv(ELO_FILE)

match_teams = set(matches_df['home_team'].dropna().unique()) | set(matches_df['away_team'].dropna().unique())
elo_teams = set(elo_df['country'].dropna().unique())

missing_in_elo = match_teams - elo_teams

print("Teams in matches but missing in ELO dataset:")
for team in sorted(list(missing_in_elo)):
    print(f"- {team}")
