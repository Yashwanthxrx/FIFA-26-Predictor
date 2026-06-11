import pandas as pd
import numpy as np
import os

# Define Paths
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
MATCHES_FILE = os.path.join(DATA_DIR, "matches_1930_2022.csv")
ELO_FILE = os.path.join(DATA_DIR, "elo_ratings_wc2026.csv")
FIFA_RANK_FILE = os.path.join(DATA_DIR, "fifa_ranking_2026-06-08.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "processed_dataset.csv")

def prepare_data():
    print("Step 1: Loading datasets...")
    # Load the historical matches. We specify columns we care about to keep it simple.
    matches_df = pd.read_csv(MATCHES_FILE)
    
    # Check the columns and select only what we need for a beginner model
    # home_team, away_team, home_score, away_score, Year, Host
    cols_to_keep = ['home_team', 'away_team', 'home_score', 'away_score', 'Year', 'Host']
    # Filter only if these columns exist to avoid errors
    cols = [c for c in cols_to_keep if c in matches_df.columns]
    matches = matches_df[cols].copy()
    
    # Drop any rows with missing scores (if a match hasn't happened or data is bad)
    matches = matches.dropna(subset=['home_score', 'away_score'])
    
    # Map country names so they match the ELO dataset
    name_mapping = {
        'IR Iran': 'Iran',
        'Korea Republic': 'South Korea',
        'Czech Republic': 'Czechia',
        'Trkiye': 'Turkey',
        'Türkiye': 'Turkey',
        "Cte d'Ivoire": 'Ivory Coast',
        "Côte d'Ivoire": 'Ivory Coast',
        'United States': 'United States'
    }
    matches['home_team'] = matches['home_team'].replace(name_mapping)
    matches['away_team'] = matches['away_team'].replace(name_mapping)
    matches['Host'] = matches['Host'].replace(name_mapping)

    # Feature Engineering: True Home Advantage
    # If the home team is the Host, they have a massive crowd/travel advantage
    matches['is_home_advantage'] = (matches['home_team'] == matches['Host']).astype(int)

    
    print(f"Loaded {len(matches)} historical matches.")

    print("Step 2: Defining the Target Variable (Who won?)")
    # In Machine Learning, the "target" is what we want to predict.
    # We will create a column called 'target'.
    # Let's say: 0 = Away Team Wins, 1 = Draw, 2 = Home Team Wins
    
    def get_outcome(row):
        if row['home_score'] > row['away_score']:
            return 2 # Home Win
        elif row['home_score'] == row['away_score']:
            return 1 # Draw
        else:
            return 0 # Away Win
            
    matches['target'] = matches.apply(get_outcome, axis=1)
    
    print("Step 3: Loading ELO Ratings (Team Strength)")
    # ELO ratings measure how strong a team is. A higher ELO means a stronger team.
    # The ELO dataset has columns: year, country, rating
    elo_df = pd.read_csv(ELO_FILE)
    
    # We want to merge the ELO rating for the Home Team in the specific match Year
    home_elo = elo_df[['year', 'country', 'rating']].rename(
        columns={'year': 'Year', 'country': 'home_team', 'rating': 'home_elo'}
    )
    
    # We also want the ELO rating for the Away Team in that same Year
    away_elo = elo_df[['year', 'country', 'rating']].rename(
        columns={'year': 'Year', 'country': 'away_team', 'rating': 'away_elo'}
    )

    print("Step 4: Merging ELO Ratings with Matches")
    # Merge home ELO
    matches = pd.merge(matches, home_elo, on=['Year', 'home_team'], how='left')
    # Merge away ELO
    matches = pd.merge(matches, away_elo, on=['Year', 'away_team'], how='left')
    
    # Sometimes team names don't match exactly between the two datasets (e.g., "USA" vs "United States").
    # For a beginner project, we will just drop the matches where we couldn't find an ELO rating.
    matches_clean = matches.dropna(subset=['home_elo', 'away_elo']).copy()
    
    print(f"Matches remaining after merging ELO ratings: {len(matches_clean)}")

    print("Step 5: Feature Engineering")
    # Features are the inputs to our model. Let's create an "ELO Difference" feature.
    # If the difference is highly positive, the home team is much stronger.
    matches_clean['elo_difference'] = matches_clean['home_elo'] - matches_clean['away_elo']
    
    print("Step 6: Saving the Processed Data")
    # Save the cleaned dataset to a new CSV file so our model training script can use it.
    matches_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"Processed dataset saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    prepare_data()
