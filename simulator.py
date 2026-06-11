import pandas as pd
import numpy as np
import joblib
import os

# Define Paths
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(DATA_DIR, "fifa_model.pkl")
ELO_FILE = os.path.join(DATA_DIR, "elo_ratings_wc2026.csv")
SCHEDULE_FILE = os.path.join(DATA_DIR, "schedule_2026.csv")

# Load the trained ML model
try:
    model = joblib.load(MODEL_FILE)
except:
    model = None

# Load ELO dataset and get the most recent rating for each country
elo_df = pd.read_csv(ELO_FILE)
# Sort by year descending, then drop duplicates to keep only the latest ELO for each country
latest_elo = elo_df.sort_values(by='year', ascending=False).drop_duplicates(subset=['country'])
elo_dict = dict(zip(latest_elo['country'], latest_elo['rating']))

def get_team_elo(team_name):
    """Safely fetch a team's ELO rating. If not found, return an average rating."""
    return elo_dict.get(team_name, 1500) # 1500 is roughly an average team

def get_all_teams():
    """Returns a sorted list of all available teams."""
    return sorted(list(elo_dict.keys()))

def predict_single_match(home_team, away_team):
    """
    Predicts the outcome of a match between two teams.
    Returns the probabilities of [Away Win, Draw, Home Win]
    """
    if model is None:
        return [0.33, 0.33, 0.34] # Fallback if model isn't loaded
        
    home_elo = get_team_elo(home_team)
    away_elo = get_team_elo(away_team)
    elo_diff = home_elo - away_elo
    
    # Create the feature dataframe exactly as the model was trained
    # For World Cup simulations, we'll assume it's a neutral venue (is_home_advantage = 0)
    features = pd.DataFrame({
        'home_elo': [home_elo],
        'away_elo': [away_elo],
        'elo_difference': [elo_diff],
        'is_home_advantage': [0] 
    })
    
    # predict_proba returns the probabilities of each class
    # Classes are: 0 (Away Win), 1 (Draw), 2 (Home Win)
    probabilities = model.predict_proba(features)[0]
    
    return {
        'Away Win': probabilities[0],
        'Draw': probabilities[1],
        'Home Win': probabilities[2]
    }

def simulate_group_matches():
    """
    Reads the 2026 schedule, predicts every match, and returns the results.
    """
    schedule = pd.read_csv(SCHEDULE_FILE)
    
    results = []
    for index, row in schedule.iterrows():
        home = row['home_team']
        away = row['away_team']
        
        # Predict the match
        probs = predict_single_match(home, away)
        
        # To make it fun, we simulate a hard outcome based on the probabilities
        # np.random.choice picks an outcome weighted by our model's probabilities!
        outcomes = ['Away Win', 'Draw', 'Home Win']
        simulated_outcome = np.random.choice(outcomes, p=[probs['Away Win'], probs['Draw'], probs['Home Win']])
        
        # Assign points based on soccer rules (3 for win, 1 for draw, 0 for loss)
        home_points = 0
        away_points = 0
        if simulated_outcome == 'Home Win':
            home_points = 3
        elif simulated_outcome == 'Away Win':
            away_points = 3
        else:
            home_points = 1
            away_points = 1
            
        results.append({
            'Match': f"{home} vs {away}",
            'Home Team': home,
            'Away Team': away,
            'Predicted Outcome': simulated_outcome,
            'Home Points': home_points,
            'Away Points': away_points
        })
        
    return pd.DataFrame(results)

def get_advancing_teams(results_df):
    """
    Takes the simulated group stage results, tallies points for all teams,
    and returns the top 32 teams to advance to the knockout stage.
    """
    points = {}
    for index, row in results_df.iterrows():
        home = row['Home Team']
        away = row['Away Team']
        points[home] = points.get(home, 0) + row['Home Points']
        points[away] = points.get(away, 0) + row['Away Points']
        
    # Create dataframe and sort by points
    standings = pd.DataFrame(list(points.items()), columns=['Team', 'Points'])
    standings = standings.sort_values(by='Points', ascending=False).reset_index(drop=True)
    
    # Return the top 32 teams
    return standings.head(32)['Team'].tolist()

def predict_knockout_match(home_team, away_team):
    """
    Predicts a knockout match where Draws are not allowed.
    Forces a Win/Loss outcome based on normalized probabilities.
    """
    probs = predict_single_match(home_team, away_team)
    
    # Remove draw probability and normalize so Win/Loss equals 100%
    home_win_prob = probs['Home Win']
    away_win_prob = probs['Away Win']
    total = home_win_prob + away_win_prob
    
    home_win_prob = home_win_prob / total
    away_win_prob = away_win_prob / total
    
    # Simulate outcome (no draws allowed!)
    simulated_outcome = np.random.choice(
        ['Away Win', 'Home Win'], 
        p=[away_win_prob, home_win_prob]
    )
    
    if simulated_outcome == 'Home Win':
        return home_team
    else:
        return away_team

def simulate_knockout_stage(teams):
    """
    Takes 32 teams, seeds them (1st vs 32nd, etc.), and resolves the bracket.
    Returns a dictionary of all stages.
    """
    bracket = {
        'Round of 32': [],
        'Round of 16': [],
        'Quarter-Finals': [],
        'Semi-Finals': [],
        'Final': [],
        'Champion': ""
    }
    
    # Seed teams: 1 vs 32, 2 vs 31, etc.
    current_round_matches = []
    for i in range(16):
        current_round_matches.append((teams[i], teams[31-i]))
        
    rounds = ['Round of 32', 'Round of 16', 'Quarter-Finals', 'Semi-Finals', 'Final']
    
    for round_name in rounds:
        next_round_teams = []
        
        # Play all matches in the current round
        for match in current_round_matches:
            home = match[0]
            away = match[1]
            winner = predict_knockout_match(home, away)
            
            bracket[round_name].append({
                'Match': f"{home} vs {away}",
                'Winner': winner
            })
            next_round_teams.append(winner)
            
        # Pair up winners for the next round
        current_round_matches = []
        for i in range(0, len(next_round_teams), 2):
            if i+1 < len(next_round_teams):
                current_round_matches.append((next_round_teams[i], next_round_teams[i+1]))
                
        if len(next_round_teams) == 1:
            bracket['Champion'] = next_round_teams[0]
            
    return bracket

if __name__ == "__main__":
    # Test the simulator
    print("Testing Argentina vs France:")
    print(predict_single_match("Argentina", "France"))
