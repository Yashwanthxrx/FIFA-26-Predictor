import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Define paths
DATA_DIR = r"d:\ML"
DATA_FILE = os.path.join(DATA_DIR, "processed_dataset.csv")
MODEL_FILE = os.path.join(DATA_DIR, "fifa_model.pkl")

def train():
    print("Step 1: Loading Processed Data")
    # We load the data we cleaned in data_prep.py
    df = pd.read_csv(DATA_FILE)
    
    # We drop any remaining missing values just in case
    df = df.dropna(subset=['elo_difference', 'target'])
    
    print(f"Total matches for training: {len(df)}")

    print("Step 2: Defining Features (X) and Target (y)")
    # X contains the features (inputs) the model will learn from.
    # We added is_home_advantage from our new data_prep.py!
    X = df[['home_elo', 'away_elo', 'elo_difference', 'is_home_advantage']]
    
    # y is the target (what we want to predict). 
    # Remember: 0 = Away Win, 1 = Draw, 2 = Home Win
    y = df['target']

    print("Step 3: Splitting the Data into Training and Testing Sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training on {len(X_train)} matches, testing on {len(X_test)} matches.")

    print("Step 4: Training and Comparing Models")
    
    # Let's perform a mini Grid Search to find the absolute best settings for Logistic Regression!
    from sklearn.model_selection import GridSearchCV
    
    # We will test different regularization strengths (C) and class weights
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'class_weight': [None, 'balanced']
    }
    
    print("Tuning Logistic Regression parameters...")
    grid_search = GridSearchCV(LogisticRegression(max_iter=2000, solver='lbfgs', random_state=42), param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    best_log_reg = grid_search.best_estimator_
    
    # Initialize all three models. 
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression (Tuned)": best_log_reg,
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    }
    
    best_model_name = ""
    best_accuracy = 0
    best_model = None

    print("\n--- Model Comparison ---")
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        # Predict
        y_pred = model.predict(X_test)
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"{name} Accuracy: {accuracy * 100:.2f}%")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model = model

    print(f"\nWinner: {best_model_name} with {best_accuracy * 100:.2f}% accuracy!")

    print("\nStep 5: Saving the Best Model")
    joblib.dump(best_model, MODEL_FILE)
    print(f"Model saved successfully to {MODEL_FILE}")

if __name__ == "__main__":
    train()
