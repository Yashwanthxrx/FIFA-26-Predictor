# ⚽ Advanced FIFA 26 Predictor & Tournament Simulator

Welcome to the **FIFA 26 Predictor**, a Machine Learning powered dashboard designed to simulate the ultimate soccer clashes and predict the pathway to the 2026 World Cup Champion!

This project leverages historical ELO ratings and decades of FIFA match data to predict the outcomes of international soccer matches using a carefully tuned **Logistic Regression** model. The entire experience is wrapped in a beautiful, interactive **Streamlit** dashboard featuring an Emerald Pitch glassmorphism UI.

---

## 🌟 Features

*   **🔮 Single Match Predictor:** Pit any two nations against each other and watch the AI calculate the exact win, loss, and draw probabilities using historical power ratings and home-field advantages. Includes interactive Donut charts!
*   **📊 Team Power Analytics:** Select a country to view its global ELO ranking and instantly simulate a gauntlet against the world's top 5 contenders (Argentina, France, Brazil, England, Spain).
*   **🏆 Mass Group Simulator:** Simulate over 100+ group stage matches instantly. The AI calculates the results and generates a heat-mapped points table.
*   **🌍 The Road to Glory (Full Tournament):** Let the AI run the *entire* World Cup! It tallies group stage points, calculates the Top 32 advancing teams, seeds them into a bracket, and resolves a sudden-death knockout stage all the way to crowning the World Champion.

---

## 🛠️ Tech Stack

*   **Language:** Python 3.12
*   **Machine Learning:** Scikit-Learn (`LogisticRegression`, `GridSearchCV`)
*   **Data Processing:** Pandas, NumPy
*   **Frontend UI:** Streamlit, Altair (for interactive charts)

---

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Yashwanthxrx/FIFA-26-Predictor
   ```

2. **Install the dependencies:**
   Make sure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Data Pipeline (Optional):**
   If you want to re-train the AI from scratch using the raw datasets, run these scripts in order:
   ```bash
   python data_prep.py     # Cleans the data and engineers features (like Home Advantage)
   python train_model.py   # Trains the ML model and saves it as fifa_model.pkl
   ```

4. **Launch the Dashboard:**
   Start the interactive Streamlit UI:
   ```bash
   streamlit run app.py
   ```
   *The app will automatically open in your browser at `http://localhost:8501`.*

---

## 📁 Project Structure

*   `app.py` - The main Streamlit dashboard application and UI layout.
*   `simulator.py` - Contains the core simulation logic, point tallying, bracket generation, and ML model inference.
*   `data_prep.py` - The ETL script that merges historical match data with global ELO ratings.
*   `train_model.py` - The machine learning script that uses Grid Search to tune and train the predictive model.
*   `fifa_model.pkl` - The pre-trained Logistic Regression model.
*   `*.csv` - Various datasets including ELO ratings, 2026 World Cup schedules, and historical matches.

---

## 🧠 How the AI Works

The core engine is a **Logistic Regression** model with accuracy score of 67.57% trained to recognize the relationship between two specific features and the match outcome:
1.  **ELO Difference:** The mathematical gap in skill between the Home and Away team.
2.  **Home Advantage:** A binary feature tracking whether a team is playing on their home turf.

When running the **Knockout Stage Simulator**, the model dynamically drops its "Draw" probability and forces a sudden-death Win/Loss decision based on normalized stats to ensure the bracket resolves down to a single champion!
