import streamlit as st
import pandas as pd
import altair as alt

# Import the functions we built in simulator.py
from simulator import get_all_teams, predict_single_match, simulate_group_matches, get_advancing_teams, simulate_knockout_stage, get_team_elo

# Set up the page layout
st.set_page_config(page_title="FIFA Win Predictor", page_icon="⚽", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Emerald Pitch & Glassmorphism UI
st.markdown("""
<style>
    /* Main Background - Deep Turf Green */
    .stApp {
        background: linear-gradient(135deg, #072615, #0F3E22, #1A5333);
        color: #ffffff;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 35, 20, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Headers and Text */
    h1, h2, h3 {
        color: #A3FFC2 !important; /* Soft neon green accent */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Glassmorphism Containers */
    div[data-testid="stVerticalBlock"] > div {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #FFD700 !important; /* Gold for stats */
        font-weight: bold;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #2D764E 0%, #1A5333 100%);
        color: white;
        border: 1px solid #A3FFC2;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(163, 255, 194, 0.5);
        border: 1px solid #FFFFFF;
    }
    
    /* Dataframes/Tables text color fix for dark mode */
    .stDataFrame {
        color: white;
    }
    
    /* Hide some default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR NAVIGATION -----------------
st.sidebar.title("⚽ Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select a Module:",
    ["🔮 Match Predictor", "📊 Team Analysis", "🏆 Group Simulator", "🌍 The Road to Glory"]
)

st.sidebar.markdown("---")
st.sidebar.info("Powered by a Random Forest Machine Learning Model trained on decades of FIFA data.")

teams = get_all_teams()

# ----------------- MAIN CONTENT AREA -----------------

if page == "🔮 Match Predictor":
    st.title("⚽ Match Predictor")
    st.markdown("*Initialize the AI to simulate an epic clash.*")
    
    col1, col_vs, col2 = st.columns([2, 1, 2])
    with col1:
        home_team = st.selectbox("Home Team (or Team A)", teams, index=teams.index("Argentina") if "Argentina" in teams else 0)
        home_elo = get_team_elo(home_team)
        st.metric(label=f"{home_team} Power Rating", value=int(home_elo))
        
    with col_vs:
        st.markdown("<h1 style='text-align: center; margin-top: 20px; color: #FFFFFF !important;'>VS</h1>", unsafe_allow_html=True)
        
    with col2:
        away_team = st.selectbox("Away Team (or Team B)", teams, index=teams.index("France") if "France" in teams else 1)
        away_elo = get_team_elo(away_team)
        st.metric(label=f"{away_team} Power Rating", value=int(away_elo), delta=int(away_elo - home_elo), delta_color="normal")
        
    if st.button("Predict Outcome 🚀", key="predict_single"):
        if home_team == away_team:
            st.error("Teams must be different!")
        else:
            with st.spinner("Analyzing turf conditions and historical data..."):
                probs = predict_single_match(home_team, away_team)
                
                st.markdown("### Prediction Results")
                
                # Visual Metric row
                col_a, col_b, col_c = st.columns(3)
                col_a.metric(f"🟢 {home_team} Win", f"{probs['Home Win']*100:.1f}%")
                col_b.metric("⚪ Draw", f"{probs['Draw']*100:.1f}%")
                col_c.metric(f"🟡 {away_team} Win", f"{probs['Away Win']*100:.1f}%")
                
                # Interactive Donut Chart
                chart_data = pd.DataFrame({
                    'Outcome': [f"{home_team} Win", "Draw", f"{away_team} Win"],
                    'Probability': [probs['Home Win'], probs['Draw'], probs['Away Win']],
                    'Color': ['#A3FFC2', '#FFFFFF', '#FFD700'] # Neon Green, White, Gold
                })
                
                donut_chart = alt.Chart(chart_data).mark_arc(innerRadius=50).encode(
                    theta=alt.Theta(field="Probability", type="quantitative"),
                    color=alt.Color(field="Outcome", type="nominal", scale=alt.Scale(domain=chart_data['Outcome'].tolist(), range=chart_data['Color'].tolist())),
                    tooltip=['Outcome', alt.Tooltip('Probability:Q', format='.1%')]
                ).properties(height=300, title="Win Probability Distribution")
                
                st.altair_chart(donut_chart, use_container_width=True)

elif page == "📊 Team Analysis":
    st.title("📊 Team Analytics")
    st.markdown("*Deep dive into a specific nation's global standing.*")
    
    selected_team = st.selectbox("Select a Team to Analyze", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
    
    elo = get_team_elo(selected_team)
    
    st.subheader(f"Current Status: {selected_team}")
    st.metric("Global ELO Rating", int(elo))
    
    st.markdown(f"**How {selected_team} matches up against top contenders:**")
    top_contenders = ["Argentina", "France", "Brazil", "England", "Spain"]
    contender_results = []
    
    for contender in top_contenders:
        if contender != selected_team and contender in teams:
            probs = predict_single_match(selected_team, contender)
            contender_results.append({
                "Opponent": contender,
                "Win Chance": probs['Home Win'],
                "Draw Chance": probs['Draw'],
                "Loss Chance": probs['Away Win']
            })
            
    if contender_results:
        df_contenders = pd.DataFrame(contender_results)
        df_contenders['Win Chance'] = df_contenders['Win Chance'].apply(lambda x: f"{x*100:.1f}%")
        df_contenders['Draw Chance'] = df_contenders['Draw Chance'].apply(lambda x: f"{x*100:.1f}%")
        df_contenders['Loss Chance'] = df_contenders['Loss Chance'].apply(lambda x: f"{x*100:.1f}%")
        st.table(df_contenders)

elif page == "🏆 Group Simulator":
    st.title("🏆 Mass Group Simulator")
    st.markdown("*Run 100+ group stage matches through the AI instantly.*")
    
    if st.button("Simulate All Group Matches 🎲"):
        with st.spinner("Processing group stages..."):
            results_df = simulate_group_matches()
            st.success("Group Stage Resolved!")
            # Use greens for the table background gradient
            st.dataframe(results_df.style.background_gradient(cmap='Greens', subset=['Home Points', 'Away Points']), use_container_width=True)

elif page == "🌍 The Road to Glory":
    st.title("🌍 The Road to Glory")
    st.markdown("*Simulate the ENTIRE World Cup from groups to the grand final.*")
    
    if st.button("Simulate the Tournament 🌌", key="sim_full"):
        with st.spinner("Generating World Cup Timeline..."):
            # 1. Group Stage
            group_results = simulate_group_matches()
            
            # 2. Advancing Teams
            advancing_teams = get_advancing_teams(group_results)
            
            # 3. Bracket
            bracket = simulate_knockout_stage(advancing_teams)
            
            st.balloons()
            st.markdown(f"<h2 style='text-align: center; color: #FFD700 !important; font-size: 3em;'>🏆 {bracket['Champion'].upper()} 🏆</h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; font-size: 1.2em; color: #FFFFFF;'>2026 WORLD CHAMPION</p>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            rounds_to_show = ['Round of 32', 'Round of 16', 'Quarter-Finals', 'Semi-Finals', 'Final']
            for r in rounds_to_show:
                st.markdown(f"### ⚔️ {r}")
                
                matches = bracket[r]
                cols = st.columns(4) 
                for i, match in enumerate(matches):
                    with cols[i % 4]:
                        st.info(f"**{match['Match']}**\n\n✅ *{match['Winner']}*")
                st.markdown("---")
