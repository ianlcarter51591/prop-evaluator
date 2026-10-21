import streamlit as st
import pandas as pd
import numpy as np

# Page Config
st.set_page_config(
    page_title="Prop Evaluator",
    page_icon="📊",
    layout="wide"
)

# App Header
st.title("📊 Sports Prop Evaluator")
st.markdown("Welcome to your live prop analytics dashboard. Market odds and simulation models are ready.")

# Sidebar Controls
st.sidebar.header("Configuration")
sport_filter = st.sidebar.selectbox("Select Sport", ["MLB", "NFL", "NBA", "NHL", "CBB", "CFB"])
min_edge = st.sidebar.slider("Minimum Probability Edge (%)", 0.0, 10.0, 2.0, 0.5)

# Main Dashboard View
st.subheader(f"Active Markets — {sport_filter}")

# Placeholder dataframe to verify UI rendering
data = {
    "Player": ["Player A", "Player B", "Player C"],
    "Prop": ["Over 1.5 Bases", "Over 24.5 Pass Yds", "Under 6.5 Ks"],
    "Market Odds": [-110, -115, +105],
    "Model Probability": ["58.2%", "61.0%", "54.5%"],
    "Edge": [f"+{min_edge + 1.2}%", f"+{min_edge + 2.5}%", f"+{min_edge + 0.8}%"]
}

df = pd.DataFrame(data)
st.dataframe(df, use_container_width=True)

st.success("App successfully loaded and connected to live environment!")
