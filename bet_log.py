import os
import pandas as pd
from datetime import datetime

LOG_FILE = "bet_log.csv"

def save_bet(bet_data: dict) -> None:
    """Append a placed bet to the local CSV log."""
    df_new = pd.DataFrame([bet_data])
    if os.path.exists(LOG_FILE):
        df_old = pd.read_csv(LOG_FILE)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined.to_csv(LOG_FILE, index=False)
    else:
        df_new.to_csv(LOG_FILE, index=False)

def load_bet_log() -> pd.DataFrame:
    """Load the saved bet history."""
    if os.path.exists(LOG_FILE):
        return pd.read_csv(LOG_FILE)
    return pd.DataFrame(columns=[
        "date", "sport", "player", "stat", "line", "side",
        "odds_taken", "stake", "closing_odds", "clv_beaten"
    ])
