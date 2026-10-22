import datetime
import numpy as np
import pandas as pd
from sports_data import fetch_nba_player_logs
def american_to_implied_prob(american_odds):
    try:
        odds = float(american_odds)
        if odds < 0:
            return round((-odds / (-odds + 100)) * 100, 1)
        else:
            return round((100 / (odds + 100)) * 100, 1)
    except Exception:
        return 52.4

def get_nba_logs(player_name, season="2025-26"):
    """Fetch NBA game logs via ESPN in sports_data."""
    df, err = fetch_nba_player_logs(player_name)
    if err:
        return None, err
    return df, None
    clean_name = player_name.strip().lower()
    all_players = nba_players.get_players()
    match = [p for p in all_players if clean_name in p['full_name'].lower()]
    
    if not match:
        return None, f"NBA player '{player_name}' not found."
        
    player_id = match[0]['id']
    df = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
    return df, None

def get_nfl_logs(player_name):
    """Directly loads nflverse weekly player statistics from GitHub releases without 404 crashes."""
    curr_yr = datetime.datetime.now().year
    seasons_to_try = [curr_yr, curr_yr - 1, 2024]
    
    query = player_name.strip().lower()
    parts = query.split()
    first_init = f"{parts[0][0]}.{parts[-1]}" if len(parts) > 1 else query
    
    p_df = pd.DataFrame()
    
    for yr in seasons_to_try:
        url = f"https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{yr}.parquet"
        try:
            # Read directly from nflverse GitHub release mirror
            df_temp = pd.read_parquet(url)
            if df_temp is None or df_temp.empty:
                continue
            
            # Filter strictly for Regular Season
            if 'season_type' in df_temp.columns:
                df_temp = df_temp[df_temp['season_type'] == 'REG']
            
            # Filter by case-insensitive name matching
            mask = pd.Series(False, index=df_temp.index)
            
            if 'player_display_name' in df_temp.columns:
                disp = df_temp['player_display_name'].fillna('').str.lower()
                mask |= (disp == query)
                mask |= disp.str.contains(query, regex=False)
                
            if 'player_name' in df_temp.columns:
                short = df_temp['player_name'].fillna('').str.lower()
                mask |= (short == query)
                mask |= (short == first_init)
                mask |= short.str.contains(query, regex=False)
            
            matched = df_temp[mask].copy()
            
            if not matched.empty:
                p_df = matched.sort_values(by='week').reset_index(drop=True)
                break
        except Exception:
            # Silently catch HTTP 404/network errors and try prior year dataset
            continue
            
    if p_df.empty:
        return None, f"NFL player '{player_name}' not found in recent nflverse datasets."
        
    return p_df, None

def evaluate_multi_sport_prop(league, player_name, stat_category, line, odds=-110):
    league = league.upper()
    stat_category = stat_category.upper()
    stats = []
    labels = []
    
    if league == "NBA":
        df, err = get_nba_logs(player_name)
        if err: return {"error": err}
        stat_map = {"PTS": "PTS", "REB": "REB", "AST": "AST", "3PM": "3PM"}
        if stat_category not in stat_map: return {"error": f"Stat '{stat_category}' not supported for NBA."}
        stats = df[stat_map[stat_category]].astype(float).values
        labels = df['MATCHUP'].values if 'MATCHUP' in df.columns else [f"Game {i+1}" for i in range(len(stats))]
        
    elif league == "NFL":
        df, err = get_nfl_logs(player_name)
        if err: return {"error": err}
        
        if stat_category == "RUSH_REC_YDS":
            stats = (df['rushing_yards'].fillna(0) + df['receiving_yards'].fillna(0)).astype(float).values
        elif stat_category == "PASS_RUSH_YDS":
            stats = (df['passing_yards'].fillna(0) + df['rushing_yards'].fillna(0)).astype(float).values
        elif stat_category == "KICKING_POINTS":
            stats = (df['field_goals_made'].fillna(0) * 3 + df['extra_points_made'].fillna(0)).astype(float).values
        else:
            stat_map = {
                "PASS_YDS": "passing_yards", "PASS_TDS": "passing_tds", "INTERCEPTIONS": "passing_interceptions",
                "PASS_ATTEMPTS": "passing_attempts", "PASS_COMPLETIONS": "passing_completions",
                "RUSH_YDS": "rushing_yards", "RUSH_TDS": "rushing_tds", "RUSH_ATTEMPTS": "carries",
                "REC_YDS": "receiving_yards", "REC_TDS": "receiving_tds", "RECEPTIONS": "receptions", "TARGETS": "targets",
                "FG_MADE": "field_goals_made", "PAT_MADE": "extra_points_made"
            }
            if stat_category not in stat_map: return {"error": f"Stat '{stat_category}' not supported for NFL."}
            stats = df[stat_map[stat_category]].fillna(0).astype(float).values
        
        # Build Opponent Matchup Labels (e.g. "Wk 1 vs PHI")
        game_labels = []
        for _, row in df.iterrows():
            wk = int(row['week'])
            opp = row.get('opponent_team', '')
            if pd.notna(opp) and str(opp).strip():
                game_labels.append(f"Wk {wk} vs {opp}")
            else:
                game_labels.append(f"Wk {wk}")
        labels = game_labels
        
    elif league == "MLB":
        return {"error": "MLB real-time parser requires specific game-log mapping."}
    else:
        return {"error": "Invalid league selected."}
        
    total_games = len(stats)
    if total_games == 0: return {"error": "No games found for this player."}
    
    mean = round(float(np.mean(stats)), 2)
    median = round(float(np.median(stats)), 2)
    recent_sample = 5 if league == "NFL" else 10
    recent_stats = stats[-recent_sample:]
    
    season_hit_rate = round(float(np.sum(stats > line) / total_games * 100), 1)
    recent_hit_rate = round(float(np.sum(recent_stats > line) / len(recent_stats) * 100), 1)
    
    implied_prob = american_to_implied_prob(odds)
    edge = round(recent_hit_rate - implied_prob, 1)
    
    skew = mean - median
    verdict = "PASS"
    reason = ""
    
    if median > line and recent_hit_rate >= 60.0 and edge > 0:
        verdict = f"BET OVER ({odds})"
        reason = f"Median ({median}) > Line ({line}) with a {recent_hit_rate}% recent hit rate. Holds a +{edge}% edge over implied odds."
    elif median > line and recent_hit_rate >= 60.0 and edge <= 0:
        verdict = "PASS (BAD VALUE)"
        reason = f"Hit rate is solid ({recent_hit_rate}%), but heavy juice ({odds} = {implied_prob}% implied) eliminates your edge."
    elif median < line and mean < line and recent_hit_rate <= 40.0:
        verdict = "FADE (BET UNDER)"
        reason = f"Both Median ({median}) and Mean ({mean}) are under line ({line}). Recent over hit rate is only {recent_hit_rate}%."
    elif skew > (line * 0.15) and median < line:
        verdict = "FADE / PASS (TRAP BET)"
        reason = f"Mean ({mean}) is heavily inflated by outliers, but Median ({median}) stays under line ({line})."
    else:
        verdict = "PASS"
        reason = f"No clear statistical edge. Median ({median}) vs Line ({line}). Edge: {edge}%."
        
    return {
        "league": league, "player": player_name, "stat": stat_category, "line": line,
        "odds": odds, "implied_prob": f"{implied_prob}%", "edge": f"{edge:+}%",
        "games_played": total_games, "mean": mean, "median": median,
        "season_hit_rate": f"{season_hit_rate}%", "recent_hit_rate": f"{recent_hit_rate}% (Last {len(recent_stats)} Games)",
        "verdict": verdict, "reason": reason, "raw_stats": stats, "game_labels": labels
    }
