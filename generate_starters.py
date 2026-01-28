"""Generate starters cache from NBA API"""
from nba_api.stats.endpoints import leaguedashplayerstats
import pandas as pd
import json
import sys

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("Fetching player stats from NBA API...")
stats = leaguedashplayerstats.LeagueDashPlayerStats(
    season='2025-26', 
    per_mode_detailed='PerGame', 
    timeout=120
)
df = stats.get_data_frames()[0]
print(f'Got {len(df)} players')

# Build starters for each team (top 5 by minutes)
all_starters = {}
teams = df['TEAM_ABBREVIATION'].unique()

for team in sorted(teams):
    team_players = df[df['TEAM_ABBREVIATION'] == team].sort_values('MIN', ascending=False).head(5)
    starters = []
    for _, p in team_players.iterrows():
        # Infer position from stats
        reb = float(p.get('REB', 0) or 0)
        ast = float(p.get('AST', 0) or 0)
        blk = float(p.get('BLK', 0) or 0)
        if ast > 5 and reb < 6:
            pos = "G"
        elif reb > 8 or blk > 1.5:
            pos = "C"
        elif reb > 5:
            pos = "F"
        else:
            pos = "G-F"
        
        starters.append({
            'name': p['PLAYER_NAME'],
            'position': pos,
            'pts': round(float(p['PTS']), 1),
            'reb': round(float(p['REB']), 1),
            'ast': round(float(p['AST']), 1),
            'min': round(float(p['MIN']), 1)
        })
    all_starters[team] = starters
    print(f'{team}: {len(starters)} players')

# Save to file
output_path = 'data/api_data/starters_cache.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(all_starters, f, indent=2, ensure_ascii=False)
print(f'\nSaved {len(all_starters)} teams to {output_path}')
