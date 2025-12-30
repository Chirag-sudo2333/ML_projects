import pandas as pd
import numpy as np

player_stats = {
    "Player_ID": ["P01","P02","P03","P04","P05","P06","P07","P08","P09","P10",
                  "P11","P12","P13","P14","P15","P16","P17","P18","P19","P20"],
    
    "Team": ["A","A","A","A","A","B","B","B","B","B",
             "C","C","C","C","C","D","D","D","D","D"],
    
    "Position": ["G","F","C","G","F","G","F","C","G","F",
                 "G","F","C","G","F","G","F","C","G","F"],
    
    "Minutes": [28,30,26,22,24,32,29,27,20,25,
                31,28,24,18,26,33,30,28,21,25],
    
    "PTS": [14,18,10,9,12,21,16,8,7,13,
            19,15,9,6,14,22,17,11,8,12],
    
    "REB": [3,9,11,2,7,4,8,12,1,6,
            5,9,10,2,7,3,8,13,2,6],
    
    "AST": [6,2,1,4,3,7,2,1,3,2,
            5,3,1,2,2,8,2,1,3,2],
    
    "TOV": [4,3,2,5,2,6,3,2,4,2,
            5,3,1,3,2,7,4,2,5,2],
    
    "FG_pct": [0.42,0.51,0.56,0.38,0.47,0.44,0.49,0.58,0.36,0.46,
               0.43,0.50,0.54,0.35,0.48,0.45,0.47,0.59,0.37,0.44],
    
    "TP_pct": [0.31,0.22,0.00,0.29,0.35,0.39,0.18,0.00,0.26,0.33,
               0.34,0.21,0.00,0.28,0.31,0.41,0.25,0.00,0.27,0.32],
    
    "FT_pct": [0.78,0.65,0.60,0.81,0.72,0.85,0.70,0.55,0.75,0.68,
               0.82,0.66,0.58,0.77,0.70,0.88,0.69,0.62,0.80,0.73],
    
    "STL": [2,1,0,1,2,3,1,0,1,2,
            2,1,0,1,2,3,1,0,1,2],
    
    "BLK": [0,1,2,0,1,0,1,3,0,1,
            0,1,2,0,1,0,1,3,0,1]
}
league_baseline = {
    "PTS_per_min": [0.55],
    "FG_pct": [0.45],
    "TP_pct": [0.33],
    "FT_pct": [0.72],
    "AST_TOV_ratio": [1.5],
    "REB_per_game": [6.5]
}
df_new = pd.DataFrame()
df_baseline = pd.DataFrame(league_baseline)
df_player = pd.DataFrame(player_stats)
df_new['Player_id'] = df_player['Player_ID']
df_new['Position'] = df_player['Position']
df_new['PTS_MIN'] = df_player['PTS']/df_player['Minutes']
df_new['PTS_status'] = np.where(df_new['PTS_MIN'] > (df_baseline['PTS_per_min'].loc[0]), 'Above Avg', 'Below Avg')
df_new['FG_status'] = np.where(df_player['FG_pct']> df_baseline['FG_pct'].loc[0], 'Above Avg', 'Below Avg')
df_new['TP_status'] = np.where(df_player['TP_pct'] > df_baseline['TP_pct'].iloc[0], 'Above Avg', 'Below Avg' )
df_new['FT_status'] = np.where(df_player['FT_pct'] > df_baseline['FT_pct'].iloc[0], 'Above Avg', 'Below Avg')
df_new['AST_TOV_ratio'] = df_player['AST']/df_player['TOV']
df_new['AST_TOV_status'] = np.where(df_new['AST_TOV_ratio'] > df_baseline['AST_TOV_ratio'].iloc[0], 'Above Avg', 'Below Avg')
df_new['REB_status'] = np.where(df_player['REB'] > df_baseline['REB_per_game'].iloc[0], 'Above Avg', 'Below Avg')
print(df_new)

