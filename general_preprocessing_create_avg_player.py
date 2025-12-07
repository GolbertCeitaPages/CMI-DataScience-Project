import pandas as pd

df = pd.read_csv("atp_transformed/2000-2024 players_2.csv")

# List of all columns to compute median
median_columns = [
    'ace', 'double_faults', 'points_on_serve', 'first_serve_in', '1stWon', '2ndWon',
    'service_games', 'break_points_saved', 'break_points_faced',
    'round_1_numb', 'round_1_tb_numb', 'round_2_numb', 'round_2_tb_numb',
    'round_3_numb', 'round_3_tb_numb', 'round_4_numb', 'round_4_tb_numb',
    'round_5_numb', 'round_5_tb_numb', 'round_1_diff', 'round_1_tb_diff',
    'round_2_diff', 'round_2_tb_diff', 'round_3_diff', 'round_3_tb_diff',
    'round_4_diff', 'round_4_tb_diff', 'round_5_diff', 'round_5_tb_diff',
    'mean_numb', 'median_numb', 'total_numb', 'mean_diff', 'median_diff', 'total_diff',
    'mean_tb_numb', 'median_tb_numb', 'total_tb_numb', 'mean_tb_diff', 'median_tb_diff',
    'total_tb_diff', 'sets_won', 'sets_lost', 'tb_sets_won', 'tb_sets_lost',
    'set_dominance', 'tb_dominance', 'highest_finish_position', 'minutes_rolling_med_10',
    'draw_size_rolling_med_10', 'finish_position_rolling_med_10', 'highest_finish_position_rolling_med_10',
    'ace_rolling_mean_10', 'double_faults_rolling_mean_10', 'points_on_serve_rolling_mean_10',
    'first_serve_in_rolling_mean_10', '1stWon_rolling_mean_10', '2ndWon_rolling_mean_10',
    'service_games_rolling_mean_10', 'break_points_saved_rolling_mean_10',
    'break_points_faced_rolling_mean_10', 'elo_pre_match_rolling_mean_10',
    'opponent_elo_pre_match_rolling_mean_10', 'set_dominance_rolling_mean_10',
    'tb_dominance_rolling_mean_10', 'player_rank_rolling_mean_10', 'mean_numb_rolling_mean_10',
    'median_numb_rolling_mean_10', 'total_numb_rolling_mean_10', 'mean_diff_rolling_mean_10',
    'median_diff_rolling_mean_10', 'total_diff_rolling_mean_10', 'mean_tb_numb_rolling_mean_10',
    'median_tb_numb_rolling_mean_10', 'total_tb_numb_rolling_mean_10', 'mean_tb_diff_rolling_mean_10',
    'median_tb_diff_rolling_mean_10', 'total_tb_diff_rolling_mean_10'
]

# Bin days_of_experience into 30-day intervals
df['days_bin'] = (df['days_of_experience'] // 30) * 30

# median columns
median_columns += ['player_rank', 'elo_pre_match']

# Compute medians per bin
synthetic_player = df.groupby('days_bin')[median_columns].median().reset_index()

# Add synthetic metadata
synthetic_player['tourney_date'] = pd.to_datetime('2024-11-30') - pd.to_timedelta(synthetic_player['days_bin'], unit='d')
synthetic_player['tourney_level'] = 'Synthetic Level'
synthetic_player['tourney_name'] = 'Synthetic tournament'
synthetic_player['tourney_id'] = 9999
synthetic_player['player_name'] = 'Synthetic Player'
synthetic_player['player_id'] = 0
synthetic_player['player_country'] = 'SYN'

# Compute derived columns
synthetic_player['months_of_experience'] = synthetic_player['days_bin'] / 30
synthetic_player['years_of_experience'] = synthetic_player['days_bin'] / 365
synthetic_player['player_age'] = (synthetic_player['years_of_experience']).round()
synthetic_player = synthetic_player.rename(columns={'days_bin': 'days_of_experience'})

# Add first_tourney_date
synthetic_player['first_tourney_date'] = synthetic_player['tourney_date'].min()

df_with_synth = pd.concat([df,synthetic_player])

df_with_synth.to_csv("atp_transformed/2000-2024 players_2_syn_player.csv")