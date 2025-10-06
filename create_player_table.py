import pandas as pd
from pathlib import Path
import numpy as np
import os

# These are the column names:
"""
Unique: 
tourney_id,tourney_name,surface,draw_size,tourney_level,tourney_date,match_num,score,best_of,round,minutes

prefixed with winner/loser:
winner_id,winner_seed,winner_entry,winner_name,winner_hand,winner_ht,winner_ioc,winner_age,winner_rank,winner_rank_points

prefixed with w/l:
w_ace,w_df,w_svpt,w_1stIn,w_1stWon,w_2ndWon,w_SvGms,w_bpSaved,w_bpFaced
"""

def create_tables(year):
    
    # load all datasets of given year
    atp_matches_df = pd.read_csv(Path(__file__).resolve().parent / "atp_dataset" / f"atp_matches_{year}.csv")
    atp_qual_chall_df = pd.read_csv(Path(__file__).resolve().parent / "atp_dataset" / f"atp_matches_qual_chall_{year}.csv")
    atp_futures_df = pd.read_csv(Path(__file__).resolve().parent / "atp_dataset" / f"atp_matches_futures_{year}.csv")

    # Add the tournament type to all rows
    atp_matches_df["tourney_type"] = "matches"
    atp_qual_chall_df["tourney_type"] = "challengers"
    atp_futures_df["tourney_type"] = "futures"

    source_df = pd.concat([atp_matches_df,atp_qual_chall_df,atp_futures_df])

    # Rename round to match_round to avoid calling round method 
    source_df = source_df.rename(columns={"round":"tourney_round"})

    big_table_list = []

    # counter for debugging/testing
    row_count = 0
    for x in source_df.itertuples():
        
        if not pd.isna(x.score):
            # split the rounds 
            split_rounds = x.score.split(" ")

            if len(split_rounds) == 1:
                pass
                rounds_list = (split_rounds[0],np.nan,np.nan,np.nan,np.nan)
            elif len(split_rounds) == 2:
                rounds_list = (split_rounds[0],split_rounds[1],np.nan,np.nan,np.nan)
            elif len(split_rounds) == 3:
                rounds_list = (split_rounds[0],split_rounds[1],split_rounds[2],np.nan,np.nan)
            elif len(split_rounds) == 4:
                rounds_list = (split_rounds[0],split_rounds[1],split_rounds[2],split_rounds[3],np.nan)
            elif len(split_rounds) == 5:
                rounds_list = (split_rounds[0],split_rounds[1],split_rounds[2],split_rounds[3],split_rounds[4])
        else:
            rounds_list = (np.nan,)*5

        # construct the row for winners and losers. This will create a big singular table with no normalisation.
        full_winner_row = (
            x.tourney_id,
            x.tourney_name,
            x.tourney_type,
            x.surface,
            x.draw_size,
            x.tourney_level,
            x.tourney_date,
            x.match_num,
            x.score,
            *rounds_list,
            x.best_of,
            x.tourney_round,
            x.minutes,
            x.winner_id,
            x.winner_seed,
            x.winner_name,
            x.winner_hand,
            x.winner_ht,
            x.winner_ioc,
            x.winner_age,
            x.winner_rank,
            x.winner_rank_points,
            x.w_ace,x.w_df,
            x.w_svpt,x.w_1stIn,
            x.w_1stWon,x.w_2ndWon,
            x.w_SvGms,x.w_bpSaved,
            x.w_bpFaced,
            1, # Winner is 1
            str(x.winner_id) + ":" + str(x.tourney_id) + ":" + str(x.match_num)
        )
        full_loser_row = (
            x.tourney_id,
            x.tourney_name,
            x.tourney_type,
            x.surface,
            x.draw_size,
            x.tourney_level,
            x.tourney_date,
            x.match_num,
            x.score,
            *rounds_list,
            x.best_of,
            x.tourney_round,
            x.minutes,
            x.loser_id,
            x.loser_seed,
            x.loser_name,
            x.loser_hand,
            x.loser_ht,
            x.loser_ioc,
            x.loser_age,
            x.loser_rank,
            x.loser_rank_points,
            x.w_ace,x.w_df,
            x.w_svpt,x.w_1stIn,
            x.w_1stWon,x.w_2ndWon,
            x.w_SvGms,x.w_bpSaved,
            x.w_bpFaced,
            0, # loser is 0
            str(x.loser_id) + ":" + str(x.tourney_id) + ":" + str(x.match_num)
        )

        big_table_list.append(full_winner_row)
        big_table_list.append(full_loser_row)

        # row_count += 1
        # if row_count == 5:
        #     break

    # Create a flattened table for easy analysis.
    big_table_df = pd.DataFrame(big_table_list,columns=[
                                        # Tournament data
                                        "tourney_id","tourney_name","tourney_type","surface","draw_size","tourney_level","tourney_date","match_num","score",
                                        "round_1","round_2","round_3","round_4","round_5",
                                        "best_of","tourney_round","minutes",
                                        # Player data
                                        "player_id","player_seed","player_name","player_hand","player_height","player_country","player_age","player_rank","player_rank_points",
                                        "ace","double_faults","points_on_serve","first_serve_in","1st_won","2nd_won","service_games","break_points_saved","break_points_faced",
                                        "match_outcome","player_tourney_match_id"
                                                        ])

    # Post processing to round the table
    big_table_df["player_age"] = round(big_table_df["player_age"],0)

    # create folder if it does not exist
    os.makedirs("atp_transformed", exist_ok=True)

    #big_table_df.to_csv(Path(__file__).resolve().parent / "atp_transformed" / f"{year}.csv",index=False)
    return big_table_df

# tool to make statements for all matches of a specific time period
# list_of_dfs = []

# for y in range(2000,2025):
#     if y not in [2020,2021]:
#         print(f"df_{y} = create_tables({y})")
#         list_of_dfs.append(f"df_{y}")
# print(f"df_big = pd.concat({list_of_dfs})")

df_2000 = create_tables(2000)
df_2001 = create_tables(2001)
df_2002 = create_tables(2002)
df_2003 = create_tables(2003)
df_2004 = create_tables(2004)
df_2005 = create_tables(2005)
df_2006 = create_tables(2006)
df_2007 = create_tables(2007)
df_2008 = create_tables(2008)
df_2009 = create_tables(2009)
df_2010 = create_tables(2010)
df_2011 = create_tables(2011)
df_2012 = create_tables(2012)
df_2013 = create_tables(2013)
df_2014 = create_tables(2014)
df_2015 = create_tables(2015)
df_2016 = create_tables(2016)
df_2017 = create_tables(2017)
df_2018 = create_tables(2018)
df_2019 = create_tables(2019)
df_2020 = create_tables(2020)
df_2021 = create_tables(2021)
df_2022 = create_tables(2022)
df_2023 = create_tables(2023)
df_2024 = create_tables(2024)
df_big = pd.concat([df_2000, df_2001, df_2002, df_2003, df_2004, df_2005, df_2006, df_2007, df_2008, df_2009, df_2010, df_2011, df_2012, df_2013, df_2014, df_2015, df_2016, df_2017, df_2018, df_2019,df_2020,df_2021, df_2022, df_2023, df_2024])

df_big["tourney_date"] = pd.to_datetime(df_big["tourney_date"].astype(str), format="%Y%m%d")

df_big.to_csv(Path(__file__).resolve().parent / "atp_transformed" / f"2000-2024_complete.csv",index=False)