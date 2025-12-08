import pandas as pd
from pathlib import Path
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

base_df = pd.read_csv("atp_transformed/2000-2024 XXL.csv")

df_merged = base_df

df_merged["avg_outcome"] = (
    df_merged[
        ["encoded_num_round_1", "encoded_num_round_2", "encoded_num_round_3", "encoded_num_round_4", "encoded_num_round_5"]
    ]
    .mode(axis=1)
    .iloc[:, 0]  # select the first mode value in case of ties
)


def view_correlation(df):

    numeric_df = df.select_dtypes(include='number')

    correlation_matrix_pearson = numeric_df.corr() 
    correlation_matrix_spearman = numeric_df.corr("spearman") 

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix_pearson, 
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )
    plt.title("Correlation Matrix Pearson")
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix_spearman, 
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )
    plt.title("Correlation Matrix Spearman")

    plt.show()

#df_merged.to_csv("atp_transformed/2000-2024 XXL_enc.csv",index=False)

#view_correlation(df_merged)

import pandas as pd
from scipy import stats
# Split player_rank values into groups by avg_outcome
groups = [group["player_rank"].dropna().values for _, group in df_merged.groupby("avg_outcome")]

# Perform one-way ANOVA
f_stat, p_value = stats.f_oneway(*groups)

print(f"F-statistic: {f_stat:.3f}, p-value: {p_value:.10f}")

# # Sort avg_outcome so boxes appear in order
# order = sorted(df_merged["avg_outcome"].unique())

# plt.figure(figsize=(10, 6))
# sns.boxplot(
#     data=df_merged,
#     x="avg_outcome",
#     y="player_rank",
#     order=order,
# )

# plt.title("Distribution of Player Rank by Average Outcome")
# plt.xlabel("Average Encoded Outcome (avg_outcome)")
# plt.ylabel("Player Rank (lower = better)")
# plt.gca().invert_yaxis()  # optional: so top-ranked players appear higher up

# plt.show()
