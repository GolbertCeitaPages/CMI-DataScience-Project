import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import math

full_df = pd.read_csv('atp_transformed/2000-2024_clean.csv')

selected_features = [
'tourney_type',
'surface',
'draw_size',
'tourney_level',
'match_num',
'round_1',
'round_2',
'round_3',
'round_4',
'round_5',
'best_of',
'tourney_round',
'minutes',
'player_id',
'player_seed',
'player_hand',
'player_height',
'player_country',
'player_age',
'player_rank_points',
'ace',
'double_faults',
'points_on_serve',
'first_serve_in',
'1st_won',
'2nd_won',
'service_games',
'break_points_saved',
'break_points_faced',
'match_outcome',
'binned_rank' # target
]

# drop all nan values since elo is not integrated in this solution (using elo here would be better so everyone starts at 1500)
full_df['player_rank'] = full_df['player_rank'].dropna()

bins = (
    [0, 1, 2, 3, 4, 6, 11, 21, 31, 41, 51, 101, 151, 201, 251, 301, 401] +
    list(range(501, 2501, 100))  # 501-600, 601-700, ..., up to 2401-2500
)

# Create labels for bins
labels = []

# Manually for the first ones
labels += ['1', '2', '3', '4-5', '6-10', '11-20', '21-30', '31-40', '41-50', '51-100', 
           '101-150', '151-200', '201-250', '251-300', '301-400', '401-500']

# Then dynamically for the 500+ ranges
for start in range(501, 2501, 100):
    end = min(start + 99, 2500)
    labels.append(f'{start}-{end}')

# Bin the ranks
full_df['binned_rank'] = pd.cut(full_df['player_rank'], bins=bins, labels=labels, right=False)

# select all features to use in the prediction
df_subset = full_df[selected_features]

# drop all values where nan because knn cannot deal with empty values
df_subset = df_subset.dropna()

# Print numeric columns
# numeric_cols = df_subset.select_dtypes(exclude=[np.number]).columns
# print(numeric_cols)

# encode non numeric values
encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop first to avoid multicollinearity
surface_encoded = encoder.fit_transform(df_subset[['tourney_type', 'surface', 'tourney_level', 'round_1', 'round_2','round_3', 'round_4', 'round_5', 'tourney_round', 'player_hand', 'player_country']])
surface_df = pd.DataFrame(surface_encoded, 
                          columns=encoder.get_feature_names_out(['tourney_type', 'surface', 'tourney_level', 'round_1', 'round_2','round_3', 'round_4', 'round_5', 'tourney_round', 'player_hand', 'player_country']),
                          index=df_subset.index)
df_encoded = pd.concat([df_subset.drop(['tourney_type', 'surface', 'tourney_level', 'round_1', 'round_2','round_3', 'round_4', 'round_5', 'tourney_round', 'player_hand', 'player_country'], axis=1), surface_df], axis=1)

# prepare data
X = df_encoded.drop('binned_rank', axis=1)  # Features
y = df_encoded['binned_rank']  # Target

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale features to normalise
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# set the K as the square root of total rows (this is allegedly a rule of thumb)
total_rows = len(full_df)
train_size = 0.8
k = int(math.sqrt(total_rows * train_size))
print(f"k used: {k}")

# train
knn = KNeighborsClassifier(n_neighbors=k) # using trial and error 77 seemed to be the best score but it's still bad
knn.fit(X_train_scaled, y_train)

# predict
y_pred = knn.predict(X_test_scaled)

# metrics
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Check if there's overfitting
train_score = knn.score(X_train_scaled, y_train)
test_score = knn.score(X_test_scaled, y_test)
print(f"Train accuracy: {train_score:.4f}")
print(f"Test accuracy: {test_score:.4f}")