import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
from flaml import AutoML

import time
# classifiers
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score # evaluation of classifier models
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron, PassiveAggressiveClassifier, RidgeClassifier, RidgeClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier

# regressors
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error # evaluation of regression models
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.dummy import DummyRegressor

full_df = pd.read_csv('atp_transformed/2000-2024 players_2.csv')

selected_features = ['surface', 
'tourney_level', 
#'tourney_date', 
'match_num', 
'player_seed', 
'player_height', 
'player_country', 
'player_age', 
#'elo_pre_match', 
#'opponent_elo_pre_match', 
'opponent_rank', 
'mean_numb', 
'median_numb', 
'total_numb', 
'mean_diff', 
'median_diff', 
'total_diff', 
'mean_tb_numb', 
'median_tb_numb', 
'total_tb_numb', 
'mean_tb_diff', 
'median_tb_diff', 
'total_tb_diff', 
'days_of_experience',
'career_year', 
'rest_days', 
'set_dominance', 
'tb_dominance', 
'highest_finish_position', 
'minutes_rolling_med_10', 
'draw_size_rolling_med_10', 
'highest_finish_position_rolling_med_10', 
'ace_rolling_mean_10', 
'double_faults_rolling_mean_10', 
'points_on_serve_rolling_mean_10', 
'first_serve_in_rolling_mean_10', 
'1stWon_rolling_mean_10', 
'2ndWon_rolling_mean_10', 
'service_games_rolling_mean_10', 
'break_points_saved_rolling_mean_10', 
'break_points_faced_rolling_mean_10', 
#'elo_pre_match_rolling_mean_10', 
#'opponent_elo_pre_match_rolling_mean_10', 
'set_dominance_rolling_mean_10', 
'tb_dominance_rolling_mean_10', 
'player_rank_rolling_mean_10', 
'mean_numb_rolling_mean_10', 
'median_numb_rolling_mean_10', 
'total_numb_rolling_mean_10', 
'mean_diff_rolling_mean_10', 
'median_diff_rolling_mean_10', 
'total_diff_rolling_mean_10', 
'mean_tb_numb_rolling_mean_10', 
'median_tb_numb_rolling_mean_10', 
'total_tb_numb_rolling_mean_10', 
'mean_tb_diff_rolling_mean_10', 
'median_tb_diff_rolling_mean_10', 
'total_tb_diff_rolling_mean_10', 
'elo_next_match']

# replace missing elo scores with default starting value
full_df['elo_next_match'] = full_df['elo_next_match'].replace(np.nan,1500)
#full_df['rolling_opponent_elo'] = full_df['opponent_elo_pre_match_rolling_mean_10'].replace(np.nan,1500) 
#full_df['rolling_elo'] = full_df['elo_pre_match_rolling_mean_10'].replace(np.nan,1500)

df_subset = full_df[selected_features]

# drop all values where nan
df_subset = df_subset.dropna()

# encode non numeric values
encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop first to avoid multicollinearity
surface_encoded = encoder.fit_transform(df_subset[['surface','player_country','tourney_level']])
surface_df = pd.DataFrame(surface_encoded, 
                          columns=encoder.get_feature_names_out(['surface','player_country','tourney_level']),
                          index=df_subset.index)
df_encoded = pd.concat([df_subset.drop(['surface','player_country','tourney_level'], axis=1), surface_df], axis=1)

# prepare data
X = df_encoded.drop('elo_next_match', axis=1)  # features
y = df_encoded['elo_next_match']  # target

# bin the data here
# n_bins = 20
# y_binned = pd.qcut(y, q=n_bins, labels=[f"Bin{i+1}" for i in range(n_bins)])

# print sample count per bin
# print(y_binned.value_counts())

# reassign y because I'm lazy
# y = y_binned

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale features to normalise
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def do_auto_ml(model_category):

    automl = AutoML()
    
    if model_category == 'class':
        automl.fit(
            X_train_scaled, y_train,
            task="classification",
            time_budget=600,
            metric="f1"
        )

        models = {
            "FLAML Best": automl.model,   # FLAML-selected model
            "RandomForest": RandomForestClassifier(),
            "ExtraTrees": ExtraTreesClassifier(),
            "Bagging": BaggingClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "KNN": KNeighborsClassifier(),
            "GaussianNB": GaussianNB(),
            "BernoulliNB": BernoulliNB(),
            "DecisionTree": DecisionTreeClassifier(),
            "ExtraTree": ExtraTreeClassifier(),
            "SVC": SVC(probability=True),
            "LinearSVC": LinearSVC(),
            "LogisticRegression": LogisticRegression(max_iter=2000),
            "SGDClassifier": SGDClassifier(),
            "Perceptron": Perceptron(),
            "PassiveAggressive": PassiveAggressiveClassifier(),
            "RidgeClassifier": RidgeClassifier(),
            "RidgeClassifierCV": RidgeClassifierCV(),
            "LDA": LinearDiscriminantAnalysis(),
            "QDA": QuadraticDiscriminantAnalysis(),
            "DummyClassifier": DummyClassifier(strategy="most_frequent")
        }

    elif model_category == 'reg':
        automl.fit(
            X_train_scaled,
            y_train,
            task="regression",
            time_budget=600,
            metric="r2"
        )

        models = {
            "FLAML Best": automl.model,   # the model FLAML found
            "RandomForest": RandomForestRegressor(),
            "ExtraTrees": ExtraTreesRegressor(),
            "Bagging": BaggingRegressor(),
            "AdaBoost": AdaBoostRegressor(),
            "KNN": KNeighborsRegressor(),
            "DecisionTree": DecisionTreeRegressor(),
            "ExtraTree": ExtraTreeRegressor(),
            "SVR": SVR(),
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "ElasticNet": ElasticNet(),
            "SGDRegressor": SGDRegressor(),
            "MLPRegressor": MLPRegressor(max_iter=500),
            "DummyRegressor": DummyRegressor(strategy="mean"),
        }

    rows = []

    for name, model in models.items():
        start = time.time()

        try:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            if model_category == 'reg':
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1e-8)))
            
            elif model_category == 'class':
                acc = accuracy_score(y_test, y_pred)
                bal_acc = balanced_accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="macro")

                try:
                    y_prob = model.predict_proba(X_test_scaled)
                    auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
                except:
                    auc = None

            dur = time.time() - start

            if model_category == 'reg':
                rows.append([name, r2, mae, mse, rmse, mape, dur])
            elif model_category == 'class':
                rows.append([name, acc, bal_acc, auc, f1, dur])

        except Exception:
            if model_category == 'reg':
                rows.append([name, None, None, None, None, None, None])
            elif model_category == 'class':
                rows.append([name, None, None, None, None, None])

    df_results_reg = pd.DataFrame(
        rows,
        columns=["Model", "R2", "MAE", "MSE", "RMSE", "MAPE", "Time Taken"] if model_category == 'reg' else ["Model", "Accuracy", "Balanced Accuracy", "ROC AUC", "F1 Score", "Time Taken"]
    )
    
    if model_category == 'reg':
        print(df_results_reg.sort_values("R2", ascending=False))
    else:
        print(df_results_reg.sort_values("Accuracy", ascending=False))

do_auto_ml('reg')

# general metrics
print(f"Min: {y.min()}")
print(f"Max: {y.max()}")
print(f"Range: {y.max() - y.min()}")
print(f"Mean: {y.mean()}")
print(f"Std: {y.std()}")