# Imports
import sys
import os

sys.path.append(os.getcwd())
from config.config import paths_data
from config.config import seed
from config.config import imputation
import pandas as pd
import random

# RF with smaller hyperparameter grid for faster run
from scr.tools.rfe_performance import run_rfe_performance
from scr.models.prediction_models import cv_fit_RF_simple
from sklearn.metrics import roc_auc_score
from scr.tools.metrics import bootstrap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer

def main():
    # Define paths and experiment name for storing results
    project_root = os.getcwd()
    experiment_name = "RFE_permutation_importance"
    dir_permutation_performance = os.path.join(
        project_root,
        f"results/performance_models/experiment_4/{experiment_name}/{experiment_name}",
    )

    # Load features from cycles: c1, c2, c3 - at c1 all features are loaded
    # Select cycle
    cycle = "c3"
    feature_importance = pd.read_csv(paths_data.permuted_features_c3)
    random.seed(seed.current_seed)

    # Load and read in data with all features
    X_path = paths_data.features_date
    Y_path = paths_data.labels
    FEATURES = pd.read_csv(X_path)
    LABELS = pd.read_csv(Y_path)
    Y = LABELS[["PAT_DEID", "ACU_ANY", "YEAR"]]
    X = FEATURES
    X = X.drop(columns=["CHE_TX_DATE", "Unnamed: 0"], axis=1)
    Y = Y.set_index("PAT_DEID")
    X = X.set_index("PAT_DEID")

    # Load the list of features that are in the feature set after j-1 cycles of recursive feature elimination
    features = list(feature_importance["FEATURE"])
    X = X[features]

    # Select the years for defining the train set
    Y_filtered = Y[(Y["YEAR"] >= 2010) & (Y["YEAR"] <= 2020)]
    ids_to_keep = Y_filtered.index
    X_filtered = X.loc[ids_to_keep]

    X = X_filtered.sort_index()
    Y = Y_filtered.sort_index()

    # Define training set
    X_train, _, y_train, _ = train_test_split(
        X, Y["ACU_ANY"], test_size=0.001, random_state=seed.current_seed
    )
    X_train = X
    y_train = Y["ACU_ANY"]

    # Define the test set based on future years
    y_test = LABELS[(LABELS["YEAR"] >= 2021) & (LABELS["YEAR"] <= 2021)][
        ["ACU_ANY", "YEAR", "PAT_DEID"]
    ]
    y_test = (y_test.set_index("PAT_DEID"))["ACU_ANY"]
    X_test = FEATURES.set_index("PAT_DEID")
    X_test = X_test.loc[y_test.index]
    X_test = X_test[X.columns]

    # ----------- Process data -----------
    selector = VarianceThreshold(threshold=0)

    X_train_filtered = selector.fit_transform(X_train)
    X_train_filtered = pd.DataFrame(
        X_train_filtered,
        columns=X_train.columns[selector.get_support()],
        index=X_train.index,
    )
    X_test_filtered = selector.transform(X_test)
    X_test_filtered = pd.DataFrame(
        X_test_filtered,
        columns=X_test.columns[selector.get_support()],
        index=X_test.index,
    )

    # Impute
    imputer = SimpleImputer(strategy=imputation.imputation_strategy)
    X_train_imputed = imputer.fit_transform(X_train_filtered)
    X_train_imputed = pd.DataFrame(
        X_train_imputed, columns=X_train_filtered.columns, index=X_train_filtered.index
    )
    X_test_imputed = imputer.transform(X_test_filtered)
    X_test_imputed = pd.DataFrame(
        X_test_imputed, columns=X_test_filtered.columns, index=X_test_filtered.index
    )

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_train_scaled = pd.DataFrame(
        X_train_imputed, columns=X_train_imputed.columns, index=X_train_imputed.index
    )
    X_test_scaled = scaler.transform(X_test_imputed)
    X_test_scaled = pd.DataFrame(
        X_test_imputed, columns=X_test_imputed.columns, index=X_test_imputed.index
    )

    # Fit model
    model = cv_fit_RF_simple(X_train_scaled, y_train, random_state=seed.current_seed)
    model.fit(X_train_scaled, y_train)

    # Predict on test set
    probs = model.predict_proba(X_test_scaled)[:, 1]
    performance = roc_auc_score(y_test, probs)
    df_eval = pd.DataFrame({"ground_truth": y_test, "predictions": probs})
    low_95, high_95, _ = bootstrap(df_eval)

    # Scaffold output framework
    performance_results = pd.DataFrame(
        {
            "AUROC": [performance],
            "AUROC_HIGH": high_95,
            "AUROC_LOW": low_95,
            "Active_Features": [X_train_scaled.shape[1]],
        }
    )

    performance_results_random = performance_results.copy()
    number_of_negative_features = (feature_importance["FEATURE_IMPORTANCE"] < 0).sum()

    # Perform recursive feature elimination: permutation-based
    performance_results = run_rfe_performance(
        results_df=performance_results,
        features=features,
        model=model,
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        removal_type="ranking",
    )
    performance_results.to_csv(
        f"{dir_permutation_performance}_step_size_10_1020_21_{cycle}_{number_of_negative_features}_test.csv"
    )

    # Perform recursive feature elimination: random
    random.shuffle(features)
    performance_results_random = run_rfe_performance(
        results_df=performance_results_random,
        features=features,
        model=model,
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        removal_type="random",
    )
    performance_results_random.to_csv(
        f"{dir_permutation_performance}_step_size_10_1020_21_{cycle}_random_test.csv"
    )


if __name__ == "__main__":
    main()
