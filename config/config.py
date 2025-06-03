# ---- data paths ----
class paths_data:
    # Raw data
    labels = ...
    features_date = ...
    features_online = ...

    # Feature from permutation importance
    permuted_features_c1 = (
        "data/feature_importance/permutation_importance_c1_351_features_removed.csv"
    )
    permuted_features_c2 = (
        "data/feature_importance/permutation_importance_c2_183_features_removed.csv"
    )
    permuted_features_c3 = (
        "data/feature_importance/permutation_importance_c3_94_features_removed.csv"
    )
    permuted_features_c4 = (
        "data/feature_importance/permutation_importance_c4_35_features_removed.csv"
    )

    # Processed data
    shap_rf_1014_broad = "./data/processed/experiment_2/shapley_values_RF_2010_2014.csv"


# ---- helper functions ----
class tools:
    helper_functions = "./scr/utils/helper_functions.py"
    metrics = "./scr/utils/metrics.py"


# ---- imputation parameters ----
class imputation:
    imputation_strategy = "mean"
    number_neighbors = ""
    # imputation_strategy = "knn"
    # number_neighbors = 5
    # number_neighbors = 15
    # number_neighbors = 100
    # number_neighbors = 1000


# ---- permutation importance parameters ----
class permutation:
    permutation_runs = 50


# ---- random seeds ----
class seed:
    # seed = 3
    # seed = 7
    current_seed = 42
