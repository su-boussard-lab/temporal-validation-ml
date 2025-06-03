"""
This script trains models on a retrospective training cohort and evaluates them
across multiple training/test year combinations for future validation.

Workflow:
1. Data Processing: Subsetting by date, sampling, scaling, and feature engineering.
2. Model Training: Fitting models to the training data.
3. Model Evaluation: Testing models on held-out test data using performance metrics.
"""

# Import packages
import os
import sys

sys.path.append(os.getcwd())
from config.config import paths_data
from config.config import seed
from config.config import imputation

import numpy as np
import pandas as pd

# Import helper functions: scaling, subsetting, time weights
from scr.tools.metrics import get_performance
from scr.tools.metrics import create_results_df
from scr.tools.helper_functions import scale_data
from scr.tools.helper_functions import subset_data
from scr.tools.helper_functions import save_predictions
from scr.tools.helper_functions import save_results

# Import models
from scr.models.prediction_models import cv_fit_LASSO
from scr.models.prediction_models import cv_fit_GBM
from scr.models.prediction_models import cv_fit_RF


# ----------------- Fit model ----------------- #
def fit_models_predict(
    experiment_name: str,
    X,
    Y,
    val_train_period: bool,
    models_list: list,
    label_types: list,
    train_years: list,
    test_years: list,
    sample_train: bool = False,
    sample_test: bool = False,
    n_train: int = None,
    n_test: int = None,
) -> None:
    """
    This functions runs a list of models on a selection of data using a selection of models

    Args:
        Name of experiment: experiment_name
        Feature matrix: X
        Labels: Y
        List of models: models_list
        List of labels: label_types
        List of training years: train years
        List of test years: test_years
        Indicator for whether training data is sampled: sample_train
        Indicator for whether test data is sampled: sample_test
        Sample size used for sampling train data: n_train
        Sample size used for sampling test data: n_test

    Returns:
        None
    """
    # Define output for models and predictions
    output_models = []
    output_predictions = []
    output_predictions_same = []

    # Define base directory
    project_root = os.getcwd()
    predictions_dir = os.path.join(project_root, "data/predictions/", experiment_name)
    results_dir_prediction_performance = os.path.join(
        project_root, "results/performance_models/experiment_4/", experiment_name
    )

    # Initialize dataframes for storing performance results
    columns = [
        "Model",
        "Training_Years",
        "Test_Years",
        "Sample_Size_Train",
        "Sample_Size_Test",
        "Label_Type",
        "AUROC",
        "AUROC_Low_95",
        "AUROC_High_95",
        "AUROC_bootstraps",
        "AUPRC",
        "AUPRC_Low_95",
        "AUPRC_High_95",
        "AUPRC_bootstraps",
        "N_Features",
    ]
    output_performance_train_period = pd.DataFrame(columns=columns)
    output_performance_test_period = pd.DataFrame(columns=columns)

    # Labels
    for label_type in label_types:

        #  Training and test years
        for years in zip(train_years, test_years):
            print(f"\n {experiment_name}: {years}")

            # Subset and scale data
            if val_train_period:
                # Subset, remove 0 variance, features impute
                X_train, Y_train, X_test, Y_test, X_val, Y_val = subset_data(
                    X,
                    Y,
                    train_dates=years[0],
                    test_dates=years[1],
                    sample_train=sample_train,
                    sample_test=sample_test,
                    n_train=n_train,
                    n_test=n_test,
                    label=label_type,
                    val_train_period=val_train_period,
                    seed=seed.current_seed,
                )
                X_train, X_test, X_val = scale_data(X_train, X_test, X_val)

            else:
                # Subset, remove 0 variance, features impute
                X_train, Y_train, X_test, Y_test = subset_data(
                    X,
                    Y,
                    train_dates=years[0],
                    test_dates=years[1],
                    sample_train=sample_train,
                    sample_test=sample_test,
                    n_train=n_train,
                    n_test=n_test,
                    label=label_type,
                    val_train_period=val_train_period,
                    seed=seed.current_seed,
                )
                X_train, X_test = scale_data(X_train, X_test)

            for model_func in models_list:
                # Fit model and get predictions
                fitted_model = model_func(
                    X_train, Y_train, label_type, random_state=seed.current_seed
                )
                model_name = f"{model_func.__name__}_{label_type}_train_{years[0][0]}-{years[0][1]}_val_{years[1][0]}-{years[1][1]}_{seed.current_seed}"
                Y_hat = fitted_model.predict_proba(X_test)[:, 1]
                Y_hat = pd.DataFrame(Y_hat, columns=[model_name])

                # Optional: validate model on heldout test data from the same time period as training data
                if val_train_period:
                    Y_hat_train_period = fitted_model.predict_proba(X_val)[:, 1]
                    Y_hat_train_period = pd.DataFrame(
                        Y_hat_train_period, columns=[model_name]
                    )
                    output_predictions_same.append(Y_hat_train_period)

                    performance_train_period = get_performance(
                        Y_val, Y_hat_train_period
                    )
                    results_on_train_period = create_results_df(
                        train="validation_train_period",
                        performance_results=performance_train_period,
                        model_name=model_func.__name__,
                        years=years,
                        sample_size_train=len(X_train),
                        sample_size_test=len(X_val),
                        label_type=label_type,
                        n_features=X_train.shape[1],
                    )
                    output_performance_train_period = pd.concat(
                        [output_performance_train_period, results_on_train_period]
                    )

                # Append predictions and models
                output_predictions.append(Y_hat)
                output_models.append(fitted_model)

                # Performance test period
                performance_test_period = get_performance(Y_test, Y_hat)
                results_on_test_period = create_results_df(
                    train="validation_test_period",
                    performance_results=performance_test_period,
                    model_name=model_func.__name__,
                    years=years,
                    sample_size_train=len(X_train),
                    sample_size_test=len(X_test),
                    label_type=label_type,
                    n_features=X_train.shape[1],
                )
                output_performance_test_period = pd.concat(
                    [output_performance_test_period, results_on_test_period]
                )

                # Save predictions and performance
                if val_train_period:
                    save_predictions(
                        Y_hat=Y_hat,
                        Y_test=Y_test,
                        predictions_dir=predictions_dir,
                        model=model_func.__name__,
                        validated_on_train_period=val_train_period,
                        Y_hat_train_period=Y_hat_train_period,
                        Y_train_period=Y_val,
                    )

                else:
                    save_predictions(
                        Y_hat=Y_hat,
                        Y_test=Y_test,
                        predictions_dir=predictions_dir,
                        model=model_func.__name__,
                    )

                # Compare with blind prediction, i.e. against majority case as sanity check
                majority_class_0 = np.zeros_like(Y_test)
                majority_0_test_experiment = get_performance(Y_test, majority_class_0)
                majority_0_test_performance = create_results_df(
                    train="validation_test_period",
                    performance_results=majority_0_test_experiment,
                    model_name=f"{model_func.__name__}_vs_MajorityClass_Baseline",
                    years=years,
                    sample_size_train=len(X_train),
                    sample_size_test=len(X_test),
                    label_type=label_type,
                    n_features=X_train.shape[1],
                )
                output_performance_test_period = pd.concat(
                    [output_performance_test_period, majority_0_test_performance]
                )

    # Save output
    if val_train_period:
        save_results(
            output_performance=output_performance_test_period,
            results_dir=results_dir_prediction_performance,
            val_train_period=val_train_period,
            output_performance_train_period=output_performance_train_period,
        )
    else:
        save_results(
            output_performance=output_performance_test_period,
            results_dir=results_dir_prediction_performance,
        )

    print(f"\n {experiment_name} complete.")


# ----------------- Main ----------------- #
def main(random_state: int = seed.current_seed) -> None:
    """Main function to run script
    Args:
        random_state (int): random state
    Returns:
        None
    """
    # Read in data and remove columns that are not needed
    X_path = paths_data.features_date
    Y_path = paths_data.labels

    X = pd.read_csv(X_path).drop(columns=["CHE_TX_DATE", "Unnamed: 0"], axis=1)
    Y = pd.read_csv(Y_path)

    # Obtain feature subset from
    feature_subset = (pd.read_csv(paths_data.permuted_features_c4)["FEATURE"]).to_list()
    feature_subset.append("PAT_DEID")
    reduced_X = X[feature_subset]

    # Define list of labels
    labels = ["ACU_ANY"]
    models = [cv_fit_LASSO, cv_fit_GBM, cv_fit_RF]

    # Define imputation type through config file
    imputation_type = imputation.imputation_strategy + str(imputation.number_neighbors)

    # ------- Experiment 4c-1: Full Data ------
    experiment_name = f"experiment_4c_singlesplit_final_full_data_{imputation_type}_1020_22_s{seed.current_seed}"
    train_years = [[2010, 2020]]
    test_years = [[2022, 2022]]

    # Run model
    print(f"----- START: {experiment_name} ----- \n")
    fit_models_predict(
        experiment_name=experiment_name,
        X=X,
        Y=Y,
        val_train_period=True,
        models_list=models,
        label_types=labels,
        train_years=train_years,
        test_years=test_years,
        sample_train=False,
        sample_test=False,
        n_train=None,
        n_test=None,
    )

    # ------- Experiment 4c-2: Feature Reduction Data ------
    experiment_name = f"experiment_4c_singlesplit_final_feature_reduction_{imputation_type}_1020_22_s{seed.current_seed}"
    train_years = [[2010, 2020]]
    test_years = [[2022, 2022]]

    # Run model
    print(f"----- START: {experiment_name} ----- \n")
    fit_models_predict(
        experiment_name=experiment_name,
        X=reduced_X,
        Y=Y,
        val_train_period=True,
        models_list=models,
        label_types=labels,
        train_years=train_years,
        test_years=test_years,
        sample_train=False,
        sample_test=False,
        n_train=None,
        n_test=None,
    )

    # ------- Experiment 4c-3: Data Reduction ------
    experiment_name = f"experiment_4c_singlesplit_final_all_features_data_reduction_{imputation_type}_1020_22_s{seed.current_seed}"
    train_years = [[2014, 2020]]
    test_years = [[2022, 2022]]

    print(f"----- START: {experiment_name} ----- \n")
    fit_models_predict(
        experiment_name=experiment_name,
        X=X,
        Y=Y,
        val_train_period=True,
        models_list=models,
        label_types=labels,
        train_years=train_years,
        test_years=test_years,
        sample_train=False,
        sample_test=False,
        n_train=None,
        n_test=None,
    )

    # ------- Experiment 4c-4: Data Reduction ------
    experiment_name = f"experiment_4c_singlesplit_final_feature_reduction_data_reduction_{imputation_type}_1020_22_s{seed.current_seed}"
    train_years = [[2014, 2020]]
    test_years = [[2022, 2022]]

    print(f"----- START: {experiment_name} ----- \n")
    fit_models_predict(
        experiment_name=experiment_name,
        X=reduced_X,
        Y=Y,
        val_train_period=True,
        models_list=models,
        label_types=labels,
        train_years=train_years,
        test_years=test_years,
        sample_train=False,
        sample_test=False,
        n_train=None,
        n_test=None,
    )

    print("Script complete.")


if __name__ == "__main__":
    main(random_state=seed.current_seed)
