import os
import pandas as pd
import pickle
from typing import Tuple, Optional, Union

# Imports for train-test split, remove 0 variance features and scaling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# Imports for imputation
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from config.config import imputation


def scale_data(
    X_train: pd.DataFrame, X_test: pd.DataFrame, X_val: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Scale the numeric data to be mean zero and unit variance.
    Fit only to the training data, apply to both train and test data

    Args:
        X_train (pd.DataFrame): Training data
        X_test (pd.DataFrame): Test data
        X_val (Optional[pd.DataFrame]): Optional additional data to be scaled

    Returns:
        Tuple: Scaled training data, scaled test data, and optionally scaled additional data
    """

    # Scale data - fit on training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if X_val is not None:
        # X_val[numeric_variables] = scaler.transform(X_val[numeric_variables])
        X_val = scaler.transform(X_val)
        return X_train, X_test, X_val

    return X_train, X_test


def subset_data(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    train_dates: list,
    test_dates: list,
    sample_train: bool = False,
    sample_test: bool = False,
    n_train: int = 0,
    n_test: int = 0,
    label: str = "ACU_ANY",
    val_train_period: bool = False,
    seed: Optional[int] = None,
) -> Union[Tuple[pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Subsets and optionally samples the data based on the provided dates.

    Args:
        X (pd.DataFrame): Features.
        Y (pd.DataFrame): Labels.
        train_dates (list): List containing the start and end year of the training period.
        test_dates (list): List containing the start and end year of the test period.
        sample_train (bool): If True, samples the training data.
        sample_test (bool): If True, samples the test data.
        n_train (int): Number of samples for training.
        n_test (int): Number of samples for testing.
        label (str): The label column name.
        val_train_period (bool): If True, returns an additional validation set from the training period.
        seed (int): Seed for random operations.

    Returns:
        Tuple[pd.DataFrame]: Returns tuples of DataFrames containing subsetted and optionally sampled data.
    """
    start_train, end_train = train_dates
    start_test, end_test = (
        test_dates if len(test_dates) > 1 else (test_dates[0], test_dates[0])
    )

    # Convert all dataframe to numeric format if not
    X = X.apply(pd.to_numeric, errors="coerce")

    # Define retrospective training and prospective test sets as defined by user input
    Y_train = Y[(Y["YEAR"] >= start_train) & (Y["YEAR"] <= end_train)]
    Y_test = Y[(Y["YEAR"] >= start_test) & (Y["YEAR"] <= end_test)]

    # Sample train and test data if specified
    if sample_train:
        Y_train = Y_train.groupby("YEAR").apply(
            lambda x: x.sample(n_train, random_state=seed)
        )
    if sample_test:
        Y_test = Y_test.groupby("YEAR").apply(
            lambda x: x.sample(n_test, random_state=seed)
        )

    # Define X_train and X_test based on patient identifier
    X_train = (
        (X.loc[X["PAT_DEID"].isin(Y_train["PAT_DEID"])])
        .set_index("PAT_DEID")
        .sort_index()
    )
    X_test = (
        (X.loc[X["PAT_DEID"].isin(Y_test["PAT_DEID"])])
        .set_index("PAT_DEID")
        .sort_index()
    )

    # Set and sort by index
    Y_train = Y_train.set_index("PAT_DEID")[[label]].sort_index()
    Y_test = Y_test.set_index("PAT_DEID")[[label]].sort_index()

    # Create additional validation test using trainining set data if specified
    if val_train_period:
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_train, Y_train, test_size=0.1, random_state=seed, stratify=Y_train[label]
        )

    sel = VarianceThreshold(threshold=0.0001)
    X_train_sel = sel.fit_transform(X_train)
    X_test_sel = sel.transform(X_test)
    support = sel.get_support()
    selected_features = X_train.columns[support]

    # Reset column names
    X_train = pd.DataFrame(X_train_sel, index=X_train.index, columns=selected_features)
    X_test = pd.DataFrame(X_test_sel, index=X_test.index, columns=selected_features)
    if val_train_period:
        X_val_sel = sel.transform(X_val)
        X_val = pd.DataFrame(X_val_sel, index=X_val.index, columns=selected_features)

    # Select and perform imputation
    if imputation.imputation_strategy == "mean":
        imputer = SimpleImputer(strategy="mean")

    elif imputation.imputation_strategy == "knn":
        imputer = KNNImputer(n_neighbors=imputation.number_neighbors)

    imputer.fit(X_train)
    X_train_imp = pd.DataFrame(
        imputer.transform(X_train), index=X_train.index, columns=X_train.columns
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test), index=X_test.index, columns=X_test.columns
    )

    if val_train_period:
        X_val_imp = pd.DataFrame(
            imputer.transform(X_val), index=X_val.index, columns=X_val.columns
        )
        return X_train_imp, Y_train, X_test_imp, Y_test, X_val_imp, Y_val
    else:
        return X_train_imp, Y_train, X_test_imp, Y_test


def save_model(fitted_model, model_name: str, base_dir: str, experiment_name: str):
    """
    Save a fitted model as a pickle file in the specified directory.

    Args:
    fitted_model: The model object to be saved.
    model_name (str): Name of the model.
    base_dir (str): The base directory where the 'data/interim/models' folder is located.
    experiment_name (str): Name of the experiment.
    """
    models_dir = os.path.join(
        base_dir, "../../", "data/interim/models", experiment_name
    )
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    pickle_file_path = os.path.join(models_dir, f"models_{model_name}.pkl")

    with open(pickle_file_path, "wb") as f:
        pickle.dump(fitted_model, f)


def save_predictions(
    Y_hat,
    Y_test,
    predictions_dir,
    model,
    validated_on_train_period=False,
    Y_hat_train_period=None,
    Y_train_period=None,
):
    """
    Save the predictions to the specified directory.

    Arguments:
        Y_hat: Predictions for the test period
        Y_val: Ground truth for the test period
        base_dir: Base directory for saving the predictions
        experiment_name: Name of the experiment
        model_name: Name of the model
        val_train_period: Whether to validate on the train period
        Y_hat_train_period: Predictions for the train period, if applicable
        Y_train_period: Ground truth for the train period, if applicable
    """
    # Combine Y_hat and Y_val as a DataFrame, then save as a CSV file
    predictions_df = pd.concat(
        [Y_test.reset_index(drop=True), Y_hat.reset_index(drop=True)], axis=1
    )
    predictions_df.columns = ["y_true", "y_hat"]
    predictions_df.to_csv(
        os.path.join(f"{predictions_dir}_{model}_predictions_test_period.csv")
    )

    # If validating on the same train period, save those predictions as well
    if validated_on_train_period:
        if Y_hat_train_period is not None and Y_train_period is not None:
            predictions_train_period_df = pd.concat(
                [
                    Y_train_period.reset_index(drop=True),
                    Y_hat_train_period.reset_index(drop=True),
                ],
                axis=1,
            )
            predictions_train_period_df.columns = ["y_true", "y_hat"]
            predictions_train_period_df.to_csv(
                os.path.join(f"{predictions_dir}_{model}_predictions_train_period.csv")
            )
        else:
            print(
                "Warning: val_train_period is True, but Y_hat_train_period or Y_train_period is not provided."
            )


def save_results(
    output_performance,
    results_dir,
    val_train_period=False,
    output_performance_train_period=None,
):
    """
    Save the results to the specified directory.

    Arguments:
        output_performance: DataFrame containing the output performance for the test period
        results: Directory for saving the results
        val_train_period: Whether to validate on the train period
        output_performance_train_period: DataFrame containing the output performance for the train period, if applicable
    """
    # Create the directory if it does not exist
    output_performance.to_csv(f"{results_dir}_test_period.csv")

    # If validating on the same train period, save those results as well
    if val_train_period:
        if output_performance_train_period is not None:
            output_performance_train_period.to_csv(f"{results_dir}_train_period.csv")
        else:
            print(
                "Warning: val_train_period is True, but output_performance_train_period is not provided."
            )
