"""
Helper functions for boostrap, results printing, AUROC and AUPRC
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def bootstrap(df, func=roc_auc_score):
    """Bootstrap for calculating the confidence interval of a metric function
    Args:
        df (pd.DataFrame): dataframe containing 'predictions' and ' outcomes'
        func (function): metric function that takes (y_true, y_pred) as parameters
    Returns:
        lower, upper 95% confidence interval
    """
    metric = []
    for i in range(1000):
        sample = df.sample(n=df.shape[0] - int(df.shape[0] / 5), random_state=i)
        metric.append(round(func(sample["ground_truth"], sample["predictions"]), 5))
    return (
        np.percentile(np.array(metric), 2.5),
        np.percentile(np.array(metric), 97.5),
        metric,
    )


def create_results_df(
    train: str,
    performance_results: pd.DataFrame,
    model_name: str,
    years: tuple,
    sample_size_train: int,
    sample_size_test: int,
    label_type: str,
    n_features: int,
) -> pd.DataFrame:
    """
    Create a DataFrame containing the performance metrics of a model.

    Args:
    exp_result (pd.DataFrame): DataFrame containing the calculated performance metrics.
    model_name (str): Name of the model.
    years (tuple): Tuple containing the training and test years.
    sample_size_train (int): Number of samples in the training set.
    sample_size_test (int): Number of samples in the test/validation set.
    label_type (str): Type of label used.

    Returns:
    pd.DataFrame: DataFrame containing the performance results.
    """
    metrics = ["AUROC", "AUPRC"]

    # Store results on heldout validation set from training period
    if train == "validation_train_period":
        results = {
            "Model": model_name,
            "N_Features": n_features,
            "Training_Years": f"{years[0][0]}-{years[0][1]}",
            "Test_Years": f"{years[0][0]}-{years[0][1]}",
            "Sample_Size_Train": sample_size_train,
            "Sample_Size_Test": sample_size_test,
            "Label_Type": label_type,
        }
    elif train == "validation_test_period":
        results = {
            "Model": model_name,
            "N_Features": n_features,
            "Training_Years": f"{years[0][0]}-{years[0][1]}",
            "Test_Years": f"{years[1][0]}-{years[1][1]}",
            "Sample_Size_Train": sample_size_train,
            "Sample_Size_Test": sample_size_test,
            "Label_Type": label_type,
        }

    for metric in metrics:
        matching_rows = performance_results.loc[performance_results["Metric"] == metric]
        if not matching_rows.empty:
            results[f"{metric}"] = matching_rows["Value"].values[0]
            results[f"{metric}_Low_95"] = matching_rows["Low_95"].values[0]
            results[f"{metric}_High_95"] = matching_rows["High_95"].values[0]
            results[f"{metric}_bootstraps"] = matching_rows["bootstraps"].values
        else:
            # Handle the case where no matching rows are found
            results[f"{metric}"] = np.nan
            results[f"{metric}_Low_95"] = np.nan
            results[f"{metric}_High_95"] = np.nan
            results[f"{metric}_bootstraps"] = np.nan

    return pd.DataFrame(results, index=[0])


# Create a function that takes y_true and y_pred as parameters and returns a dataframe with the performance metrics
def get_performance(y_true, y_pred) -> pd.DataFrame:
    """Print all the performance metrics (AUROC, AUPRC)
    Args:
        y_true (np.ndarray): true labels
        y_pred (np.ndarray): predicted scores
    Returns:
        None
    """
    # Convert y_true and Y_pred to numpy arrays
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)

    # Define threshold and metrics
    metrics = [roc_auc_score, average_precision_score]
    results_df = pd.DataFrame(
        columns=["Metric", "Value", "Low_95", "High_95", "bootstraps"]
    )
    results_df["Metric"] = ["AUROC", "AUPRC"]

    # Establish data frame for continous evaluations
    labels_predictions = {"predictions": y_pred, "ground_truth": y_true}
    predictions_to_eval = pd.DataFrame(data=labels_predictions)

    # loop through the list of metrics and index the rows of the dataframe to report results
    for i in range(len(metrics)):

        metric = metrics[i]
        metric_value = metric(y_true, y_pred)
        low_95, high_95, bootstrap_list = bootstrap(predictions_to_eval, func=metric)

        # Now add results to the dataframe
        results_df["Value"][i] = round(metric_value, 5)
        results_df["Low_95"][i] = round(low_95, 5)
        results_df["High_95"][i] = round(high_95, 5)
        results_df["bootstraps"][i] = bootstrap_list

    # Return results
    return results_df


def get_auroc_splityears(y_true, y_pred, y_years, val_years):
    """Computes AUROC by year
    y_true: true values corresponding to get_performance y_true (pandas column)
    y_hat: probability estimates (scores) corresponding to get_performance y_hat (pandas column)
    y_years: year corresponding to the value of y_true (pandas column)
    val_years: start and end of years included in the test set
    """

    # Copy everything to prevent modification
    y_true_auroc = y_true.copy()
    y_pred_auroc = y_pred.copy()
    y_years_auroc = y_years.copy()
    val_years_auroc = val_years

    # Init empty list
    AUROC_allyears = []
    low_95_allyears = []
    high_95_allyears = []

    # Combine columns to single pandas df
    ys_and_years = pd.concat([y_years_auroc, y_true_auroc], axis=1)
    ys_and_preds_and_years = pd.concat(
        [ys_and_years.reset_index(drop=True), y_pred_auroc], axis=1
    )
    ys_and_preds_and_years.columns = ["YEAR", "Y_val", "Y_hat"]

    for curr_year in range(min(val_years_auroc[1]), max(val_years_auroc[1]) + 1):

        # Subset to current year
        ys_and_preds_and_years_curr = ys_and_preds_and_years.loc[
            ys_and_preds_and_years["YEAR"] == curr_year
        ]

        # extract values and preds
        Y_val_curr_year = ys_and_preds_and_years_curr["Y_val"]
        Y_hat_curr_year = ys_and_preds_and_years_curr["Y_hat"]

        # Compute AUROC
        AUROC, low_95, high_95 = get_performance(
            y_true=Y_val_curr_year, y_pred=Y_hat_curr_year
        )

        # Append to running list
        AUROC_allyears.append(AUROC)
        low_95_allyears.append(low_95)
        high_95_allyears.append(high_95)

    # convert lists to single string for quick and dirty saving to csv
    AUROC = "_".join(list(map(str, AUROC_allyears)))
    low_95 = "_".join(list(map(str, low_95_allyears)))
    high_95 = "_".join(list(map(str, high_95_allyears)))
    return AUROC, low_95, high_95
