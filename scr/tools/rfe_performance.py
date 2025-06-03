import numpy as np
import pandas as pd
import random

from sklearn.metrics import roc_auc_score
from scr.tools.metrics import bootstrap

"""
Evaluates the performance of Recursive Feature Elimination (RFE) for a given model and feature set.

This function takes a scaffold DataFrame for the RFE results, a list of features, and a model. It performs
iterative RFE and removes 10 features at a time, while recording the model performance at each step.

Args:
    results_df (pd.DataFrame): A DataFrame containing performance metrics and features in each iteration.
    features (list): A list of feature names used in RFE.
    model (sklearn.base.BaseEstimator): The model used for feature selection and evaluation.
    X_train:
    X_train (pd.DataFrame or np.ndarray): The training feature matrix.
    X_test (pd.DataFrame or np.ndarray): The testing feature matrix.
    Y_train (pd.Series or np.ndarray): The target variable for training.
    Y_test (pd.Series or np.ndarray): The target variable for testing.

Returns:
    pd.DataFrame: A DataFrame summarizing the feature importance rankings and model performance
                    metrics after RFE.
"""


def run_rfe_performance(
    results_df: pd.DataFrame,
    features: list,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    removal_type: str = "ranking",
) -> pd.DataFrame:

    iteration_num = (len(features) - 15) // 10

    for iteration in range(iteration_num):
        if removal_type == "ranking":
            last_features_removed = features[-10:]
            current_features = features[:-10]
            print("Features removed based on ranking")
        elif removal_type == "random":
            last_features_removed = random.sample(features, 10)
            current_features = list(set(features) - set(last_features_removed))
            print("Features removed randomly")
        else:
            raise ValueError("Wrong removal type. Choose either 'ranking' or 'random'.")

        # Run model
        model.fit(X_train[current_features], y_train)
        probs = model.predict_proba(X_test[current_features])[:, 1]

        # Perform evaluation
        performance = roc_auc_score(y_test, probs)
        df_eval = pd.DataFrame({"ground_truth": y_test, "predictions": probs})
        low_95, high_95, _ = bootstrap(df_eval)
        new_results = {
            "AUROC": performance,
            "AUROC_HIGH": high_95,
            "AUROC_LOW": low_95,
            "Active_Features": len(current_features),
            "Last Removed": last_features_removed,
        }
        results_df = results_df.append(new_results, ignore_index=True)

        print(f"Iteration {iteration+1} Results:")
        print(results_df)

        last_features_removed = []
        features = current_features

    # Return result
    return results_df
