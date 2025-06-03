import pandas as pd
from sklearn.inspection import permutation_importance
from config.config import seed
from config.config import permutation


def feature_permutation_importance_and_removal(
    model, X_train: pd.DataFrame, X_test: pd.DataFrame, y_test, path: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs feature selection by removing features with negative importance based on permutation importance.

    This function evaluates feature importance using permutation importance, removes features with
    negative importance values from the dataset, and saves the feature importance results to a CSV file.

    Args:
        model: A trained machine learning model used for feature importance evaluation.
        X_train (pd.DataFrame): Training feature matrix.
        X_test (pd.DataFrame): Testing feature matrix.
        y_test: Testing target values to evaluate feature importance on.
        save_path (str): Path to save the feature importance CSV file.

    Returns:
        tuple:
            - X_train_filtered (pd.DataFrame): Training data after removing low-importance features.
            - X_test_filtered (pd.DataFrame): Testing data after removing low-importance features.
    """
    # Obtain permutation importance and store in dataframe
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=permutation.permutation_runs,
        random_state=seed.current_seed,
        n_jobs=-1,
    )
    feature_importances = pd.DataFrame(
        {"FEATURE": X_test.columns, "FEATURE_IMPORTANCE": result.importances_mean}
    ).sort_values(by="FEATURE_IMPORTANCE", ascending=False)

    # Identify and count negative importance features
    negative_features = feature_importances.loc[
        feature_importances["FEATURE_IMPORTANCE"] < 0, "FEATURE"
    ]
    print(f"Number of negative features: {len(negative_features)}")

    # Remove negative importance features from datasets
    X_train_filtered = X_train.drop(columns=negative_features)
    X_test_filtered = X_test.drop(columns=negative_features)

    # Save feature importances to CSV
    feature_importances.to_csv(
        f"{path}_{len(negative_features)}_features_removed.csv", index=False
    )

    return X_train_filtered, X_test_filtered
