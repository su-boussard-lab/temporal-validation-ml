# Imports
import sys
import os

sys.path.append(os.getcwd())
from config.config import paths_data
from config.config import seed
import pandas as pd

from scr.models.feature_permutation_importance import (
    feature_permutation_importance_and_removal,
)
from scr.models.prediction_models import cv_fit_RF_simple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer


def main():

    # Load data
    X_path = paths_data.features_date_new
    Y_path = paths_data.labels_new
    FEATURES = pd.read_csv(X_path)
    LABELS = pd.read_csv(Y_path)

    # Subset dataframes
    Y = LABELS[["PAT_DEID", "ACU_ANY", "YEAR"]]
    X = FEATURES
    X = X.drop(columns=["CHE_TX_DATE", "Unnamed: 0"], axis=1)

    Y = Y.set_index("PAT_DEID")
    X = X.set_index("PAT_DEID")

    # Select the years for defining the train set
    Y_filtered = Y[(Y["YEAR"] >= 2010) & (Y["YEAR"] <= 2020)]
    ids_to_keep = Y_filtered.index
    X_filtered = X.loc[ids_to_keep]
    X = X_filtered.sort_index()
    Y = Y_filtered.sort_index()

    # Define training set: no validation set needed from same period
    X_train, _, y_train, _ = train_test_split(
        X, Y["ACU_ANY"], test_size=0.001, random_state=42
    )
    X_train = X
    y_train = Y["ACU_ANY"]

    # Define test set based on future years
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
    imputer = SimpleImputer(strategy="mean")
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

    # Loop over the number of cycles
    cycles = ["c1", "c2", "c3", "c4"]

    # Define
    X_train_new = X_train_scaled.copy()
    X_test_new = X_test_scaled.copy()
    model = cv_fit_RF_simple(X_train_scaled, y_train, random_state=seed.current_seed)

    for cycle in cycles:
        model.fit(X_train_new, y_train)
        path_name = f"data/feature_importance/permutation_importance_{cycle}"
        X_train_new, X_test_new = feature_permutation_importance_and_removal(
            model=model,
            X_train=X_train_new,
            X_test=X_test_new,
            y_test=y_test,
            path=path_name,
        )


if __name__ == "__main__":
    main()
