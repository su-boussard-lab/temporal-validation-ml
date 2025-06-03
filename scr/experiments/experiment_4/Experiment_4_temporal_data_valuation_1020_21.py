# Imports
import sys
import os

sys.path.append(os.getcwd())
from config.config import seed
from config.config import paths_data

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from opendataval.model.api import ClassifierSkLearnWrapper
from opendataval.experiment import ExperimentMediator
from opendataval.dataval import DataOob
from opendataval.dataloader import Register, DataFetcher
from opendataval.dataval import DataBanzhaf, DataOob, KNNShapley, LeaveOneOut
from opendataval.experiment.exper_methods import save_dataval


def main(random_state):
    # Load full data with all features
    X_path = paths_data.features_date
    Y_path = paths_data.labels

    # Read in data
    FEATURES = pd.read_csv(X_path)
    LABELS = pd.read_csv(Y_path)
    Y = LABELS[["PAT_DEID", "ACU_ANY", "YEAR"]]
    X = FEATURES
    X = X.drop(columns=["CHE_TX_DATE", "Unnamed: 0"], axis=1)

    Y = Y.set_index("PAT_DEID")
    X = X.set_index("PAT_DEID")

    # Select the years for defining the train set
    Y_filtered = Y[(Y["YEAR"] >= 2010) & (Y["YEAR"] <= 2020)]
    X_filtered = X.loc[Y_filtered.index]
    X = X_filtered.sort_index()
    Y = Y_filtered.sort_index()

    # Define train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, Y["ACU_ANY"], test_size=0.2, random_state=seed.current_seed
    )

    # Define the held-out, prospective test set
    y_test = LABELS[(LABELS["YEAR"] >= 2021) & (LABELS["YEAR"] <= 2021)][
        ["ACU_ANY", "YEAR", "PAT_DEID"]
    ]
    y_test = (y_test.set_index("PAT_DEID"))["ACU_ANY"]
    X_test = FEATURES.set_index("PAT_DEID")
    X_test = X_test.loc[y_test.index]
    X_test = X_test[X.columns]

    # ---------- Preprocess data ----------
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

    X_val_filtered = selector.transform(X_val)
    X_val_filtered = pd.DataFrame(
        X_val_filtered, columns=X_val.columns[selector.get_support()], index=X_val.index
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

    X_val_imputed = imputer.transform(X_val_filtered)
    X_val_imputed = pd.DataFrame(
        X_val_imputed, columns=X_val_filtered.columns, index=X_val_filtered.index
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_train_scaled = pd.DataFrame(
        X_train_imputed, columns=X_train_imputed.columns, index=X_train_imputed.index
    )

    X_test_scaled = scaler.transform(X_test_imputed)
    X_test_scaled = pd.DataFrame(
        X_test_imputed, columns=X_test_imputed.columns, index=X_test_imputed.index
    )

    X_val_scaled = scaler.transform(X_val_imputed)
    X_val_scaled = pd.DataFrame(
        X_val_imputed, columns=X_val_imputed.columns, index=X_val_imputed.index
    )

    # Stack index and dataframe
    train_num = len(X_train_scaled)
    val_num = len(X_val_scaled)
    num_train_and_val = train_num + val_num
    test_num = len(X_test_scaled)

    train_indices = list(range(train_num))
    val_indices = list(range(train_num, num_train_and_val))
    test_indices = list(range(num_train_and_val, num_train_and_val + test_num))

    # Stack indices
    train_index_df = pd.DataFrame({"index": train_indices})
    val_index_df = pd.DataFrame({"index": val_indices})
    test_index_df = pd.DataFrame({"index": test_indices})
    index_combined = pd.concat([train_index_df, val_index_df, test_index_df])

    # Stack target vector
    y_combined = pd.concat([y_train, y_val, y_test])

    # Stack design matrix
    X_combined = pd.concat([X_train_imputed, X_val_imputed, X_test_imputed])
    index_combined["PAT_DEID"] = X_combined.index
    X_combined = X_combined.reset_index(drop=True)

    # Dataval Setup
    dataset_name = "my_data"
    pd_dataset = Register(dataset_name=dataset_name, one_hot=True).from_data(
        X_combined.to_numpy(), y_combined.to_numpy()
    )

    fetcher = DataFetcher(dataset_name, "...", False).split_dataset_by_indices(
        train_indices=train_indices,
        valid_indices=val_indices,
        test_indices=test_indices,
    )

    # ------------- Begin Data Valuation ---------------
    data_evaluators = [
        KNNShapley(k_neighbors=100),
        DataOob(num_models=100),
        # DataBanzhaf(num_models=50),
        # LeaveOneOut(num_models=50)
    ]

    # Set up predictor model and experiment mediator
    pred_model = ClassifierSkLearnWrapper(
        RandomForestClassifier,
        fetcher.label_dim[0],
        max_depth=50,
        min_samples_split=5,
        n_estimators=500,
        min_samples_leaf=3,
        random_state=seed.current_seed,
        n_jobs=-2,
    )

    exper_med = ExperimentMediator(fetcher, pred_model)
    exper_med = exper_med.compute_data_values(data_evaluators=data_evaluators)

    # Store results
    results_df = exper_med.evaluate(exper_func=save_dataval, save_output=False)
    results_df = results_df.reset_index(drop=False)
    repeated_index = np.tile(X_train_imputed.index, 2)
    results_df["PAT_DEID"] = repeated_index
    results_df = results_df.merge(Y[["YEAR"]], how="left", on="PAT_DEID")
    results_df.to_csv(f"data/processed/data_values_1020_on_21_s42_knn_dataoob.csv")


if __name__ == "__main__":
    main(random_state=seed.current_seed)
