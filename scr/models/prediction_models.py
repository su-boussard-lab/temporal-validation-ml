#!/usr/bin/env python3
import numpy as np
import pandas as pd

# Import for ML models
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# ----------------- XGBoost ----------------- #
def cv_fit_GBM(
    X_train: np.ndarray, y_train: np.ndarray, label_type: str, random_state: int
) -> XGBClassifier:
    """
    Grid-search CV for GBM model

    Args:
        X_train (np.ndarray): training features
        y_train (np.ndarray): training labels
        label_type (str): label type
        random_state (int): random state

    Returns:
        GBM: fitted model with parameters selected from CV
    """
    if label_type not in y_train:
        raise ValueError(f"{label_type} not found in y_train")
    y_train = y_train[label_type]

    parameters = {
        "max_depth": [3, 5],
        "n_estimators": [300, 500],
        "learning_rate": [0.1],
        "min_child_weight": [5, 10],
        "colsample_bytree": [0.8],
    }

    model = XGBClassifier(objective="binary:logistic", seed=random_state)
    gbm_CV = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        cv=10,
        scoring="roc_auc",
        verbose=1,
        n_jobs=-1,
    )

    # Fit model
    gbm_CV.fit(X_train, y_train)
    XGBoost_tuned = gbm_CV.best_estimator_
    return XGBoost_tuned


# ----------------- LASSO ----------------- #
def cv_fit_LASSO(
    X_train: pd.DataFrame, y_train: pd.Series, label_type: str, random_state: int
) -> LogisticRegression:
    """
    Grid-search CV for LASSO

    Args:
        X_train (pd.DataFrame): training features
        y_train (pd.Series): training labels
        label_type (str): label type
        random_state (int): random state

    Returns:
        LogisticRegression: fitted model with parameters selected from CV
    """
    if label_type not in y_train:
        raise ValueError(f"{label_type} not found in y_train")
    y_train = y_train[label_type]

    model = LogisticRegression(
        penalty="l1", max_iter=1700, solver="liblinear", random_state=random_state
    )
    clf = GridSearchCV(
        estimator=model,
        param_grid={"C": np.logspace(-2, 0, 6)},
        cv=10,
        scoring="roc_auc",
        verbose=1,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    lasso_tuned = clf.best_estimator_
    return lasso_tuned


# ----------------- Random Forest ----------------- #
def cv_fit_RF(
    X_train: pd.DataFrame, y_train: pd.Series, label_type: str, random_state: int
) -> RandomForestClassifier:
    """
    Grid-search CV for Random Forest

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        label_type (str): Label type
        random_state (int): Random state

    Returns:
        RandomForestClassifier: fitted model with parameters selected from CV
    """
    if label_type not in y_train:
        raise ValueError(f"{label_type} not found in y_train")
    y_train = y_train[label_type]

    param_grid = {
        "n_estimators": [300, 600],  # [150, 300, 600], #[300, 450, 600],
        "max_depth": [10, 30],  # , 50],
        "min_samples_split": [3, 5],
        "min_samples_leaf": [2, 4],
    }

    model = RandomForestClassifier(random_state=random_state)
    clf = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=10,
        scoring="roc_auc",
        verbose=1,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf.best_estimator_


# ----------------- Random Forest with reduced hyerparameter grid ----------------- #
def cv_fit_RF_simple(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int
) -> RandomForestClassifier:
    """
    Grid-search CV for Random Forest

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        label_type (str): Label type
        random_state (int): Random state

    Returns:
        RandomForestClassifier: fitted model with parameters selected from CV
    """
    # Increase grid if needed
    param_grid = {
        "n_estimators": [500],
        "max_depth": [50],
        "min_samples_split": [5],
        "min_samples_leaf": [1, 3],
    }

    model = RandomForestClassifier(random_state=random_state)
    clf = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=10,
        scoring="roc_auc",
        verbose=1,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf.best_estimator_
