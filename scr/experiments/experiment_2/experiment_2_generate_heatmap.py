"""
This file plots the evolution of features over time using a custom heatmap. The file proceeds by
    i) Selects a number of features (e.g. highest ranking features by shapley values)
    ii) Aggregates the values for those features for each mean across all units
    iii) Outputs these aggregate estimates as a heatmap over time
"""

import sys
import os

sys.path.append(os.getcwd())
from config.config import paths_data
from config.config import paths_results

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap


def generate_heatmap(
    data: pd.DataFrame,
    date_colname: str,
    selected_features: pd.DataFrame,
    row_names: list,
    aggregate_function: str = "mean",  # [ 'median', 'std', 'var', 'max'],,
    standardize_large_features: bool = True,  # Standardizes features
    row_standardize: bool = True,
    plot_title: str = "Evolution of Features over Time",
    date_label: str = "Time [Years]",
    color_scheme: str = "viridis",
    save_dir=False,
) -> None:
    """
    Generates a time-series heatmap of summary statistics.
    Each row represents a feature, and each column represents a time interval.
    The color intensity represents the aggregated statistic of observations for each feature over time.

    Args:
        data (DataFrame): Input dataset.
        date_colname (str): Column name for date.
        selected_features (list): Features displayed
        aggregate_function (str or function): Aggregation method, here mean() is being used
        standardize_large_features (bool): Whether to standardize large features.
        row_standardize (bool): Whether to standardize the heatmap frequency matrix
        coef_axis_labels (list): Labels for the y-axis.
        plot_title (str): Title of the plot.
        date_label (str): Label for the x-axis.
        color_scheme (str or Colormap): Colormap to use.
        save_dir (str or None): Directory to save the plot. If None, it displays the plot.

    """
    data = data.copy()

    feature_names = selected_features["var_name"]

    # Standardize large features if enabled
    if standardize_large_features:
        scaler = StandardScaler()
        mask = data[feature_names].abs().max() > 1
        data.loc[:, mask.index[mask]] = scaler.fit_transform(
            data.loc[:, mask.index[mask]]
        )

    # Aggregate data
    data[date_colname] = data[date_colname].astype(str)
    frequency_matrix_for_heatmap = data.groupby(date_colname)[feature_names].agg(
        aggregate_function
    )

    # Function to process 'Year_Month'
    def process_year_month(year_month):
        year, month = year_month.split("-")
        return year if month == "01" else ""

    # Create a new list for the modified index
    new_index = []
    for i in range(len(frequency_matrix_for_heatmap.index)):
        year, month = frequency_matrix_for_heatmap.index[i].split("-")
        if month == "01":
            new_index.append(year)
        else:
            new_index.append("")

    # Replace the DataFrame's index with the new index
    frequency_matrix_for_heatmap.index = new_index

    # get frequency matrix
    if row_standardize:
        # get mean and std
        rowmeans = frequency_matrix_for_heatmap.mean(axis="columns")
        rowstd = frequency_matrix_for_heatmap.std(axis="columns")
        rowstd.loc[rowstd == 0] = (
            1  # if no variance, the mean will just become zero anyway
        )

        frequency_matrix_for_heatmap = frequency_matrix_for_heatmap.sub(
            rowmeans, axis="rows"
        )
        frequency_matrix_for_heatmap = frequency_matrix_for_heatmap.div(
            rowstd, axis="rows"
        )

    # Configure matplotlib font
    mpl.rcParams["font.family"] = "Arial"

    # Define colormap if not provided
    if color_scheme is None:
        color_scheme = LinearSegmentedColormap.from_list(
            "subset_viridis", plt.cm.viridis(np.linspace(0, 1, 256))
        )

    # Calculate plot size
    rows, cols = frequency_matrix_for_heatmap.shape
    figsize = (35, 18)  # Adjust for your plot - use rows and columns for adjustment

    # Generate heatmap plot
    plt.figure(figsize=figsize)
    heatmap = sns.heatmap(
        frequency_matrix_for_heatmap.T,
        cmap=color_scheme,
        yticklabels=row_names,
        annot=False,
        cbar=True,
        cbar_kws={"orientation": "horizontal", "location": "top", "shrink": 0.5},
    )
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=35)  # Adjust color bar font size
    cbar.ax.set_position([0.85, 0.3, 0.03, 0.4])
    plt.subplots_adjust(bottom=0.2, left=0.2)
    plt.xticks(rotation=45, fontsize=30)
    plt.yticks(rotation=0, fontsize=25)
    plt.title(plot_title, fontsize=35)
    plt.xlabel(date_label, fontsize=35)
    plt.ylabel(
        "Top 15 Features", fontsize=35
    )  # Customize depending on what your are plotting
    plt.savefig(save_dir, dpi=300, bbox_inches="tight")


# ----------------- Main ----------------- #
def main():

    # Read data
    path_shap_values = paths_data.shap_rf_1014_broad
    feature_path = paths_data.features_date
    shap_values_features = pd.read_csv(path_shap_values)
    X = pd.read_csv(feature_path).drop(columns=["Unnamed: 0"], axis=1)

    # Define directory for storing plot
    save_dir = paths_results.figure_3_heatmap

    # Select features used for heatmap, here top 20 features ranked by Shapley
    selected_features = shap_values_features.iloc[0:15, :]
    feature_list = selected_features["var_name"].to_list()
    feature_list.append("YEAR_MONTH")

    # Transform dates so that they can be grouped by months using
    X["YEAR_MONTH"] = (pd.to_datetime(X["CHE_TX_DATE"])).dt.strftime("%Y-%m")
    X = X.drop("CHE_TX_DATE", axis=1)
    X = X.loc[:, feature_list]

    # Define x- and y-axis
    row_names = [
        f"{row['var_name']}: {round(row['feature_importance_vals'], 2)}"
        for index, row in selected_features.iterrows()
    ]

    generate_heatmap(
        data=X,
        date_colname="YEAR_MONTH",
        selected_features=selected_features,
        aggregate_function=["mean"],  # [ 'median', 'std', 'var', 'max'],,
        standardize_large_features=True,  # Standardizes features
        row_standardize=True,
        row_names=row_names,
        plot_title="Evolution of Features",
        date_label="Time [Months]",
        color_scheme="viridis",
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
