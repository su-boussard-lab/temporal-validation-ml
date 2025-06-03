""" 
This file plots the evolution of features (experiment 2) over time using a custom heatmap. The file proceeds in three steps
    i) Selects a number of features (e.g. highest ranking features by shapley values)
    ii) Aggregates the values for those features for each mean across all units
    iii) Outputs (normalized) aggregate estimates as a heatmap over time
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
    input_data,
    date_colname,
    selected_features,
    row_names,
    agg='mean',  # Options: ['median', 'std', 'var', 'max']
    standardize_large_features=True,
    standardize_freq_matrix=True,  # Row normalization
    plot_title='Evolution of Features over Time',
    date_label='Time [Years]',
    color_scheme='viridis',
    save_dir=None,
):
    """
    Generates a time-series heatmap of summary statistics.

    Each row represents a feature, and each column represents a time interval.
    The color intensity reflects the aggregated statistic for each feature over time.

    Args:
        data (DataFrame): Input dataset.
        date_colname (str): Name of the column containing dates.
        selected_features (list): List of features to include in the heatmap.
        row_names (list): Labels for the rows (y-axis).
        agg (str or function): Aggregation method to apply (default: 'mean').
        standardize_large_features (bool): Whether to standardize features with large values.
        standardize_freq_matrix (bool): Whether to apply row-wise normalization to the heatmap.
        plot_title (str): Title of the heatmap.
        date_label (str): Label for the x-axis.
        color_scheme (str or Colormap): Colormap used for the heatmap.
        save_dir (str or None): Directory to save the plot; if None, the plot is displayed.
    """
    data = input_data.copy()
    
    feature_names = selected_features["var_name"]
    
    # Standardize large features if enabled
    if standardize_large_features:
        scaler = StandardScaler()
        mask = data[feature_names].abs().max() > 1
        data.loc[:, mask.index[mask]] = scaler.fit_transform(data.loc[:, mask.index[mask]])
    
    # Aggregate data 
    data[date_colname] = data[date_colname].astype(str)
    frequency_matrix_for_heatmap = data.groupby(date_colname)[feature_names].agg(agg)

    # Function to process 'Year_Month'
    def process_year_month(year_month):
        year, month = year_month.split('-')
        return year if month == '01' else ''
    
    # Create a new list for the modified index
    new_index = []
    for i in range(len(frequency_matrix_for_heatmap.index)):
        year, month = frequency_matrix_for_heatmap.index[i].split('-')
        if month == '01':
            new_index.append(year)
        else:
            new_index.append('')
    
    # Replace the DataFrame's index with the new index
    frequency_matrix_for_heatmap.index = new_index
    
    # get frequency matrix
    if standardize_freq_matrix:
        rowmeans = frequency_matrix_for_heatmap.mean(axis="columns")
        rowstd = frequency_matrix_for_heatmap.std(axis="columns")
        rowstd.loc[rowstd == 0] = 1 # if no variance, the mean will just become zero anyway
        frequency_matrix_for_heatmap = frequency_matrix_for_heatmap.sub(rowmeans, axis = "rows")
        frequency_matrix_for_heatmap = frequency_matrix_for_heatmap.div(rowstd, axis = "rows")
    
    # Configure matplotlib font
    mpl.rcParams['font.family'] = 'Arial'
    
    # Define colormap if not provided
    if color_scheme is None:
        color_scheme = LinearSegmentedColormap.from_list('subset_viridis', plt.cm.viridis(np.linspace(0, 1, 256)))
    
    # Generate heatmap plot
    figsize = (35, 18)
    plt.figure(figsize=figsize)
    heatmap = sns.heatmap(
        frequency_matrix_for_heatmap.T,
        cmap=color_scheme,
        yticklabels=row_names,
        annot=False,
        cbar=True,
        cbar_kws={'orientation': 'horizontal', 'location': 'top', "shrink": 0.5}
    )
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=35) 
    cbar.ax.set_position([0.85, 0.3, 0.03, 0.4])
    plt.subplots_adjust(bottom=0.2, left=0.2)
    plt.xticks(rotation=45, fontsize=30)
    plt.yticks(rotation=0, fontsize=25)
    plt.title(plot_title, fontsize=35)
    plt.xlabel(date_label, fontsize=35)
    plt.ylabel(f'Top 15 Features', fontsize=35)
    plt.savefig(save_dir, dpi=300, bbox_inches='tight')
 

# ----------------- Main ----------------- #
def main():

    # Read data
    shap_values_features = pd.read_csv(paths_data.shap_rf_1014_broad)
    X = pd.read_csv(paths_data.features_date_new)

    # Define directory for storing plot
    save_dir = paths_results.figure_3_heatmap

    # Select features used for heatmap, here top 15 features ranked by Shapley
    selected_features = shap_values_features.iloc[0:15,:]
    feature_list = selected_features['var_name'].to_list()
    feature_list.append('YEAR_MONTH')

    # Transform dates so that they can be grouped by months using
    X['YEAR_MONTH'] = (pd.to_datetime(X['CHE_TX_DATE'])).dt.strftime('%Y-%m')
    X = X.drop('CHE_TX_DATE', axis=1)
    X = X.loc[:,feature_list]

    # Define x- and y-axis
    row_names = [f"{row['var_name']}: {round(row['feature_importance_vals'], 2)}" for index, row in selected_features.iterrows()]

    generate_heatmap(
        data = X, 
        date_colname = 'YEAR_MONTH', 
        selected_features = selected_features,
        agg = ['mean'], # [ 'median', 'std', 'var', 'max'],,
        standardize_large_features = True, # Standardizes features
        standardize_freq_matrix = True,
        row_names = row_names,
        plot_title = 'Evolution of Features',
        date_label = 'Time [Months]',
        color_scheme ='viridis',
        save_dir = save_dir
        )

if __name__=='__main__':
    main()
