#!/usr/bin/env python3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataframe with diagnosis and ACU events
op_df = ...

# Define time to ACU in days
op_df["CHE_TO_HOSP"] = (op_df["ADMISSION_TIME"] - op_df["CHE_TX_DATE"]).dt.days

# Remove any duplicate diagnoses
op_df = op_df[['PAT_DEID', 'CHE_TX_DATE', 'CHE_TO_HOSP', 'OP35_CATEGORY', 'LABEL_TYPE']].drop_duplicates()

# Split into retrospective and propsective data
op_df_10to18 = op_df.loc[(op_df['CHE_TX_DATE']>='2010-01-01 00:00:00') & (op_df['CHE_TX_DATE']<='2018-12-31 23:59:59')]
op_df_19to22 = op_df.loc[(op_df['CHE_TX_DATE']>='2019-01-01 00:00:00') & (op_df['CHE_TX_DATE']<='2022-12-31 23:59:59')]

# Subset to ACU events within 180 days; convert to percentages
ACU_10_18 = op_df_10to18[op_df_10to18["CHE_TO_HOSP"]<=180].OP35_CATEGORY.value_counts(normalize=True)[::-1]
ACU_19_22 = op_df_19to22[op_df_19to22["CHE_TO_HOSP"]<=180].OP35_CATEGORY.value_counts(normalize=True)[::-1]

# Create dataframe for plotting
ACU_combined = pd.concat([ACU_10_18, ACU_19_22], axis=1)
ACU_combined.columns = ['2010-2018', '2019-2022']

# Plot figure
sns.set(
    style="whitegrid",
    font_scale=1.2,
    rc={
        "figure.figsize": (10, 6),
        "font.size": 20,
        "axes.labelsize": 14}
    )

ACU_combined.plot(kind="barh", figsize=(8, 6), width=0.6, title="Frequency of diagnoses associated with Acute Care Utilization (ACU)", color =['#2D708EFF', '#F8766D'])
plt.xlabel('Frequency [%]', fontname='Arial')
plt.ylabel('Diagnoses', fontname='Arial')
plt.gcf().set_edgecolor('black')
for spine in plt.gca().spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(0.7)

# Save output
plt.savefig('results/figure_3/figure_3c_frequency_diagnoses.pdf', bbox_inches='tight', transparent=True)