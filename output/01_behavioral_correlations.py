#

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 01: Behavioral Profiling & Aptitude-Multilingualism Correlations

Description:
    1. Evaluates sample descriptives for language profile and aptitude data.
    2. Calculates pairwise correlation matrices between aptitude, language entropy, 
       and count of advanced languages.
    3. Generates publication-ready joint plots and multivariate pairplots.

Inputs:
    - ./data/participant_lang_profile.tsv
    - ./data/participants_motivation_aptitude.tsv

Outputs:
    - ./output/plots/ (Correlation plots and regression grids)
    
Author:IB
Last update: 2026.07.21
"""

# %% Packages
import os
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pingouin as pg


# %% Path Setup & Directory Creation
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "behavioural_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# %% Data Loading
lang = pd.read_table(os.path.join(DATA_DIR, "participant_lang_profile.tsv"))
apt = pd.read_table(os.path.join(DATA_DIR, "participants_aptitude_individual_scores_imputed.tsv"))
leap =pd.read_table(os.path.join(DATA_DIR, "leapq_entropy_competence_filtered.tsv"))


df = pd.merge(lang, apt, on='participant_id')

# Define core variables and covariate
x_var = 'aptitude_total_z'
y_var = 'entropy_competence_speak'
covar = 'advanced_lang_total_count'

# Drop rows with missing values in any of the target columns
#df = df[[x_var, y_var, covar]].dropna().copy()


#%%descriptives

descriptives=df.describe().T.round(2)
print(descriptives)

# 2. Select and rename columns to match APA standards
# APA tables typically use M for Mean and SD for Standard Deviation
apa_table = descriptives[['count', 'mean', 'std', 'min', 'max']].copy()
apa_table.columns = ['n', 'M', 'SD', 'Min', 'Max']

# 3. Round to 2 decimal places (APA standard)
apa_table = apa_table.round(2)

# 4. Clean up the index names for a professional look
apa_table.index = [col.replace('_', ' ').title() for col in apa_table.index]

print(apa_table)

#%% corr and partial correlation apatitude and multi
# 1. Define your variables
# Primary variables of interest
x = 'aptitude_total_z' #or motivation_total
y = 'entropy_competence_speak' # or 'elective_z_delta'
df['Sex_binary'] = df['Sex'].map({'m': 1, 'f': 0})
df['Sex_binary'] = pd.to_numeric(df['Sex_binary'], errors='coerce')
# Covariates to control for
covariates = ['Sex_binary','age', 'edu' ]#'Hand_index'

# 2. Standard Correlation (Zero-order)
# This is the "correlation as is"
standard_corr = pg.corr(df[x], df[y])

# 3. Partial Correlation
# This calculates the correlation between x and y while removing the 
# influence of Age, Edu, and Hand Index.
partial_corr = pg.partial_corr(data=df, x=x, y=y, covar=covariates)

# 4. Create a comparison table for APA reporting
results = pd.DataFrame({
    'Analysis': ['Standard (Raw)', 'Partial (Controlled)'],
    'n': [standard_corr['n'].iloc[0], partial_corr['n'].iloc[0]],
    'r': [standard_corr['r'].iloc[0], partial_corr['r'].iloc[0]],
    'p_val': [standard_corr['p_val'].iloc[0], partial_corr['p_val'].iloc[0]],
    'CI95': [standard_corr['CI95'].iloc[0], partial_corr['CI95'].iloc[0]]
})

print("--- Correlation Comparison ---")
print(results.round(3))


# 2. Calculate Residuals (The "Controlled" variance)
def get_residuals(data, target, covars):
    # Perform linear regression and return the residuals
    X = np.column_stack([np.ones(len(data)), data[covars]])
    y = data[target]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ beta

# X residuals (Aptitude controlled for demographics)
x_res = get_residuals(df, x, covariates)
# Y residuals (Entropy controlled for demographics)
y_res = get_residuals(df, y, covariates)

# 3. Get Partial Correlation Stats
p_corr_res = pg.partial_corr(data=df, x=x, y=y, covar=covariates)
r_partial = p_corr_res['r'].iloc[0]
p_partial = p_corr_res['p_val'].iloc[0]

# 4. Plotting
sns.set_style("white")
# Create a temporary dataframe for residuals to use with JointGrid
df_res = pd.DataFrame({'x_res': x_res, 'y_res': y_res})

g = sns.JointGrid(data=df_res, x='x_res', y='y_res',
                  xlim=(-6.8,6), ylim=(-1.5,1.5))


# Regression line through residuals
g.plot_joint(sns.regplot, color='steelblue', 
             scatter_kws={'alpha': 0.5, 'edgecolor': 'white'},
             line_kws={'color': 'dodgerblue', 'lw': 2})

g.plot_marginals(sns.kdeplot, fill=True, color='steelblue')


# Embedding partial correlation stats
stats_text = f'Partial r = {r_partial:.1f} \np = {p_partial:.2f}'# if you want to add sig \np = {p_partial:.3f}
g.ax_joint.annotate(
    stats_text, 
    xy=(0.05, 0.95), 
    xycoords='axes fraction',
    ha='left', 
    va='top',
    fontsize=12,
    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5, ec='gray')
)

g.ax_joint.grid(True, linestyle='--', alpha=0.5)

# Labeling as Residuals (Standard for reporting partial correlations)
g.set_axis_labels("Aptitude score (Residuals)", "Multilingualism (Residuals)")
g.fig.suptitle("Partial Correlation (Controlled for Age, Edu, Sex)", y=1.03)

sns.despine(fig=g.fig)
#plt.show()

# 5. Save
# Note: Ensure 'timestamp' and 'output_dir' are defined in your environment
filename = f"{x}_{y}_partial_corr_{timestamp}.png"
save_path = os.path.join(OUTPUT_DIR, filename)
#g.savefig(save_path, bbox_inches='tight', dpi=300)



#%%linear model to report explained variance
X = df[[x] + covariates]
Y = df[y]

lm = pg.linear_regression(X, Y)

# 3. Display Results
# -------------------------
print(f"--- Multiple Linear Regression: Predicting {y} ---")
# Rounding to 3 decimals for clarity
print(lm.round(3))

#%% Single Plot: Overlaid Twin-Axis (Speaking Proficiency & Aptitude)


# 1. Recreate the exact participant order from the LEAP-Q competence scores
comp_cols = [col for col in leap.columns if col.endswith('competence_speak')]
leap_filtered = leap[leap['participant_id'].isin(df['participant_id'])].copy()

# Diagnostics
print("--- Data Alignment Diagnostics ---")
print(f"Participants in your main 'df': {df['participant_id'].nunique()}")
print(f"Participants in 'leap' file:    {leap['participant_id'].nunique()}")
print(f"Overlapping participants found: {leap_filtered['participant_id'].nunique()}")

if len(leap_filtered) > 0 and len(comp_cols) > 0:
    leap_filtered['total_competence'] = leap_filtered[comp_cols].sum(axis=1)
    participant_order = leap_filtered.sort_values('total_competence', ascending=False)['participant_id'].tolist()
else:
    participant_order = df['participant_id'].unique().tolist()

# 2. Prepare Aptitude Data & Force Positive (Min-Max Scaling to 0-10)
apt_cols = ['artgram_corr_z', 'hindi_score_weighted_z', 'mlat5_corr_z', 'farsi']
apt_data = df.set_index('participant_id').reindex(participant_order).fillna(0)

apt_data_positive = pd.DataFrame(index=apt_data.index)
for col in apt_cols:
    if col in apt_data.columns:
        col_min = apt_data[col].min()
        col_max = apt_data[col].max()
        if col_max == col_min:
            apt_data_positive[col] = 0.0
        else:
            apt_data_positive[col] = ((apt_data[col] - col_min) / (col_max - col_min)) * 10
    else:
        apt_data_positive[col] = 0.0

# 3. Prepare Language Data (Long format)
if len(comp_cols) > 0:
    leap_long = leap_filtered.melt(
        id_vars=['participant_id'],
        value_vars=comp_cols,
        var_name='language',
        value_name='competence'
    ).dropna(subset=['competence'])
else:
    leap_long = pd.DataFrame(columns=['participant_id', 'language', 'competence'])

# --------------------------------------------------------------
# 4. BUILD THE OVERLAID SINGLE PLOT (TWIN-AXIS)
# --------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(15, 6), facecolor='white')
ax2 = ax1.twinx()

# --- LAYER 1: Language Competence (White Stacked Bars with Black Edges on Left Axis) ---
bottom_tracker_lang = pd.Series(0.0, index=participant_order)

if not leap_long.empty:
    for lang_name in sorted(leap_long['language'].unique()):
        lang_data = leap_long[leap_long['language'] == lang_name]
        lang_series = (
            lang_data.groupby('participant_id')['competence']
            .first()
            .reindex(participant_order)
            .fillna(0)
        )
        
        ax1.bar(
            range(len(participant_order)),
            lang_series.values,
            bottom=bottom_tracker_lang.values,
            color='white',
            edgecolor='black',
            linewidth=0.35,
            width=1.0,
            zorder=1
        )
        bottom_tracker_lang += lang_series

# --- LAYER 2: Aptitude Components (Overlaid Blue Bars on Right Axis) ---
bottom_tracker_apt = pd.Series(0.0, index=participant_order)

for col in apt_cols:
    apt_series = apt_data_positive[col]
    ax2.bar(
        range(len(participant_order)),
        apt_series.values,
        bottom=bottom_tracker_apt.values,
        color='#a0c4df',        # Muted light blue matching reference figure
        alpha=0.55,             # Semi-transparent overlay
        edgecolor='none',
        width=1.0,
        zorder=2
    )
    bottom_tracker_apt += apt_series

# --------------------------------------------------------------
# 5. AXIS + STYLING (MATCHING REFERENCE FIGURE)
# --------------------------------------------------------------
# X-Axis Styling
ax1.set_xticks(np.arange(len(participant_order))[::10])
ax1.set_xticklabels(np.arange(len(participant_order))[::10])
ax1.set_xlabel("Participants (Ordered by Total Speaking Competence)", fontsize=11, labelpad=10)

# Left Y-Axis (Language Proficiency)
ax1.set_ylabel("Speaking Proficiency (White Bars)", fontsize=11, color='black', labelpad=10)
ax1.set_ylim(bottom=0)

# Right Y-Axis (Aptitude Components)
blue_color = '#2b5c8f'
ax2.set_ylabel("Rescaled Aptitude Components (0-10 Scale, Blue Bars)", fontsize=11, color=blue_color, labelpad=10)
ax2.tick_params(axis='y', labelcolor=blue_color)
ax2.set_ylim(bottom=0, top=42)  # Scaled to fit max cumulative aptitude (~40)

# Limits & Horizontal Grid
N = len(participant_order)
margin = 0.5
ax1.set_xlim(-margin, N - 1 + margin)

ax1.grid(axis='y', color='#e0e0e0', linestyle='-', linewidth=0.7, alpha=0.6)
ax1.set_axisbelow(True)

# Spine Cleanup (Hide outer box borders)
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.title("Speaking Proficiency vs. Cumulative Aptitude Components (Normalized)", pad=20, fontsize=13)
plt.tight_layout()

# Save the figure
filename_super = f"lang_aptitude_overlay_{timestamp}.png"
save_path_super = os.path.join(OUTPUT_DIR, filename_super)
plt.savefig(save_path_super, format='png', dpi=300, bbox_inches='tight')

plt.show()

