#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script fits GLM interaction models (Multilingualism * Aptitude + Covariates) 
and visualizes the predicted interaction surfaces as heatmaps for significant 
fMRI clusters in interaction contrast (Contrast 6).

Author: IB
Last update: 2026.07.21
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf

#%% ==========================================
# 1. Configuration and Paths
# ==========================================

# Base repo path
repo_root = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else "."

# Subfolder paths
beh_dir = os.path.join(repo_root, "behavioural_data")
fsl_dir = os.path.join(repo_root, "fsl")

# Behavioral data files
lang_tsv_path = os.path.join(beh_dir, "participant_lang_profile.tsv")
aptitude_tsv_path = os.path.join(beh_dir, "participants_aptitude_individual_scores_imputed.tsv")

# Base directory for FSL cluster activity CSVs
activity_base_dir = fsl_dir

# Output directory for plots
plot_output_dir = os.path.join(repo_root, "output/plots")
os.makedirs(plot_output_dir, exist_ok=True)

#%% ==========================================
# 2. Load and Prepare Behavioral & Covariate Data
# ==========================================

print("Loading behavioral datasets and covariates...")

# Helper to ensure participant IDs match the "sub-XXX" format
def clean_id(pid):
    pid = str(pid).strip()
    return pid if pid.startswith("sub-") else f"sub-{pid}"

# Load primary behavioral variables
df_lang = pd.read_csv(lang_tsv_path, sep='\t')[['participant_id', 'entropy_competence_speak']]
df_apt = pd.read_csv(aptitude_tsv_path, sep='\t')[['participant_id', 'aptitude_total_z']]

df_lang['participant_id'] = df_lang['participant_id'].apply(clean_id)
df_apt['participant_id'] = df_apt['participant_id'].apply(clean_id)

# Merge main behavioral metrics
df_behav = pd.merge(df_lang, df_apt, on='participant_id', how='inner')
df_behav.rename(columns={'participant_id': 'Participant'}, inplace=True)

# Load covariates from language profile TSV
df_cov = pd.read_csv(lang_tsv_path, sep='\t')
df_cov = df_cov.rename(columns={'participant_id': 'Participant'})
df_cov['Participant'] = df_cov['Participant'].apply(clean_id)
cov_cols = ['Participant', 'Hand_index', 'Sex', 'age', 'edu']
df_cov = df_cov[cov_cols]

print(f"Loaded behavioral metrics and covariates for {len(df_behav)} subjects.")

#%% ==========================================
# 3. Locate Contrast 6 Activity Files
# ==========================================

search_pattern = os.path.join(activity_base_dir, "**", "*contrast_6_cluster_activity.csv")
contrast_6_files = glob.glob(search_pattern, recursive=True)

if not contrast_6_files:
    print(f"No Contrast 6 CSV files found under {activity_base_dir}. Please check your extraction pipeline.")
    exit()

#%% ==========================================
# 4. GLM Interaction Heatmaps
# ==========================================

print("\n--- Generating GLM Interaction Heatmaps ---")

for file_path in contrast_6_files:
    filename = os.path.basename(file_path)
    condition_name = filename.replace("_contrast_6_cluster_activity.csv", "")
    
    # Load and merge brain activity with behavioral data & covariates
    df_activity = pd.read_csv(file_path)
    df_merged = pd.merge(df_activity, df_behav, on='Participant', how='inner')
    df_merged = pd.merge(df_merged, df_cov, on='Participant', how='inner').dropna()
    
    if df_merged.empty:
        print(f"Skipping {condition_name}: No overlapping data after merge.")
        continue
    
    # Identify cluster activity columns
    cluster_cols = [col for col in df_merged.columns if col.startswith("Cluster_") and col.endswith("_Mean")]
    
    for cluster in cluster_cols:
        print(f"  -> Fitting GLM and plotting prediction map for {condition_name} [{cluster}]...")
        
        x = df_merged['entropy_competence_speak'].values
        y = df_merged['aptitude_total_z'].values

        # 1. Run GLM (Adjusted for Covariates)
        formula = f"{cluster} ~ entropy_competence_speak * aptitude_total_z + Hand_index + Sex + age + edu"
        model = smf.ols(formula, data=df_merged).fit()
        
        # 2. Generate Prediction Grid (Holding Covariates Constant at Mean/Mode)
        grid_x, grid_y = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
        
        hand_index_ref = df_merged['Hand_index'].mean() if np.issubdtype(df_merged['Hand_index'].dtype, np.number) else df_merged['Hand_index'].mode()[0]
        sex_ref = df_merged['Sex'].mode()[0]
        age_ref = df_merged['age'].mean()
        edu_ref = df_merged['edu'].mean() if np.issubdtype(df_merged['edu'].dtype, np.number) else df_merged['edu'].mode()[0]
        
        grid_flat = pd.DataFrame({
            'entropy_competence_speak': grid_x.ravel(),
            'aptitude_total_z': grid_y.ravel(),
            'Hand_index': hand_index_ref,
            'Sex': sex_ref,
            'age': age_ref,
            'edu': edu_ref
        })
        
        grid_z_flat = model.predict(grid_flat).values.copy()
        grid_z = grid_z_flat.reshape(grid_x.shape)
        
        # 3. Plot Heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle(f"GLM Interaction Map (Data Bounds Only)\n{condition_name} - {cluster}", fontsize=14)
        
        heatmap = ax.pcolormesh(grid_x, grid_y, grid_z, cmap='RdBu_r', shading='auto', alpha=0.85)
        
        # Formatting and Colorbar
        cbar = fig.colorbar(heatmap, ax=ax)
        cbar.set_label('Predicted Brain Activity (z-stat)', rotation=270, labelpad=15)
        
        ax.set_xlabel('Multilingualism (Entropy Speak)')
        ax.set_ylabel('Aptitude (Total Z-Score)')
        
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        
        # Save output
        map_save_path = os.path.join(plot_output_dir, f"{condition_name}_{cluster}_heatmap.png")
        plt.savefig(map_save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"     Saved heatmap: {map_save_path}")

print("\nHeatmap generation complete! Plots saved in the 'plots' folder.")