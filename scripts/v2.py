import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json

# --- File Paths and Directories ---
csv_path = "/zfsauton2/home/rmalcewi/summer2025/cifar10-gradient-inversion/saved_experiments/clip_experiment_summary.csv"
output_dir = "data_output"
plots_dir = os.path.join(output_dir, "plots")
json_dir = os.path.join(output_dir, "json")

# Create directories if they don't exist
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(json_dir, exist_ok=True)
print(f"Output directories created at: {plots_dir} and {json_dir}")

# --- Plotting and Data Processing Functions ---
random_baseline = {0: 0.81, 1: 0.18, 2: 0.01}

def get_plot_data(data, group_by_col):
    """Processes data for a given grouping column and returns counts and percentages."""
    success_values = [0, 1, 2]
    recon_counts = data.groupby(group_by_col)['recon_successes'].value_counts().unstack(fill_value=0)
    ori_counts = data.groupby(group_by_col)['ori_successes'].value_counts().unstack(fill_value=0)

    for val in success_values:
        if val not in recon_counts.columns:
            recon_counts[val] = 0
        if val not in ori_counts.columns:
            ori_counts[val] = 0

    recon_counts = recon_counts.reindex(columns=success_values, fill_value=0)
    ori_counts = ori_counts.reindex(columns=success_values, fill_value=0)

    recon_percentages = recon_counts.div(recon_counts.sum(axis=1), axis=0) * 100
    ori_percentages = ori_counts.div(ori_counts.sum(axis=1), axis=0) * 100

    return {
        'recon_counts': recon_counts.to_dict('index'),
        'ori_counts': ori_counts.to_dict('index'),
        'recon_percentages': recon_percentages.to_dict('index'),
        'ori_percentages': ori_percentages.to_dict('index')
    }

def plot_success_counts_with_baseline(data, group_by_col, title_prefix, baseline):
    """Generates and saves count and percentage plots with the random baseline."""
    success_values = [0, 1, 2]
    recon_counts = data.groupby(group_by_col)['recon_successes'].value_counts().unstack(fill_value=0)
    ori_counts = data.groupby(group_by_col)['ori_successes'].value_counts().unstack(fill_value=0)

    for val in success_values:
        if val not in recon_counts.columns:
            recon_counts[val] = 0
        if val not in ori_counts.columns:
            ori_counts[val] = 0

    recon_counts = recon_counts.reindex(columns=success_values, fill_value=0)
    ori_counts = ori_counts.reindex(columns=success_values, fill_value=0)

    # Plot counts
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    fig.suptitle(f'{title_prefix} - Success Counts', fontsize=16)

    recon_counts.plot(kind='bar', ax=axes[0], rot=0)
    axes[0].set_title('Recon Successes')
    axes[0].set_ylabel('Count')
    axes[0].set_xlabel(group_by_col.replace('_', ' ').title())
    axes[0].legend(title='Success Value', labels=success_values)

    ori_counts.plot(kind='bar', ax=axes[1], rot=0)
    axes[1].set_title('Ori Successes')
    axes[1].set_xlabel(group_by_col.replace('_', ' ').title())
    axes[1].legend(title='Success Value', labels=success_values)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(plots_dir, f'{title_prefix.replace(" ", "_").lower()}_counts.png'))
    plt.close(fig)

    # Plot percentages with baseline
    recon_percentages = recon_counts.div(recon_counts.sum(axis=1), axis=0) * 100
    ori_percentages = ori_counts.div(ori_counts.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    fig.suptitle(f'{title_prefix} - Success Percentages with Random Baseline', fontsize=16)

    recon_percentages.plot(kind='bar', ax=axes[0], rot=0)
    axes[0].set_title('Recon Successes')
    axes[0].set_ylabel('Percentage')
    axes[0].set_xlabel(group_by_col.replace('_', ' ').title())

    ori_percentages.plot(kind='bar', ax=axes[1], rot=0)
    axes[1].set_title('Ori Successes')
    axes[1].set_xlabel(group_by_col.replace('_', ' ').title())

    colors = ['blue', 'orange', 'green']
    for i, val in enumerate(success_values):
        axes[0].axhline(y=baseline[val] * 100, color=colors[i], linestyle='--', label=f'Baseline Success {val}')
        axes[1].axhline(y=baseline[val] * 100, color=colors[i], linestyle='--', label=f'Baseline Success {val}')

    axes[0].legend(title='Success Value')
    axes[1].legend(title='Success Value')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(plots_dir, f'{title_prefix.replace(" ", "_").lower()}_percentages_with_baseline.png'))
    plt.close(fig)

# --- Main Script Execution ---

# Load the data
df = pd.read_csv(csv_path)

# Create a combined class pair column
df['class_pair'] = df['class_a'] + ' - ' + df['class_b']

# Generate and save all plots
print("Generating and saving plots...")
plot_success_counts_with_baseline(df, 'class_pair', 'Successes per Class Pair', random_baseline)
individual_classes = df.melt(id_vars=['batch_size', 'batch_idx', 'repetition', 'recon_successes', 'ori_successes'],
                           value_vars=['class_a', 'class_b'], var_name='class_type', value_name='individual_class')
plot_success_counts_with_baseline(individual_classes, 'individual_class', 'Successes per Individual Class', random_baseline)
plot_success_counts_with_baseline(df, 'batch_size', 'Successes per Batch Size', random_baseline)
plot_success_counts_with_baseline(df, 'batch_idx', 'Successes per Batch Index', random_baseline)
print("Plots generated and saved successfully.")

# Prepare and save JSON data
print("Preparing and saving JSON data...")
json_data = {}
json_data['per_class_pair'] = get_plot_data(df, 'class_pair')
json_data['per_individual_class'] = get_plot_data(individual_classes, 'individual_class')
json_data['per_batch_size'] = get_plot_data(df, 'batch_size')
json_data['per_batch_index'] = get_plot_data(df, 'batch_idx')

with open(os.path.join(json_dir, 'full_plot_data.json'), 'w') as f:
    json.dump(json_data, f, indent=4)
print(f"JSON data saved to {os.path.join(json_dir, 'full_plot_data.json')}")