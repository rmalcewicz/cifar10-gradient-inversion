import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
# NOTE: Update this path to your actual file location if running locally.
FILE_PATH = '/zfsauton2/home/rmalcewi/summer2025/cifar10-gradient-inversion/saved_experiments_pres/combined_results.csv'

# Column names based on the data provided
RECON_SUCCESS = 'recon_majority_success'
ORI_SUCCESS = 'ori_majority_success'
RECON_AGG_SUCCESS = 'recon_majority_success_agg'
BATCH_SIZE = 'batch_size'
MAJORITY_CLASS = 'recon_majority_class'
CLASS_A = 'class_a'
CLASS_B = 'class_b'
PREDICTED_A = 'recon_predicted_class_a'
PREDICTED_B = 'recon_predicted_class_b'

# --- Function Definitions ---

def load_data(file_path):
    """Loads the CSV data and performs initial checks."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}. Please ensure the file exists at the specified path.")
        return None
    
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Ensure success columns are numeric
    for col in [RECON_SUCCESS, ORI_SUCCESS, RECON_AGG_SUCCESS]:
        if col in df.columns:
            # Coerce errors to NaN, then fill NaN with 0 for success metrics
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    print(f"Successfully loaded {len(df)} rows.")
    
    # --- Diagnostic Addition: Print unique batch sizes found ---
    if BATCH_SIZE in df.columns:
        unique_batches = df[BATCH_SIZE].unique()
        print(f"Unique batch sizes found in data: {unique_batches}")
    # -----------------------------------------------------------
    
    return df

# ----------------------------------------------------------------------
# PLOT 1: Majority Success vs. Batch Size (with Standard Error)
# ----------------------------------------------------------------------
def plot_majority_success_vs_batch_size(df):
    """
    1. Bar plot: recon_majority_success vs batch_size (left) 
        and ori_majority_success vs batch_size (right).
    """
    
    df_melted = df.melt(
        id_vars=BATCH_SIZE, 
        value_vars=[RECON_SUCCESS, ORI_SUCCESS], 
        var_name='Data Type', 
        value_name='Average Majority Success Rate'
    )
    
    data_type_map = {
        RECON_SUCCESS: 'Reconstructed', 
        ORI_SUCCESS: 'Original'
    }
    df_melted['Data Type'] = df_melted['Data Type'].map(data_type_map)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=BATCH_SIZE, 
        y='Average Majority Success Rate', 
        hue='Data Type', 
        data=df_melted,
        palette=['#1e3a8a', '#60a5fa'],
    )
    
    plt.title('Majority Success per Batch Size with Standard Error', fontsize=14)
    plt.ylabel('Average Success Rate')
    plt.xlabel('Batch Size')
    plt.ylim(0, 1) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='')
    
    output_path = 'success_vs_batch_size_with_se.png'
    plt.savefig(output_path)
    plt.close()

# ----------------------------------------------------------------------
# PLOT 2: Aggregated vs. Non-Aggregated Success (with Standard Error)
# ----------------------------------------------------------------------
def plot_agg_vs_batch_success(df):
    """
    2. Bar plot: majority_success_agg_recon compared with majority_success_recon 
        for each batch size.
    """
    if RECON_AGG_SUCCESS not in df.columns:
        print(f"Skipping plot 2: Aggregate column '{RECON_AGG_SUCCESS}' not found in data.")
        return
        
    df_melted = df.melt(
        id_vars=BATCH_SIZE, 
        value_vars=[RECON_SUCCESS, RECON_AGG_SUCCESS], 
        var_name='Metric Type', 
        value_name='Average Success Rate'
    )

    metric_map = {
        RECON_SUCCESS: 'Non-Aggregated', 
        RECON_AGG_SUCCESS: '5-batch Aggregated'
    }
    df_melted['Metric Type'] = df_melted['Metric Type'].map(metric_map)
    
    title = 'Aggregated vs Non-Aggregated Majority Success with Standard Error'

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=BATCH_SIZE, 
        y='Average Success Rate', 
        hue='Metric Type', 
        data=df_melted,
        palette=['#facc15', '#ef4444'],
    )
    
    plt.title(title, fontsize=14)
    plt.ylabel('Average Success Rate')
    plt.xlabel('Batch Size')
    plt.ylim(0, 1) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='')
    
    output_path = 'agg_vs_batch_success_with_se.png'
    plt.savefig(output_path)
    plt.close()

# ----------------------------------------------------------------------
# PLOT 3: Distribution of Guesses for Majority Class
# ----------------------------------------------------------------------
def plot_majority_guess_distribution(df):
    """
    3. Bar plot showing the distribution of predicted classes for the true majority class.
    """
    
    # 1. Determine the True Majority Class Name for each row
    df['true_majority_class_name'] = df.apply(
        lambda row: row[CLASS_A] if row[MAJORITY_CLASS] == 0 else row[CLASS_B],
        axis=1
    )

    # 2. Determine the Predicted Majority Class Name for each row (Reconstruction)
    # If majority_class is 0 (Class A), use the prediction for Class A.
    # If majority_class is 1 (Class B), use the prediction for Class B.
    df['predicted_majority_class'] = df.apply(
        lambda row: row[PREDICTED_A] if row[MAJORITY_CLASS] == 0 else row[PREDICTED_B],
        axis=1
    )

    # 3. Calculate the distribution of predictions for each true majority class
    prediction_counts = df.groupby(['true_majority_class_name', 'predicted_majority_class']).size().reset_index(name='count')
    total_counts = prediction_counts.groupby('true_majority_class_name')['count'].sum().reset_index(name='total_count')

    # Merge to calculate the proportion
    df_distribution = pd.merge(prediction_counts, total_counts, on='true_majority_class_name')
    df_distribution['Proportion'] = df_distribution['count'] / df_distribution['total_count']
    
    # --- PLOTTING ---
    plt.figure(figsize=(12, 8))
    
    # Use a FacetGrid (catplot) to create separate plots for each true majority class
    g = sns.catplot(
        x='predicted_majority_class', 
        y='Proportion', 
        col='true_majority_class_name', 
        data=df_distribution, 
        kind='bar',
        palette='viridis',
        height=5, 
        aspect=1.2,
        sharex=False, # Allow different X-axis categories/ordering for each facet
        col_wrap=2 # Wrap facets to a maximum of 2 columns
    )

    # Labeling and Formatting
    g.set_titles("True Majority Class: {col_name}")
    g.set_axis_labels("Predicted Class", "Proportion of Guesses")
    g.fig.suptitle('Distribution of Predicted Classes for the True Majority Class (Reconstructed)', y=1.02, fontsize=16)

    # Rotate x-axis labels for better readability
    for ax in g.axes.flat:
        if ax.get_xticklabels():
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    output_path = 'majority_guess_distribution.png'
    g.savefig(output_path)
    # Ensure all figures opened by catplot are closed
    plt.close('all') 

# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------
def main():
    df = load_data(FILE_PATH)
    if df is None:
        return

    plot_majority_success_vs_batch_size(df)
    plot_agg_vs_batch_success(df)
    target_batch_size = 16
    df_filtered = df[df['batch_size'] == target_batch_size].copy()

    # Call the plotting function on the filtered data
    plot_majority_guess_distribution(df_filtered)
    #plot_majority_guess_distribution(df) # <-- New plot added
    print("Plotting complete. Three images saved: 'success_vs_batch_size_with_se.png', 'agg_vs_batch_success_with_se.png', and 'majority_guess_distribution.png'.")

if __name__ == "__main__":
    main()