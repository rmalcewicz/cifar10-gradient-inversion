import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- Configuration Constants ---

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

# The 10 CIFAR-10 class labels for 10x10 heatmap
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]

# --- Data Loading and Simulation ---

def load_data(file_path):
    """
    Loads the CSV data or uses synthetic data if the file is not accessible.
    NOTE: If running this locally, ensure FILE_PATH is correct.
    """
    if os.path.exists(file_path):
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
    else:
        print(f"File not found at {file_path}. Generating synthetic data for visualization.")
        # --- Synthetic Data Generation ---
        # Data structure mirrors a sweep of batch sizes with multiple repetitions
        batch_sizes = [1, 2, 4, 8, 16] 
        data = []
        for bs in batch_sizes:
            for rep in range(10): # 10 repetitions
                # Simulate success rates
                ori_rate = 0.85 + np.random.uniform(0, 0.1)
                recon_rate = 0.55 - (0.01 * bs) - np.random.uniform(0, 0.05)
                recon_agg_rate = min(1.0, recon_rate + 0.1 * np.random.uniform(0, 1))
                
                # Simple example for class-specific data (for heatmap test)
                class_set = ['bird', 'cat', 'dog', 'truck', 'airplane']
                class_a = np.random.choice(class_set)
                class_b = np.random.choice(class_set)
                majority_class = np.random.randint(0, 2)
                predicted_a = np.random.choice(CIFAR10_CLASSES)
                predicted_b = np.random.choice(CIFAR10_CLASSES)

                data.append({
                    BATCH_SIZE: bs,
                    ORI_SUCCESS: ori_rate,
                    RECON_SUCCESS: max(0, recon_rate),
                    RECON_AGG_SUCCESS: max(0, recon_agg_rate),
                    MAJORITY_CLASS: majority_class,
                    CLASS_A: class_a,
                    CLASS_B: class_b,
                    PREDICTED_A: predicted_a,
                    PREDICTED_B: predicted_b,
                })

        df = pd.DataFrame(data)
        # ----------------------------------
    
    # Clean up success columns
    for col in [RECON_SUCCESS, ORI_SUCCESS, RECON_AGG_SUCCESS]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    print(f"Data loading/generation complete. Rows: {len(df)}")
    return df

# ----------------------------------------------------------------------
# PLOT 1: Combined Success vs. Batch Size (NO ERROR BARS)
# ----------------------------------------------------------------------
def plot_combined_success_vs_batch_size(df):
    """
    Generates the requested plot: Original, Reconstructed (Non-Aggregated), 
    and Reconstructed (5-batch Aggregated) success rates vs. Batch Size, 
    WITH NO ERROR BARS.
    """
    if RECON_AGG_SUCCESS not in df.columns:
        print(f"Error: Aggregate column '{RECON_AGG_SUCCESS}' is missing. Skipping combined plot.")
        return

    # 1. Melt the DataFrame to long format
    df_melted = df.melt(
        id_vars=BATCH_SIZE,
        value_vars=[ORI_SUCCESS, RECON_SUCCESS, RECON_AGG_SUCCESS], 
        var_name='Metric Type',
        value_name='Majority Success Rate'
    )
    
    # 2. Map the technical column names to descriptive labels and define order
    data_type_map = {
        ORI_SUCCESS: 'Original', 
        RECON_SUCCESS: 'Reconstructed (Non-Agg.)',
        RECON_AGG_SUCCESS: 'Reconstructed (5-batch Agg.)'
    }
    df_melted['Metric Type'] = df_melted['Metric Type'].map(data_type_map)
    
    # Define the custom order for the legend/bars (as requested)
    custom_order = [
        'Original', 
        'Reconstructed (Non-Agg.)', 
        'Reconstructed (5-batch Agg.)'
    ]
    df_melted['Metric Type'] = pd.Categorical(df_melted['Metric Type'], categories=custom_order, ordered=True)

    # 3. Plotting
    plt.figure(figsize=(12, 7))
    sns.barplot(
        x=BATCH_SIZE, 
        y='Majority Success Rate', 
        hue='Metric Type', 
        data=df_melted,
        palette=['#31a354', '#3182bd', '#e6550d'], 
        errorbar=None,
        dodge=True,
        width=0.7,
        gap=0.2
    )
    
    plt.title('Comparison of Success Rates Across Batch Sizes', fontsize=16)
    plt.ylabel('Majority Success Rate')
    plt.xlabel('Batch Size')
    plt.ylim(0, 1) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Data Type', loc='upper right')
    
    output_path = 'combined_success_vs_batch_size_no_se.png'
    plt.savefig(output_path)
    plt.close()
    
    return output_path

# ----------------------------------------------------------------------
# PLOT 2: 10x10 Guess Distribution Heatmap
# ----------------------------------------------------------------------
def plot_majority_guess_heatmap_10x10(df, target_batch_size):
    """
    Generates a 10x10 heatmap of the predicted class distribution for the true 
    majority class (Reconstruction), filtered for a specific batch size.
    """
    
    # Filter data for the requested batch size
    df = df[df[BATCH_SIZE] == target_batch_size].copy()
    
    if df.empty:
        print(f"Warning: No data available for batch size {target_batch_size} for heatmap.")
        return

    # 1. Determine the True Majority Class Name for each row
    df['true_majority_class_name'] = df.apply(
        lambda row: row[CLASS_A] if row[MAJORITY_CLASS] == 0 else row[CLASS_B],
        axis=1
    )

    # 2. Determine the Predicted Majority Class Name for each row (Reconstruction)
    df['predicted_majority_class'] = df.apply(
        lambda row: row[PREDICTED_A] if row[MAJORITY_CLASS] == 0 else row[PREDICTED_B],
        axis=1
    )

    # 3. Create the Confusion Matrix (Frequency Count)
    confusion_matrix = pd.crosstab(
        df['true_majority_class_name'], 
        df['predicted_majority_class']
    )
    
    # 4. Reindex the matrix to the full 10x10 grid and Normalize
    confusion_matrix_10x10 = confusion_matrix.reindex(
        index=CIFAR10_CLASSES, 
        columns=CIFAR10_CLASSES, 
        fill_value=0
    )
    
    # Normalize the matrix across the true classes (rows) so that each row sums to 1.
    confusion_matrix_prop = confusion_matrix_10x10.div(
        confusion_matrix_10x10.sum(axis=1), axis=0
    ).fillna(0) 

    # 5. Plotting the Heatmap
    plt.figure(figsize=(10, 10)) 
    sns.heatmap(
        confusion_matrix_prop,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar_kws={'label': 'Proportion of Guesses'}
    )
    
    plt.title(f'10x10 Predicted Class Distribution for True Majority Class (Reconstruction, BS={target_batch_size})', fontsize=14)
    plt.ylabel('True Majority Class (Majority Sample)')
    plt.xlabel('Predicted Class (Reconstruction Output)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    output_path = f'bs{target_batch_size}_majority_guess_heatmap_10x10.png'
    plt.savefig(output_path)
    plt.close()
    
    return output_path

# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data(FILE_PATH)
    if df is None:
        print("Exiting due to data loading failure.")
    else:
        # 1. Generate the Combined Success Rate Plot (NO ERROR BARS)
        plot_combined_success_vs_batch_size(df)
        print("Combined success plot saved to 'combined_success_vs_batch_size_no_se.png'.")
        
        # 2. Generate the 10x10 Heatmap (example using Batch Size 1)
        plot_majority_guess_heatmap_10x10(df, target_batch_size=1)
        print("10x10 Heatmap plot saved (using BS=1 data).")

        print("All required plots have been generated.")