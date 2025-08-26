import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json

# --- File Paths and Directories ---
csv_path = "/zfsauton2/home/rmalcewi/summer2025/cifar10-gradient-inversion/saved_experiments/combined_results.csv"
output_dir = "data_output_agg"
plots_dir = os.path.join(output_dir, "plots")
json_dir = os.path.join(output_dir, "json")

# Create directories if they don't exist
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(json_dir, exist_ok=True)
print(f"Output directories created at: {plots_dir} and {json_dir}")

# --- Plotting and Data Processing Functions ---
random_baseline = {0: 0.81, 1: 0.18, 2: 0.01}


def get_plot_data(data, group_by_col, suffix=""):
    """
    Processes data for a given grouping column and returns counts and percentages.

    Args:
        data (pd.DataFrame): The input DataFrame.
        group_by_col (str): The column to group by.
        suffix (str): The suffix for the success columns (e.g., '_agg').

    Returns:
        dict: A dictionary containing counts and percentages.
    """
    success_values = [0, 1, 2]
    recon_col = f"recon_successes{suffix}"

    recon_counts = data.groupby(group_by_col)[recon_col].value_counts().unstack(fill_value=0)

    for val in success_values:
        if val not in recon_counts.columns:
            recon_counts[val] = 0

    recon_counts = recon_counts.reindex(columns=success_values, fill_value=0)

    recon_percentages = recon_counts.div(recon_counts.sum(axis=1), axis=0) * 100

    return {
        "recon_counts": recon_counts.to_dict("index"),
        "recon_percentages": recon_percentages.to_dict("index"),
    }


def plot_agg_comparison_with_baseline(data, group_by_col, title, baseline):
    """
    Generates and saves a comparison plot of aggregated vs. non-aggregated results for recon successes.

    Args:
        data (pd.DataFrame): The input DataFrame.
        group_by_col (str): The column to group by (e.g., 'batch_size').
        title (str): The title of the plot.
        baseline (dict): The random baseline values.
    """
    success_values = [0, 1, 2]

    # Calculate percentages for both original and aggregated data
    recon_percentages = (
        data.groupby(group_by_col)["recon_successes"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        * 100
    )
    recon_agg_percentages = (
        data.groupby(group_by_col)["recon_successes_agg"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        * 100
    )

    # Ensure all success values are present as columns
    for df in [recon_percentages, recon_agg_percentages]:
        for val in success_values:
            if val not in df.columns:
                df[val] = 0
        df.reindex(columns=success_values, fill_value=0)

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.suptitle(title, fontsize=16)

    groups = recon_percentages.index.tolist()
    x = np.arange(len(groups))

    # Define color pairs for non-aggregated and aggregated bars
    # Using a darker and lighter shade for each success value
    colors = {
        0: ("#1f77b4", "#aec7e8"),  # Dark Blue vs. Light Blue for Success 0
        1: ("#ff7f0e", "#ffbb78"),  # Dark Orange vs. Light Orange for Success 1
        2: ("#2ca02c", "#98df8a"),  # Dark Green vs. Light Green for Success 2
    }

    # Define bar width and offsets
    bar_width = 0.1  # Made bars thinner
    offsets = [-0.15, 0, 0.15]  # Adjusted offsets to prevent overlap

    for i, val in enumerate(success_values):
        # Calculate x positions for each pair of bars
        non_agg_x = x + offsets[i] - bar_width / 2
        agg_x = x + offsets[i] + bar_width / 2

        # Plot non-aggregated bars with darker colors
        ax.bar(
            non_agg_x,
            recon_percentages[val],
            bar_width,
            color=colors[val][0],
            label=f"Non-Aggregated - Success {val}",
        )

        # Plot aggregated bars with lighter colors
        ax.bar(
            agg_x,
            recon_agg_percentages[val],
            bar_width,
            color=colors[val][1],
            label=f"Aggregated - Success {val}",
        )

    # Add baselines
    for i, val in enumerate(success_values):
        ax.axhline(
            y=baseline[val] * 100,
            color=colors[val][0],
            linestyle="--",
            label=f"Baseline Success {val}",
            zorder=0,
        )

    # Configure axes and legend
    ax.set_title("Reconstruction Success Comparison")
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel(group_by_col.replace("_", " ").title())
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend(title="Legend", loc="best")

    plt.tight_layout(rect=[0, 0, 0, 0.96])
    plot_filename = f'{title.replace(" ", "_").lower()}.png'
    plt.savefig(os.path.join(plots_dir, plot_filename))
    plt.close(fig)


# --- Main Script Execution ---

# Load the data
df = pd.read_csv(csv_path)

# Create a combined class pair column (needed for some of the get_plot_data calls)
df["class_pair"] = df["class_a"] + " - " + df["class_b"]

# Generate and save the single comparison plot
print("Generating and saving single comparison plot for aggregated vs. non-aggregated data...")
plot_agg_comparison_with_baseline(
    data=df,
    group_by_col="batch_size",
    title="Comparison of Aggregated vs. Non-Aggregated Successes per Batch Size",
    baseline=random_baseline,
)

print("Plot generated and saved successfully.")

# Prepare and save JSON data (keeping this for completeness, but focusing on recon data)
print("Preparing and saving JSON data...")
json_data = {}
json_data["per_class_pair"] = get_plot_data(df, "class_pair")
json_data["per_individual_class"] = get_plot_data(
    df.melt(
        id_vars=["batch_size", "batch_idx", "repetition", "recon_successes"],
        value_vars=["class_a", "class_b"],
        var_name="class_type",
        value_name="individual_class",
    ),
    "individual_class",
)
json_data["per_batch_size"] = get_plot_data(df, "batch_size")
json_data["per_batch_index"] = get_plot_data(df, "batch_idx")

# Prepare and save aggregated JSON data
json_data["per_class_pair_agg"] = get_plot_data(df, "class_pair", suffix="_agg")
json_data["per_individual_class_agg"] = get_plot_data(
    df.melt(
        id_vars=["batch_size", "batch_idx", "repetition", "recon_successes_agg"],
        value_vars=["class_a", "class_b"],
        var_name="class_type",
        value_name="individual_class",
    ),
    "individual_class",
    suffix="_agg",
)
json_data["per_batch_size_agg"] = get_plot_data(df, "batch_size", suffix="_agg")
json_data["per_batch_index_agg"] = get_plot_data(df, "batch_idx", suffix="_agg")

with open(os.path.join(json_dir, "full_plot_data.json"), "w") as f:
    json.dump(json_data, f, indent=4)
print(f"JSON data saved to {os.path.join(json_dir, 'full_plot_data.json')}")
