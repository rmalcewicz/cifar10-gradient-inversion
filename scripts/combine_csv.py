import pandas as pd
import glob
import os

def combine_csv_files(base_dir, output_filename="combined_results.csv"):
    all_dataframes = []
    
    # Create a glob pattern to find all CSV files in immediate subdirectories of base_dir
    pattern = os.path.join(base_dir, "*.csv")
    file_paths = glob.glob(pattern)

    if not file_paths:
        print(f"No CSV files found in the immediate subdirectories of: {base_dir}")
        return

    print(f"Found {len(file_paths)} files to process.")
    
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            all_dataframes.append(df)
            print(f"Read file: {file_path}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_df.to_csv(f"{base_dir}/{output_filename}", index=False)
        print(f"\nSuccessfully combined {len(all_dataframes)} files into {output_filename}")
        print(f"Total rows in combined file: {len(combined_df)}")

        for file_path in file_paths:
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    else:
        print("No dataframes to combine.")

if __name__ == "__main__":
    # The base directory where your experiment subdirectories are located
    experiment_base_path = "saved_experiments/exp1"
    
    combine_csv_files(experiment_base_path)