import os
import pandas as pd
import lasio
import json
import argparse
import sys
import matplotlib.pyplot as plt

def process_las_folder(source_dir, output_csv):
    """
    Reads all .las files from a source directory, processes them into a single
    DataFrame, and saves it as a CSV file.
    """
    if not os.path.isdir(source_dir):
        print(f"❌ Error: Source folder not found at '{source_dir}'", file=sys.stderr)
        return False

    print(f"--> Searching for .las files in '{source_dir}'...")
    las_files_found = [os.path.join(root, file) for root, _, files in os.walk(source_dir) for file in files if file.lower().endswith('.las')]
    
    if not las_files_found:
        print(f"❌ Error: No .las files found in '{source_dir}'", file=sys.stderr)
        return False

    print(f"--> Processing {len(las_files_found)} LAS files...")
    all_wells_df = []
    for filepath in las_files_found:
        try:
            las = lasio.read(filepath)
            df = las.df().reset_index()
            # Ensure WELL name is read as a string to prevent type errors later
            well_name = getattr(las.well, 'WELL', os.path.splitext(os.path.basename(filepath))[0])
            df['WELL'] = str(well_name)
            
            df['GROUP'] = 'UNKNOWN'
            for param in las.params:
                if 'GROUP' in param.mnemonic.upper():
                    df['GROUP'] = str(param.value)
            all_wells_df.append(df)
        except Exception as e:
            print(f"    - ⚠️  Could not read {filepath}: {e}")

    master_df = pd.concat(all_wells_df, ignore_index=True)
    if 'DEPT' in master_df.columns:
        master_df.rename(columns={'DEPT': 'DEPTH_MD'}, inplace=True)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    master_df.to_csv(output_csv, index=False, sep=';')
    print(f"✅ Successfully created master dataset at '{output_csv}'")
    return True

def plot_well_pair(df, w1, w2, out_dir):
    df1 = df[df['WELL'] == w1]
    df2 = df[df['WELL'] == w2]
    if df1.empty or df2.empty:
        print(f"⚠️  Skipping plot for '{w1}' vs '{w2}' (one or both wells not found)")
        return

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 6), dpi=150)
    
    if 'GR' in df1.columns and 'GR' in df2.columns:
        ax.plot(df1['GR'], df1['DEPTH_MD'], label=w1, color='blue')
        ax.plot(df2['GR'], df2['DEPTH_MD'], label=w2, color='red')
        ax.set_xlabel('Gamma Ray (GR)')
    else:
        ax.text(0.5, 0.5, 'GR curve not available\nfor one or both wells.', ha='center', va='center')

    ax.invert_yaxis()
    ax.set_ylabel('Depth (MD)')
    ax.set_title(f"Well Correlation")
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    fname = f"Plot_{w1.replace('/', '_')}_vs_{w2.replace('/', '_')}.png"
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved plot: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Process a folder of LAS files and generate correlation plots.")
    parser.add_argument('--config', default='config.json', help="Path to the JSON configuration file.")
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"❌ Error: Config file '{args.config}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(args.config) as f:
        config = json.load(f)

    # --- 1. Process Data ---
    # Read paths from the new config structure
    source_folder = config['paths']['zip_folder'] # This is now a folder, not a zip
    csv_path = config['paths']['processed_csv_path']
    
    if not os.path.exists(csv_path):
        print(f"Processed CSV not found at '{csv_path}'. Generating it from folder '{source_folder}'...")
        success = process_las_folder(source_folder, csv_path)
        if not success:
            sys.exit(1)
    else:
        print(f"Using existing processed CSV: '{csv_path}'")

    # --- 2. Generate Plots ---
    df = pd.read_csv(csv_path, sep=';', low_memory=False)

    # **THE FIX IS HERE:** Ensure the 'WELL' column is treated as a string before sorting.
    wells = sorted(df['WELL'].astype(str).unique())
    
    if len(wells) < 2:
        print("⚠️  Not enough unique wells found in the data to create pairs.")
        return

    well_pairs = [(wells[i], wells[i+1]) for i in range(len(wells)-1)]

    # Your new config doesn't specify a plot directory, so we'll create a default one.
    plots_dir = 'inference_plots'
    print(f"\n--- Generating {len(well_pairs)} well pair plots in folder '{plots_dir}' ---")
    for w1, w2 in well_pairs:
        plot_well_pair(df, w1, w2, plots_dir)
    print("--- Done 🚀 ---\n")

    # --- 3. List available wells ---
    print("\n--- Available Well Names for Correlation ---")
    for w in wells:
        print(f"- {w}")
    print("------------------------------------------")

if __name__ == '__main__':
    main()