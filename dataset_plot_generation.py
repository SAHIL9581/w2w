#@title 5. Upload Full LAS Dataset (ZIP file)
import os
import pandas as pd
import lasio
from joblib import load
import json
import torch
import argparse
import sys
import os
import glob
import zipfile
import json

import pandas as pd
import matplotlib.pyplot as plt


try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

def plot_well_pair(df, w1, w2, out_dir):
    df1 = df[df['WELL'] == w1]
    df2 = df[df['WELL'] == w2]
    if df1.empty or df2.empty:
        print(f"⚠️  Skipping plot for '{w1}' vs '{w2}' (no data)")
        return

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 6), dpi=150)
    ax.plot(df1['GR'], df1['DEPTH_MD'], label=w1)
    ax.plot(df2['GR'], df2['DEPTH_MD'], label=w2)
    ax.invert_yaxis()
    ax.set_xlabel('Gamma Ray')
    ax.set_ylabel('Depth (MD)')
    ax.set_title(f"{w1} vs {w2}")
    ax.legend()

    fname = f"{w1.replace('/', '_')}_vs_{w2.replace('/', '_')}.png"
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved plot: {out_path}")

    if HAS_MLFLOW:
        mlflow.log_artifact(out_path)


def main():
    # — your existing argparse + config loading + LAS→CSV logic here —
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config', default='config.json')
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"[ERROR] Config '{args.config}' not found.", file=sys.stderr)
        sys.exit(1)

    config = json.load(open(args.config))
    zip_folder    = config['paths']['zip_folder']
    raw_las_dir   = config['paths'].get('raw_las_dir','raw_las')
    processed_csv = config['paths']['processed_csv_path']

    # … unpack and process LAS exactly as before …
    # at the end you write out processed_csv

    df = pd.read_csv(processed_csv, sep=';')

    # — now dynamically build your well‑pairs —
    wells = sorted(df['WELL'].unique())
    if len(wells) < 2:
        print("⚠️  Not enough wells to plot pairs.")
        return

    # e.g. sliding window pairs:
    well_pairs = [(wells[i], wells[i+1]) for i in range(len(wells)-1)]

    plots_dir = config.get('inference', {}).get('plots_dir', 'plots')
    print("\n--- Generating inference dashboard plots ---")
    for w1, w2 in well_pairs:
        plot_well_pair(df, w1, w2, plots_dir)
    print("--- Done 🚀 ---\n")

    # finally list all wells
    print("\n--- Available Well Names for Correlation ---")
    for w in wells:
        print(f"- {w}")
    print("------------------------------------------")


if __name__ == '__main__':
    main()

# #@title 8. 🚀 Inference Dashboard: Generate and Log Plots



# # --- ACTION REQUIRED: Define the well pairs you want to plot ---
# # Copy and paste valid well names from the output of Cell 5.
# well_pairs_to_plot = [
#     ("15_9-13 Sleipner East Appr", "16/1-2  Ivar Aasen Appr"),
#     ("16/2-6 Johan Sverdrup", "16/5-3 Johan Sverdrup Appr"),
#     ("35/11-1", "35/11-6"),
#     # Add more pairs here...
# ]
# # -----------------------------------------------------------------



# # 2. Load the trained model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = W2WTransformerModel(config).to(device)
# model.load_state_dict(torch.load(config['paths']['final_model_path'], map_location=device))
# model.eval()
# print(f"✅ Model '{config['paths']['final_model_path']}' loaded successfully onto {device}.")

# # 3. Load the full dataset for lookups
# full_data = pd.read_csv(config['paths']['processed_csv_path'], delimiter=';')
# print(f"✅ Full dataset with {len(full_data.WELL.unique())} wells loaded.")

# # 4. Generate plots and save locally
# for i, (well1, well2) in enumerate(well_pairs_to_plot):
#     print(f"\n--- Generating plot for: {well1} vs {well2} ---")
#     # Create a filesystem-safe filename
#     safe_well1 = well1.replace('/','-').replace(' ','_')
#     safe_well2 = well2.replace('/','-').replace(' ','_')
#     output_filename = f"correlation_{safe_well1}_vs_{safe_well2}.png"

#     # NOTE: This still uses the MOCK inference logic.
#     # To use the real model, you would pass data patches through `model(patches)`
#     # and interpret the output to create the similarity matrix.
#     success = generate_single_correlation_plot(config, full_data, well1, well2, output_filename)

#     if success:
#         print(f"Plot saved locally: {output_filename}")
#     else:
#         print(f"Failed to generate plot for: {well1} vs {well2}")