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


def main():
    # 1) Parse command‑line args
    parser = argparse.ArgumentParser(
        description="Unpack a ZIP of LAS files and build a master CSV based on a JSON config"
    )
    parser.add_argument(
        "-c", "--config",
        default="config.json",
        help="Path to your JSON config file (default: config.json)"
    )
    args = parser.parse_args()
    config_path = args.config

    # 2) Load config
    if not os.path.isfile(config_path):
        print(f"[ERROR] Config file not found: '{config_path}'", file=sys.stderr)
        print("→ Please create it or point to it with --config /path/to/config.json", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)

    zip_folder       = config['paths']['zip_folder']
    raw_las_dir      = config['paths'].get('raw_las_dir', 'raw_las')
    processed_csv    = config['paths']['processed_csv_path']

    # 3) Find the ZIP
    if not os.path.isdir(zip_folder):
        raise FileNotFoundError(f"Folder not found: {zip_folder}")
    zip_files = glob.glob(os.path.join(zip_folder, "*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No .zip files found in {zip_folder}")
    zip_path = zip_files[0]
    print(f"✅ Using ZIP: {zip_path}")

    # 4) Unpack
    os.makedirs(raw_las_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(raw_las_dir)

    # 5) Collect all .las paths
    las_paths = [
        os.path.join(root, fn)
        for root, _, files_ in os.walk(raw_las_dir)
        for fn in files_ if fn.lower().endswith(".las")
    ]
    if not las_paths:
        print(f"[WARNING] No LAS files found under '{raw_las_dir}'")

    # 6) Read & tag
    all_wells = []
    for fp in las_paths:
        try:
            las = lasio.read(fp)
            df = las.df().reset_index()
            df['WELL']  = las.well.WELL.value or os.path.splitext(os.path.basename(fp))[0]
            df['GROUP'] = next(
                (p.value for p in las.params if 'GROUP' in p.mnemonic.upper()),
                'UNKNOWN'
            )
            all_wells.append(df)
        except Exception as e:
            print(f"‑ Could not read {fp}: {e}", file=sys.stderr)

    # 7) Concat & save
    if all_wells:
        master_df = pd.concat(all_wells, ignore_index=True)
        if 'DEPT' in master_df.columns:
            master_df.rename(columns={'DEPT': 'DEPTH_MD'}, inplace=True)
        os.makedirs(os.path.dirname(processed_csv), exist_ok=True)
        master_df.to_csv(processed_csv, index=False, sep=';')
        print(f"\n✅ Processed {len(las_paths)} LAS files → '{processed_csv}'")
    else:
        print("[ERROR] No dataframes to concatenate. Exiting.", file=sys.stderr)
        sys.exit(1)

    # 8) List wells
    print("\n--- Available Well Names for Correlation ---")
    for well in sorted(master_df['WELL'].unique()):
        print(f"- {well}")
    print("------------------------------------------")

if __name__ == "__main__":
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

# print("--- Initializing Inference Run ---")
# # 1. Dynamically get the number of input features from the saved scaler
# scaler = load(config['paths']['std_scaler_path'])
# config['finetuning']['model_params']['in_channels'] = scaler.n_features_in_
# print(f"Loaded scaler with {scaler.n_features_in_} features.")

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