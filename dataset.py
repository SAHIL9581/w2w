#@title 1. Setup Environment & Install Libraries

# --- 1. Install Required Libraries ---
print("--> Installing libraries for inference...")
print("✅ Installation complete.")

# --- 2. Create Temporary Workspace ---
import os
os.makedirs("/content/inference_data", exist_ok=True)
os.makedirs("/content/inference_artifacts", exist_ok=True)
os.chdir("/content/inference_data")
print(f"✅ Workspace created at {os.getcwd()}")

#@title 5. Upload Full LAS Dataset (ZIP file)
import os
import pandas as pd
import lasio
from joblib import load
import json

print(">>> ACTION REQUIRED: Please upload the same ZIP file with all .las files used for training.")
uploaded = files.upload()

if uploaded:
    zip_filename = list(uploaded.keys())[0]
    print(f"\n✅ '{zip_filename}' uploaded. Processing into a single CSV for lookup...")

    # Unzip and process the files
    os.makedirs("raw_las", exist_ok=True)
    !unzip -q -o "{zip_filename}" -d raw_las/

    all_wells_df, las_files_found = [], []
    for root, dirs, files in os.walk("raw_las"):
        for file in files:
            if file.lower().endswith('.las'):
                las_files_found.append(os.path.join(root, file))

    for filepath in las_files_found:
        try:
            las = lasio.read(filepath)
            df = las.df().reset_index()
            df['WELL'] = las.well.WELL.value or os.path.splitext(os.path.basename(filepath))[0]
            df['GROUP'] = 'UNKNOWN'
            for param in las.params:
                if 'GROUP' in param.mnemonic.upper(): df['GROUP'] = param.value
            all_wells_df.append(df)
        except Exception as e: print(f"    - Could not read {filepath}: {e}")

    master_df = pd.concat(all_wells_df, ignore_index=True)
    if 'DEPT' in master_df.columns: master_df.rename(columns={'DEPT': 'DEPTH_MD'}, inplace=True)
    master_df.to_csv(config['paths']['processed_csv_path'], index=False, sep=';')

    print(f"\n✅ Successfully processed {len(las_files_found)} files into '{config['paths']['processed_csv_path']}'.")

    # Print available wells for convenience
    print("\n--- Available Well Names for Correlation ---")
    for well in sorted(master_df['WELL'].unique()):
        print(f"- {well}")
    print("------------------------------------------")

    # Clean up
    os.remove(zip_filename)
    !rm -rf raw_las
else:
    print("\n⚠️ Upload cancelled.")

#@title 8. 🚀 Inference Dashboard: Generate and Log Plots

import torch
from joblib import load
import pandas as pd

# --- ACTION REQUIRED: Define the well pairs you want to plot ---
# Copy and paste valid well names from the output of Cell 5.
well_pairs_to_plot = [
    ("15_9-13 Sleipner East Appr", "16/1-2  Ivar Aasen Appr"),
    ("16/2-6 Johan Sverdrup", "16/5-3 Johan Sverdrup Appr"),
    ("35/11-1", "35/11-6"),
    # Add more pairs here...
]
# -----------------------------------------------------------------

print("--- Initializing Inference Run ---")
# 1. Dynamically get the number of input features from the saved scaler
scaler = load(config['paths']['std_scaler_path'])
config['finetuning']['model_params']['in_channels'] = scaler.n_features_in_
print(f"Loaded scaler with {scaler.n_features_in_} features.")

# 2. Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = W2WTransformerModel(config).to(device)
model.load_state_dict(torch.load(config['paths']['final_model_path'], map_location=device))
model.eval()
print(f"✅ Model '{config['paths']['final_model_path']}' loaded successfully onto {device}.")

# 3. Load the full dataset for lookups
full_data = pd.read_csv(config['paths']['processed_csv_path'], delimiter=';')
print(f"✅ Full dataset with {len(full_data.WELL.unique())} wells loaded.")

# 4. Generate plots and log to W&B
with wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, job_type='inference-dashboard') as run:
    print(f"\n--> W&B Run for multiple plots started. View at: {run.url}")
    plots_to_log = {}
    for i, (well1, well2) in enumerate(well_pairs_to_plot):
        print(f"\n--- Generating plot for: {well1} vs {well2} ---")
        # Create a filesystem-safe filename
        safe_well1 = well1.replace('/','-').replace(' ','_')
        safe_well2 = well2.replace('/','-').replace(' ','_')
        output_filename = f"correlation_{safe_well1}_vs_{safe_well2}.png"

        # NOTE: This still uses the MOCK inference logic.
        # To use the real model, you would pass data patches through `model(patches)`
        # and interpret the output to create the similarity matrix.
        success = generate_single_correlation_plot(config, full_data, well1, well2, output_filename)

        if success:
            plots_to_log[f"Plot_{i+1}_{well1}_vs_{well2}"] = wandb.Image(output_filename)

    if plots_to_log:
        print("\n--> Logging all plots to Weights & Biases...")
        wandb.log(plots_to_log)
        print("✅ All plots logged successfully.")
    else:
        print("\n--> No plots were generated to log.")
print("hi")