import os
import pandas as pd
import lasio
import json
import argparse
import sys
import matplotlib.pyplot as plt
import io

import zipfile

def process_las_folder(zip_path, output_csv):
    """
    Reads all .las files from the given ZIP archive, concatenates them
    into a single DataFrame, and writes it out as a semicolon CSV.
    """
    if not os.path.isfile(zip_path) or not zipfile.is_zipfile(zip_path):
        print(f"❌ Error: '{zip_path}' is not a valid ZIP file", file=sys.stderr)
        return False

    las_dfs = []
    print(f"--> Extracting LAS files from ZIP '{zip_path}'…")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        for entry in zf.namelist():
            if not entry.lower().endswith('.las'):
                continue

            try:
                # 1) read raw bytes, decode to text
                raw_bytes = zf.read(entry)
                text = raw_bytes.decode('utf-8', errors='replace')

                # 2) wrap text in StringIO so lasio sees a file‐like text stream
                sio = io.StringIO(text)
                las = lasio.read(sio)

                # 3) build DataFrame
                df = las.df().reset_index()
                if hasattr(las.well, 'WELL') and hasattr(las.well.WELL, 'value'):
                    well_name = las.well.WELL.value
                else:
                    well_name = os.path.splitext(os.path.basename(entry))[0]
                df['WELL'] = str(well_name)
                df['GROUP'] = 'UNKNOWN'
                for param in las.params:
                    if 'GROUP' in param.mnemonic.upper():
                        df['GROUP'] = str(param.value)

                las_dfs.append(df)
                print(f"    ↳ Parsed {entry}")

            except Exception as e:
                print(f"    – ⚠️  Could not parse {entry}: {e}", file=sys.stderr)

    if not las_dfs:
        print(f"❌ Error: No .las files successfully read from '{zip_path}'", file=sys.stderr)
        return False

    # concatenate & rename
    master_df = pd.concat(las_dfs, ignore_index=True)
    if 'DEPT' in master_df.columns:
        master_df.rename(columns={'DEPT': 'DEPTH_MD'}, inplace=True)

    # write CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    master_df.to_csv(output_csv, index=False, sep=';')
    print(f"✅ Master dataset written to '{output_csv}'")
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
    parser = argparse.ArgumentParser(
        description="Process a folder of LAS files and generate correlation plots."
    )
    parser.add_argument(
        '--config', default='config.json',
        help="Path to the JSON configuration file."
    )
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"❌ Error: Config file '{args.config}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(args.config) as f:
        config = json.load(f)

    # --- 1. Always regenerate the master CSV from LAS files ---
    source_folder = config['paths']['zip_folder']
    csv_path      = config['paths']['processed_csv_path']

    print(f"🔄 Generating master CSV at '{csv_path}' from LAS files in '{source_folder}'…")
    success = process_las_folder(zip_path=source_folder, output_csv=csv_path)
    if not success:
        sys.exit(1)

    # --- 2. Load the freshly‐generated CSV and make plots ---
    df = pd.read_csv(csv_path, sep=';', low_memory=False)

    wells = sorted(df['WELL'].astype(str).unique())
    if len(wells) < 2:
        print("⚠️  Not enough unique wells to create correlation plots.")
        return

    well_pairs = [(wells[i], wells[i+1]) for i in range(len(wells)-1)]
    plots_dir  = 'inference_plots'
    print(f"\n--- Generating {len(well_pairs)} well‐pair plots into '{plots_dir}' ---")
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