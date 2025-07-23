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
                raw_bytes = zf.read(entry)
                text = raw_bytes.decode('utf-8', errors='replace')
                sio = io.StringIO(text)
                las = lasio.read(sio)
                
                df = las.df().reset_index()
                
                well_name = os.path.splitext(os.path.basename(entry))[0]
                if hasattr(las.well, 'WELL') and hasattr(las.well.WELL, 'value') and las.well.WELL.value:
                    well_name = las.well.WELL.value
                elif hasattr(las.well, 'APIN') and hasattr(las.well.APIN, 'value') and las.well.APIN.value:
                    well_name = las.well.APIN.value
                
                df['WELL'] = str(well_name)
                df['GROUP'] = 'UNKNOWN'
                for param in las.params:
                    if 'GROUP' in param.mnemonic.upper():
                        df['GROUP'] = str(param.value)

                las_dfs.append(df)
                print(f"      ↳ Parsed {entry} (Well ID: '{well_name}')")
            except Exception as e:
                print(f"      – ⚠️  Could not parse {entry}: {e}", file=sys.stderr)

    if not las_dfs:
        print(f"❌ Error: No .las files successfully read from '{zip_path}'", file=sys.stderr)
        return False

    master_df = pd.concat(las_dfs, ignore_index=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    master_df.to_csv(output_csv, index=False, sep=';')
    print(f"✅ Master dataset written to '{output_csv}'")
    return True

def plot_well_pair(df, w1, w2, out_dir, curve_to_plot, curve_label):
    """
    Plots a standardized correlation curve ('CORRELATION_CURVE')
    and uses a dynamic label for the x-axis.
    """
    df1 = df[df['WELL'] == w1]
    df2 = df[df['WELL'] == w2]
    if df1.empty or df2.empty:
        return

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 6), dpi=150)
    
    if curve_to_plot in df1.columns and 'DEPTH_MD' in df1.columns:
        ax.plot(df1[curve_to_plot], df1['DEPTH_MD'], label=w1, color='blue')
        ax.plot(df2[curve_to_plot], df2['DEPTH_MD'], label=w2, color='red')
        ax.set_xlabel(curve_label)
        ax.legend()
    else:
        ax.text(0.5, 0.5, f"Curve '{curve_to_plot}' or 'DEPTH_MD'\nnot available for plotting.", ha='center', va='center')

    ax.invert_yaxis()
    ax.set_ylabel('Depth (MD)')
    ax.set_title("Well Correlation")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    fname = f"Plot_{w1.replace('/', '_')}_vs_{w2.replace('/', '_')}.png"
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved plot: {out_path}")

def find_column(df_columns, aliases):
    """Helper function to find the first matching column name from a list of aliases."""
    for alias in aliases:
        if alias in df_columns:
            return alias
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json')
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"❌ Error: Config file '{args.config}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(args.config) as f:
        config = json.load(f)

    source_folder = config['paths']['zip_folder']
    csv_path      = config['paths']['processed_csv_path']

    print(f"🔄 Generating master CSV at '{csv_path}' from LAS files in '{source_folder}'…")
    success = process_las_folder(zip_path=source_folder, output_csv=csv_path)
    if not success:
        sys.exit(1)

    df = pd.read_csv(csv_path, sep=';', dtype={'WELL': str}, low_memory=False)

    print("\n--- Standardizing Column Names ---")
    original_columns = df.columns.tolist()
    df.columns = [col.upper() for col in original_columns]
    print(f"Standardized columns to uppercase: {df.columns.tolist()}")

    DEPTH_ALIASES = ['DEPT', 'DEPTH', 'MD']
    actual_depth_col = find_column(df.columns, DEPTH_ALIASES)
    if actual_depth_col:
        print(f"  ✅ Found Depth column as '{actual_depth_col}'. Renaming to 'DEPTH_MD'.")
        df.rename(columns={actual_depth_col: 'DEPTH_MD'}, inplace=True)
    else:
        print(f"  ❌ CRITICAL ERROR: Could not find any recognized Depth column. Looked for {DEPTH_ALIASES}.")
        sys.exit(1)
        
    GR_ALIASES = ['GR', 'GRC', 'CGR', 'SGR', 'GAMMA']
    SP_ALIASES = ['SP', 'SPN']
    
    curve_to_plot = 'CORRELATION_CURVE'
    curve_label = ''
    
    actual_corr_col = find_column(df.columns, GR_ALIASES)
    if actual_corr_col:
        print(f"  ✅ Found Gamma Ray column as '{actual_corr_col}'. Using for plots.")
        curve_label = f'Gamma Ray ({actual_corr_col})'
        df.rename(columns={actual_corr_col: curve_to_plot}, inplace=True)
    else:
        # --- THIS IS THE FIX: Corrected "iinstead" to "instead" ---
        print(f"  ⚠️ Gamma Ray not found. Searching for Spontaneous Potential (SP) instead...")
        # --- END OF FIX ---
        actual_corr_col = find_column(df.columns, SP_ALIASES)
        if actual_corr_col:
            print(f"  ✅ Found Spontaneous Potential column as '{actual_corr_col}'. Using for plots.")
            curve_label = f'Spontaneous Potential ({actual_corr_col})'
            df.rename(columns={actual_corr_col: curve_to_plot}, inplace=True)
        else:
            print(f"  ❌ CRITICAL ERROR: Could not find any suitable correlation curve. Looked for GR and SP.")
            sys.exit(1)

    wells = sorted(df['WELL'].astype(str).unique())
    if len(wells) < 2:
        print("\n⚠️  Not enough unique wells to create correlation plots.")
        return

    well_pairs = [(wells[i], wells[i+1]) for i in range(len(wells)-1)]
    plots_dir  = 'inference_plots' 
    print(f"\n--- Generating {len(well_pairs)} well‐pair plots into '{plots_dir}' ---")
    for w1, w2 in well_pairs:
        plot_well_pair(df, w1, w2, plots_dir, curve_to_plot, curve_label)
    print("--- Done 🚀 ---\n")

    print("\n--- Available Well Names for Correlation ---")
    for w in wells:
        print(f"- {w}")
    print("------------------------------------------")

if __name__ == '__main__':
    main()
