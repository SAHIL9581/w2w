# generate_all_plots.py

import os, argparse, pandas as pd, matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils.data_processing import find_and_prepare_ml_features, load_and_combine_las_from_zip_bytes
from utils.plotting import plot_well_log, plot_elbow_method, plot_tsne_visualization

def main():
    parser = argparse.ArgumentParser(description="Generate plots from a ZIP of LAS files.")
    parser.add_argument('plot_type', choices=['well-log', 'elbow', 'tsne'], help="Plot type to generate.")
    parser.add_argument("input_zip", help="Path to the input .zip file.")
    args = parser.parse_args()
    try:
        with open(args.input_zip, 'rb') as f: zip_bytes = f.read()
    except FileNotFoundError: print(f"Error: File not found at {args.input_zip}"); return
    master_df = load_and_combine_las_from_zip_bytes(zip_bytes)
    if master_df is None: print("Aborting: Could not load data."); return
    output_dir = "output_plots"; os.makedirs(output_dir, exist_ok=True)
    scopes = master_df['WELL'].unique() if args.plot_type == 'well-log' else [None] + list(master_df['WELL'].unique())
    for scope in scopes:
        s_name = "all_wells" if scope is None else scope
        df = master_df if scope is None else master_df[master_df['WELL'] == scope]
        if df.empty: continue
        safe_name = "".join(c for c in s_name if c.isalnum()).replace(' ', '_')
        fig = None
        if args.plot_type == 'well-log':
            X = find_and_prepare_ml_features(df)
            df_plot = df.join(pd.DataFrame(KMeans(n_clusters=7, random_state=42, n_init='auto').fit(X).labels_, index=X.index, columns=['CLASSIFICATION'])).ffill().bfill() if X is not None else df
            fig = plot_well_log(df_plot, s_name); filename = f"well_log_{safe_name}.png"
        else:
            X = find_and_prepare_ml_features(df)
            if X is None: continue
            if args.plot_type == 'elbow': fig = plot_elbow_method(X, s_name); filename = f"elbow_{safe_name}.png"
            elif args.plot_type == 'tsne':
                labels = KMeans(n_clusters=7, random_state=42, n_init='auto').fit(X).labels_
                fig = plot_tsne_visualization(X, labels, s_name); filename = f"tsne_{safe_name}.png"
        if fig:
            out_path = os.path.join(output_dir, filename)
            fig.savefig(out_path, dpi=300, bbox_inches='tight'); plt.close(fig); print(f"✅ Plot saved: {out_path}")

if __name__ == '__main__':
    main()