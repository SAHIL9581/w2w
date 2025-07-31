# generate_all_plots.py

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import warnings
import zipfile
import io
import lasio

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- CONFIGURATION ---
LOG_CONFIG = [
    {'mnemonic': 'GR', 'range': (0, 150)},
    {'mnemonic': 'SGR', 'range': (0, 300)},
    {'mnemonic': 'RSHA', 'range': (0.2, 2000), 'log_scale': True},
    {'mnemonic': 'RMED', 'range': (0.2, 2000), 'log_scale': True},
    {'mnemonic': 'RDEP', 'range': (0.2, 2000), 'log_scale': True},
    {'mnemonic': 'RXO', 'range': (0.2, 2000), 'log_scale': True},
    {'mnemonic': 'RMIC', 'range': (0.2, 2000), 'log_scale': True},
    {'mnemonic': 'SP', 'range': (-150, 150)},
    {'mnemonic': 'DTC', 'range': (40, 200)},
    {'mnemonic': 'DTS', 'range': (80, 300)},
    {'mnemonic': 'RHOB', 'range': (1.95, 2.95)},
    {'mnemonic': 'DRHO', 'range': (-0.1, 0.1)},
    {'mnemonic': 'NPHI', 'range': (0, 0.6)},
    {'mnemonic': 'PEF', 'range': (0, 10)},
    {'mnemonic': 'CALI', 'range': (6, 17)},
    {'mnemonic': 'BS', 'range': (6, 17)},
    {'mnemonic': 'DCAL', 'range': (-1, 1)},
    {'mnemonic': 'ROP', 'range': (0, 1000)},
    {'mnemonic': 'ROPA', 'range': (0, 1000)},
    {'mnemonic': 'MUDWEIGHT', 'range': (8, 22)},
]
LOG_PLOT_COLORS = ['red', 'black', 'blue']


# --- DATA LOADING ---
def load_and_combine_las_from_zip(zip_filepath):
    print(f"\n--- Loading and Combining LAS files from '{zip_filepath}' ---")
    if not os.path.exists(zip_filepath):
        print(f"❌ Error: Input ZIP file not found at '{zip_filepath}'")
        return None
    las_dfs = []
    with zipfile.ZipFile(zip_filepath, 'r') as zf:
        for entry in zf.namelist():
            if not entry.lower().endswith('.las') or entry.startswith('__MACOSX'):
                continue
            try:
                raw_bytes = zf.read(entry)
                text = raw_bytes.decode('utf-8', errors='replace')
                las = lasio.read(io.StringIO(text))
                df = las.df().reset_index()
                well_name = las.well.WELL.value if las.well.WELL.value else os.path.splitext(os.path.basename(entry))[0]
                df['WELL'] = well_name
                las_dfs.append(df)
                print(f"  ✅ Parsed '{entry}' (Well: {well_name})")
            except Exception as e:
                print(f"  ⚠️ Could not parse '{entry}'. Error: {e}")
    if not las_dfs:
        print("❌ Error: No valid LAS files were found or parsed in the ZIP archive.")
        return None
    combined_df = pd.concat(las_dfs, ignore_index=True)
    combined_df.columns = [col.upper() for col in combined_df.columns]
    depth_col = next((col for col in combined_df.columns if col in ['DEPT', 'DEPTH']), None)
    if depth_col:
        combined_df.rename(columns={depth_col: 'DEPTH'}, inplace=True)
    else:
        print("❌ CRITICAL ERROR: No 'DEPT' or 'DEPTH' column found in any LAS file.")
        return None
    return combined_df

# --- FEATURE PREPARATION for ML ---
def find_and_prepare_ml_features(df):
    print("\n--- Preparing Features for Machine Learning ---")
    cols_to_exclude = ['DEPTH', 'WELL', 'CLASSIFICATION']
    potential_features = [col for col in df.columns if col not in cols_to_exclude]
    ml_df_base = df[potential_features].select_dtypes(include=np.number).copy()
    
    print(f"  ✅ Initially found {ml_df_base.shape[1]} numeric columns: {ml_df_base.columns.tolist()}")
    
    valid_feature_cols = []
    for col in ml_df_base.columns:
        if ml_df_base[col].nunique() > 1:
            valid_feature_cols.append(col)
        else:
            print(f"  ℹ️ Note: Column '{col}' is constant. It will not be used for clustering.")

    if len(valid_feature_cols) < 2:
        print(f"\n  ❌ CRITICAL ERROR: Found fewer than 2 non-constant columns. Cannot perform clustering.")
        return None

    ml_df_valid = ml_df_base[valid_feature_cols].copy()
    print(f"  ✅ Using {len(valid_feature_cols)} columns for clustering: {valid_feature_cols}")
    
    ml_df_valid.interpolate(method='linear', inplace=True, limit_direction='both')
    ml_df_valid.bfill(inplace=True)
    ml_df_valid.ffill(inplace=True)
    ml_df_valid.dropna(how='any', inplace=True)

    if ml_df_valid.empty:
        print("  ❌ CRITICAL ERROR: No data remains after cleaning valid features.")
        return None
        
    print(f"  ✅ Cleaned data complete. Using {len(ml_df_valid)} data points for clustering.")

    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(ml_df_valid)
    X_scaled_df = pd.DataFrame(x_scaled, columns=ml_df_valid.columns, index=ml_df_valid.index)
    return X_scaled_df

# --- PLOTTING FUNCTIONS ---
def plot_well_log(df, output_filepath, source_filename):
    print(f"\n--- Generating Professional Well Log Plot for well: '{df['WELL'].unique()[0]}' ---")
    well_name_to_plot = df['WELL'].unique()[0]
    df_well = df[df['WELL'] == well_name_to_plot].copy()
    
    tracks_to_plot = LOG_CONFIG + [{'mnemonic': 'CLASSIFICATION'}]
    num_tracks = len(tracks_to_plot)

    fig, axs = plt.subplots(nrows=1, ncols=num_tracks, figsize=(num_tracks * 1.5, 15), sharey=True,
                            gridspec_kw={'width_ratios': [1]*len(LOG_CONFIG) + [0.5]})
    
    fig.suptitle(f"Well Log: {well_name_to_plot}", fontsize=18, y=1.02)
    
    min_depth, max_depth = df_well['DEPTH'].min(), df_well['DEPTH'].max()
    
    ax1 = axs[0]
    ax1.set_ylabel("Depth (m)", fontsize=12, fontweight='bold')
    ax1.set_ylim(max_depth, min_depth)
    ax1.yaxis.set_major_locator(MultipleLocator(250))
    ax1.yaxis.set_minor_locator(MultipleLocator(50))
    ax1.tick_params(axis='y', which='major', labelsize=10, labelcolor='black', width=1.5, length=10)
    ax1.tick_params(axis='y', which='minor', width=1, length=5)
    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')
    ax1.spines['left'].set_linewidth(1.5)

    color_index = 0
    for i, track_info in enumerate(tracks_to_plot):
        ax = axs[i]
        mnemonic = track_info['mnemonic']
        
        # --- THIS IS THE FIX ---
        # Increase this number to make the tracks taller and more slender
        ax.set_box_aspect(25) 
        
        ax.grid(which='major', color='lightgray', linestyle='-')
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.tick_params(axis='x', labelsize=8)
        
        if mnemonic not in df_well.columns:
            ax.set_xlabel(mnemonic, color='lightgray', fontsize=10, fontweight='bold')
            ax.set_xticks([])
            continue

        curve_data = df_well[['DEPTH', mnemonic]].dropna()
        if curve_data.empty:
            ax.set_xlabel(mnemonic, color='lightgray', fontsize=10, fontweight='bold')
            ax.set_xticks([])
            continue

        if mnemonic == 'CLASSIFICATION':
            ax.set_xlabel(mnemonic, color='black', fontsize=10, fontweight='bold')
            facies_labels = sorted(curve_data[mnemonic].unique())
            num_facies = len(facies_labels)
            cmap = plt.colormaps.get_cmap('viridis')
            colors = [cmap(i / num_facies) for i in range(num_facies)]
            facies_color_map = dict(zip(facies_labels, colors))
            for facies_val, color in facies_color_map.items():
                ax.fill_betweenx(curve_data['DEPTH'], 0, 1, where=(curve_data[mnemonic] == facies_val), facecolor=color)
            ax.set_xlim(0, 1)
            ax.set_xticks([])
        else:
            color = LOG_PLOT_COLORS[color_index % len(LOG_PLOT_COLORS)]
            min_val, max_val = track_info['range']
            min_str, max_str = f"{min_val:g}", f"{max_val:g}"
            title_str = f"{mnemonic}\n({min_str} - {max_str})"
            ax.set_xlabel(title_str, color=color, fontsize=9, fontweight='bold', ha='center')
            
            ax.plot(curve_data[mnemonic], curve_data['DEPTH'], color=color, linewidth=0.7)
            
            ax.set_xlim(track_info['range'])
            if track_info.get('log_scale', False):
                ax.set_xscale('log')
                
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            color_index += 1

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    fig.savefig(output_filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Professional Well Log Plot successfully saved to: {output_filepath}")

def plot_elbow_method(X, output_filepath):
    # This function remains unchanged
    print(f"\n--- Generating Elbow Method Plot ---")
    ks = range(1, 10)
    inertias = []
    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        inertias.append(model.inertia_)
    plt.figure(figsize=(10, 6))
    plt.plot(ks, inertias, '-o', color='b', markerfacecolor='red', markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=14)
    plt.ylabel('Inertia (Sum of Squared Distances)', fontsize=14)
    plt.title('Elbow Method for Optimal k', fontsize=16, fontweight='bold')
    plt.xticks(ks)
    plt.grid(True, linestyle='--', alpha=0.6)
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Elbow Method Plot successfully saved to: {output_filepath}")

def plot_tsne_visualization(X, y_labels, output_filepath, title_suffix=""):
    # This function remains unchanged
    print(f"\n--- Generating t-SNE Visualization Plot {title_suffix} ---")
    perplexity_value = min(30, len(X) - 1) if len(X) > 1 else 1
    model = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=perplexity_value, random_state=42)
    tsne_features = model.fit_transform(X)
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(tsne_features[:,0], tsne_features[:,1], c=y_labels, cmap='viridis', alpha=0.7)
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    plt.title(f't-SNE Visualization {title_suffix}', fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    if len(np.unique(y_labels)) > 0:
        plt.legend(handles=scatter.legend_elements(num=len(np.unique(y_labels)))[0], labels=[f'Cluster {i}' for i in np.unique(y_labels)])
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ t-SNE Plot successfully saved to: {output_filepath}")

# --- MAIN SCRIPT EXECUTION ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate plots for all wells from a ZIP file.")
    parser.add_argument('plot_type', type=str, choices=['well-log', 'elbow', 'tsne'], help="The type of plot to generate for all wells.")
    parser.add_argument("input_zip", type=str, help="Path to the input .zip file.")
    args = parser.parse_args()

    master_df = load_and_combine_las_from_zip(args.input_zip)
    if master_df is None:
        print("\nAborting due to data loading errors.")
        exit()

    unique_wells = master_df['WELL'].unique().tolist()
    scopes_to_process = unique_wells if args.plot_type == 'well-log' else [None] + unique_wells
    output_dir = "output_plots"

    for scope in scopes_to_process:
        if scope is None:
            scope_name = "all_wells"
            safe_scope_name = "all_wells"
            df_to_use = master_df
            print(f"\n\n=================================================")
            print(f"  PROCESSING PLOTS FOR: ALL WELLS COMBINED")
            print(f"=================================================")
        else:
            scope_name = scope
            safe_scope_name = "".join(c for c in scope_name if c.isalnum() or c in (' ', '.')).rstrip().replace(' ', '_')
            df_to_use = master_df[master_df['WELL'] == scope].copy()
            print(f"\n\n=================================================")
            print(f"  PROCESSING PLOTS FOR WELL: {scope_name}")
            print(f"=================================================")

        if df_to_use.empty:
            print(f"  ⚠️ Skipping scope '{scope_name}' as it contains no data.")
            continue
            
        if args.plot_type == 'well-log':
            scaled_X = find_and_prepare_ml_features(df_to_use)
            
            if scaled_X is not None:
                kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
                predicted_labels = kmeans.fit_predict(scaled_X)
                
                labels_df = pd.DataFrame(predicted_labels, index=scaled_X.index, columns=['CLASSIFICATION'])
                df_with_classification = df_to_use.join(labels_df)

                df_with_classification['CLASSIFICATION'].ffill(inplace=True)
                df_with_classification['CLASSIFICATION'].bfill(inplace=True)
                print("  ✅ Classification extended to full well depth.")

                filename = f"well_log_with_classification_{safe_scope_name}.png"
                output_path = os.path.join(output_dir, filename)
                plot_well_log(df_with_classification, output_path, os.path.basename(args.input_zip))
            else:
                print(f"  ⚠️ Could not generate classification for '{scope_name}'. Plotting original data only.")
                filename = f"well_log_original_only_{safe_scope_name}.png"
                output_path = os.path.join(output_dir, filename)
                plot_well_log(df_to_use, output_path, os.path.basename(args.input_zip))
        
        else:
            scaled_X = find_and_prepare_ml_features(df_to_use)
            if scaled_X is None:
                print(f"\n  Aborting '{args.plot_type}' plot for '{scope_name}' because ML features cannot be prepared.")
                continue

            if args.plot_type == 'elbow':
                filename = f"elbow_method_{safe_scope_name}.png"
                output_path = os.path.join(output_dir, filename)
                plot_elbow_method(scaled_X, output_path)

            elif args.plot_type == 'tsne':
                kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
                predicted_labels = kmeans.fit_predict(scaled_X)
                filename = f"tsne_by_kmeans_{safe_scope_name}.png"
                output_path = os.path.join(output_dir, filename)
                plot_tsne_visualization(scaled_X, predicted_labels, output_path, title_suffix=f"by K-Means Clusters ({safe_scope_name})")
