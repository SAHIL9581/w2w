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

# Suppress warnings from KMeans to keep the output clean
warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn.cluster._kmeans')

# --- Configuration for Well Log Plot ---
LOG_PLOT_EXCLUDE_CURVES = [] 
LOG_PLOT_COLOR_PALETTE = plt.cm.get_cmap('Set2').colors 

# --- DATA LOADING & PREPROCESSING (Shared by ML Plots) ---
def load_and_preprocess_for_ml(csv_filepath):
    """Loads and prepares the data specifically for machine learning plots."""
    try:
        df = pd.read_csv(csv_filepath, header=0)
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at '{csv_filepath}'")
        return None, None
        
    df.columns = ['Depth', 'Gamma-ray', 'Shale_volume', 'Resistivity', 'Delta T', 'Vp', 'Vs',\
                  'density', 'density_calculated', 'Neutron Porosity', 'Density_porosity', 'Poissons_ratio','classification']
    
    df.fillna(value=0, inplace=True)
    feature_columns = 'Gamma-ray density Vp Vs Density_porosity Poissons_ratio'.split()
    X = df[feature_columns]
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(X.values)
    X_scaled_df = pd.DataFrame(x_scaled, columns=feature_columns)
    
    print(f"✅ Data loaded and preprocessed for ML plots from '{os.path.basename(csv_filepath)}'.")
    return df, X_scaled_df

# --- PLOTTING FUNCTIONS ---

def plot_well_log(csv_filepath, output_filepath):
    """Generates the main multi-track well log plot with all aesthetic improvements and descriptive headers."""
    print("\n--- Generating Main Well Log Plot ---")
    
    try:
        df = pd.read_csv(csv_filepath, header=0)
        
        # --- THIS IS THE FIX: Assign meaningful headers to the columns ---
        meaningful_headers = [
            'Depth', 'Gamma-ray', 'Shale_volume', 'Resistivity', 'Delta T', 'Vp', 'Vs',
            'Density', 'Density_calculated', 'Neutron Porosity', 'Density_porosity', 'Poissons_ratio', 'Classification'
        ]
        
        if len(df.columns) == len(meaningful_headers):
            df.columns = meaningful_headers
            print("   ✅ Applied descriptive headers to plot tracks.")
        else:
            print(f"   ⚠️ Warning: CSV has {len(df.columns)} columns, but {len(meaningful_headers)} headers were expected. Using generic names.")
            df.columns = ['DEPTH'] + [f'CURVE_{i}' for i in range(1, len(df.columns))]

        # Standardize to uppercase for internal logic
        df.columns = [col.upper() for col in df.columns]
        # --- END OF FIX ---
        
        depth_col_name = next((col for col in df.columns if col in ['DEPT', 'DEPTH']), None)
        if not depth_col_name:
            raise ValueError("Could not find a 'DEPT' or 'DEPTH' column.")
        
    except Exception as e:
        print(f"❌ Error during well log data loading: {e}")
        return

    tracks, color_index = [], 0
    classification_curve = next((c for c in df.columns if 'CLASSIFICATION' in c), None)

    for curve_name in df.columns:
        if curve_name in ['DEPTH'] + LOG_PLOT_EXCLUDE_CURVES:
            continue

        curve_data = df[['DEPTH', curve_name]].dropna()
        if curve_data.empty: continue

        trace = {'data': curve_data, 'curve': curve_name, 'color': LOG_PLOT_COLOR_PALETTE[color_index % len(LOG_PLOT_COLOR_PALETTE)]}
        if curve_name == classification_curve:
            trace['fill'] = 1
            num_classes = len(curve_data[curve_name].unique())
            trace['cmap'] = plt.cm.get_cmap('viridis', num_classes)
        else:
            trace['fill'] = 0
            min_val, max_val = curve_data[curve_name].min(), curve_data[curve_name].max()
            padding = (max_val - min_val) * 0.05 if max_val > min_val else 1
            trace['range'] = (min_val - padding, max_val + padding)
        tracks.append({'traces': [trace]})
        color_index += 1
    
    fig_width = 6 + len(tracks) * 2; fig_height = 16
    min_depth, max_depth = df['DEPTH'].min(), df['DEPTH'].max()
    fig, axs = plt.subplots(nrows=1, ncols=len(tracks), figsize=(fig_width, fig_height), sharey=True)
    if len(tracks) == 1: axs = [axs]
    fig.suptitle(f"Well Log: {os.path.basename(csv_filepath)}", fontsize=22, fontweight='bold', y=0.96)
    
    ax1 = axs[0]
    ax1.set_ylabel("Depth", fontsize=16, fontweight='bold')
    ax1.set_ylim(max_depth, min_depth)
    ax1.yaxis.set_major_locator(MultipleLocator(500)); ax1.yaxis.set_minor_locator(MultipleLocator(100))
    ax1.tick_params(axis='y', labelsize=12)
    ax1.grid(which='major', color='gray', linestyle='-', linewidth=0.7)
    ax1.grid(which='minor', color='lightgray', linestyle=':', linewidth=0.5)

    for ax, track_info in zip(axs, tracks):
        trace = track_info['traces'][0]; curve_data = trace['data']; twin_ax = ax.twiny()
        if trace['fill'] == 1:
            y = curve_data['DEPTH'].to_numpy(); z = curve_data[trace['curve']].to_numpy()
            y_mesh = np.append(y, y[-1] + np.diff(y)[-1]); x_mesh = np.array([0, 1]); z_mesh = z.reshape(-1, 1)
            twin_ax.pcolormesh(x_mesh, y_mesh, z_mesh, cmap=trace['cmap'], shading='auto', vmin=z.min(), vmax=z.max())
            twin_ax.set_xticks([])
        else:
            twin_ax.plot(curve_data[trace['curve']], curve_data['DEPTH'], color=trace['color'], linewidth=1.5)
            twin_ax.set_xlim(trace['range']); twin_ax.grid(which='major', color='lightgray', linestyle='--', linewidth=0.5)
        
        # This line now uses the meaningful names for the plot headers
        twin_ax.set_xlabel(trace['curve'].replace('_', ' ').title(), fontsize=14, color=trace['color'], fontweight='bold')
        twin_ax.tick_params(axis='x', labelsize=12, colors=trace['color']); twin_ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        twin_ax.spines['top'].set_position(('outward', 10)); twin_ax.spines['top'].set_edgecolor(trace['color']); twin_ax.spines['top'].set_linewidth(2)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    fig.savefig(output_filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Well Log Plot successfully saved to: {output_filepath}")

def plot_elbow_method(X, output_filepath):
    """Generates the Elbow Method plot to find the optimal number of clusters."""
    print("\n--- Generating Elbow Method Plot ---")
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

def plot_tsne_visualization(X, y_labels, output_filepath):
    """Generates a t-SNE plot to visualize clusters in 2D."""
    print("\n--- Generating t-SNE Visualization Plot ---")
    model = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30, random_state=42)
    tsne_features = model.fit_transform(X)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(tsne_features[:,0], tsne_features[:,1], c=y_labels, cmap='viridis', alpha=0.7)
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    plt.title('t-SNE 2D Visualization of Well Data Clusters', fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(handles=scatter.legend_elements(num=len(np.unique(y_labels)))[0], labels=[f'Cluster {i}' for i in np.unique(y_labels)])
    
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ t-SNE Plot successfully saved to: {output_filepath}")


# --- MAIN SCRIPT EXECUTION ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate various plots from well log data.")
    
    parser.add_argument(
        '--plot-type', 
        type=str, 
        required=True,
        choices=['well-log', 'elbow', 'tsne'],
        help="The type of plot to generate: 'well-log' for the main log, 'elbow' for the KMeans inertia plot, 'tsne' for the 2D cluster visualization."
    )
    parser.add_argument("input_file", type=str, help="Path to the input .csv file (e.g., 'WellA.csv').")
    
    args = parser.parse_args()

    output_dir = "output_plots"
    
    if args.plot_type == 'well-log':
        output_path = os.path.join(output_dir, "professional_well_log.png")
        plot_well_log(args.input_file, output_path)
        
    elif args.plot_type in ['elbow', 'tsne']:
        original_df, scaled_X = load_and_preprocess_for_ml(args.input_file)
        
        if scaled_X is None:
            print("Aborting due to data loading error.")
        elif args.plot_type == 'elbow':
            output_path = os.path.join(output_dir, "elbow_method_plot.png")
            plot_elbow_method(scaled_X, output_path)
        elif args.plot_type == 'tsne':
            output_path = os.path.join(output_dir, "tsne_clusters_plot.png")
            kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_X)
            plot_tsne_visualization(scaled_X, cluster_labels, output_path)
    
    else:
        print(f"❌ Error: Plot type '{args.plot_type}' is not recognized.")