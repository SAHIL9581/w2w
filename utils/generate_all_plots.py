# utils/generate_all_plots.py
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Import shared config
from .config import LOG_CONFIG, LOG_PLOT_COLORS

# --- PLOTTING FUNCTIONS (MODIFIED FOR API USE) ---

def plot_well_log(df: pd.DataFrame, well_name_to_plot: str):
    """
    Generates a Professional Well Log Plot and returns the figure object.
    """
    print(f"\n--- Generating Well Log Plot for well: '{well_name_to_plot}' ---")
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
    ax1.tick_params(axis='y', which='major', labelsize=10)

    color_index = 0
    for i, track_info in enumerate(tracks_to_plot):
        ax = axs[i]
        mnemonic = track_info['mnemonic']
        
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
            cmap = plt.get_cmap('viridis', num_facies)
            colors = [cmap(i) for i in range(num_facies)]
            facies_color_map = dict(zip(facies_labels, colors))
            for facies_val, color in facies_color_map.items():
                ax.fill_betweenx(curve_data['DEPTH'], 0, 1, where=(curve_data[mnemonic] == facies_val), facecolor=color)
            ax.set_xlim(0, 1)
            ax.set_xticks([])
        else:
            color = LOG_PLOT_COLORS[color_index % len(LOG_PLOT_COLORS)]
            min_val, max_val = track_info['range']
            title_str = f"{mnemonic}\n({min_val:g} - {max_val:g})"
            ax.set_xlabel(title_str, color=color, fontsize=9, fontweight='bold', ha='center')
            
            ax.plot(curve_data[mnemonic], curve_data['DEPTH'], color=color, linewidth=0.7)
            
            ax.set_xlim(track_info['range'])
            if track_info.get('log_scale', False):
                ax.set_xscale('log')
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            color_index += 1

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # IMPORTANT: Return the figure object for the API to handle
    return fig

def plot_elbow_method(X: pd.DataFrame, title_name: str):
    """
    Generates an Elbow Method plot and returns the figure object.
    """
    print(f"\n--- Generating Elbow Method Plot for {title_name} ---")
    ks = range(1, 10)
    inertias = []
    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init='auto')
        model.fit(X)
        inertias.append(model.inertia_)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ks, inertias, '-o', color='b', markerfacecolor='red', markersize=8)
    ax.set_xlabel('Number of Clusters (k)', fontsize=14)
    ax.set_ylabel('Inertia (Sum of Squared Distances)', fontsize=14)
    ax.set_title(f'Elbow Method for {title_name}', fontsize=16, fontweight='bold')
    ax.set_xticks(ks)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # IMPORTANT: Return the figure object
    return fig

def plot_tsne_visualization(X: pd.DataFrame, y_labels: np.ndarray, title_name: str):
    """
    Generates a t-SNE visualization and returns the figure object.
    """
    print(f"\n--- Generating t-SNE Visualization for {title_name} ---")
    perplexity_value = min(30, len(X) - 1) if len(X) > 1 else 1
    model = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=perplexity_value, random_state=42)
    tsne_features = model.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(tsne_features[:,0], tsne_features[:,1], c=y_labels, cmap='viridis', alpha=0.7)
    ax.set_xlabel('t-SNE Component 1', fontsize=14)
    ax.set_ylabel('t-SNE Component 2', fontsize=14)
    ax.set_title(f't-SNE Visualization for {title_name}', fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    unique_labels = np.unique(y_labels)
    if len(unique_labels) > 0:
        legend_handles, _ = scatter.legend_elements(num=len(unique_labels))
        legend_labels = [f'Cluster {i}' for i in unique_labels]
        ax.legend(legend_handles, legend_labels)
        
    # IMPORTANT: Return the figure object
    return fig