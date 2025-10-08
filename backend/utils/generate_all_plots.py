# utils/generate_all_plots.py
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore", category=FutureWarning)
from .config import LOG_CONFIG, LOG_PLOT_COLORS

def plot_well_log(df: pd.DataFrame, well_name_to_plot: str):
    """
    Generates a professional well log plot using a simple red, black, blue color rotation.
    """
    df_well = df[df['WELL'] == well_name_to_plot].copy()
    if df_well.empty or 'DEPTH' not in df_well.columns:
        print(f"  ⚠️ Skipping plot for '{well_name_to_plot}': No data found.")
        return None

    log_config_map = {log['mnemonic']: log for log in LOG_CONFIG}
    
    tracks_to_plot = []
    for log in LOG_CONFIG:
        if log['mnemonic'] in df_well.columns and df_well[log['mnemonic']].notna().any():
            tracks_to_plot.append(log['mnemonic'])
            
    has_classification = 'CLASSIFICATION' in df_well.columns and df_well['CLASSIFICATION'].notna().any()
    if has_classification:
        tracks_to_plot.append('CLASSIFICATION')
    
    if not tracks_to_plot:
        print(f"  ⚠️ Skipping plot for '{well_name_to_plot}': No plottable curves found.")
        return None

    num_tracks = len(tracks_to_plot)
    fig, axs = plt.subplots(nrows=1, ncols=num_tracks, figsize=(num_tracks * 2, 15), sharey=True, squeeze=False)
    axs = axs.flatten()
    
    fig.suptitle(f"Well Log: {well_name_to_plot}", fontsize=16, y=0.95)
    
    min_depth, max_depth = df_well['DEPTH'].min(), df_well['DEPTH'].max()
    axs[0].set_ylim(max_depth, min_depth)
    axs[0].set_ylabel("Depth (m)")

    color_index = 0
    for i, mnemonic in enumerate(tracks_to_plot):
        ax = axs[i]
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.xaxis.set_ticks_position('top'); ax.xaxis.set_label_position('top')

        if mnemonic == 'CLASSIFICATION':
            # --- THIS IS THE FIX ---
            ax.set_title("CLASSIFICATION")
            # --- END OF FIX ---
            facies_labels = sorted(df_well[mnemonic].dropna().unique())
            num_facies = len(facies_labels)
            cmap = plt.get_cmap('viridis', num_facies)
            colors = cmap(np.linspace(0, 1, num_facies))
            facies_color_map = dict(zip(facies_labels, colors))
            
            for facies_val, color in facies_color_map.items():
                ax.fill_betweenx(df_well['DEPTH'], 0, 1, where=(df_well[mnemonic] == facies_val), facecolor=color)
            ax.set_xlim(0, 1); ax.set_xticks([])
        else:
            track_info = log_config_map.get(mnemonic, {})
            color = LOG_PLOT_COLORS[color_index % len(LOG_PLOT_COLORS)]
            
            ax.set_title(f"{mnemonic}\n({track_info.get('range', ['N/A'])[0]} - {track_info.get('range', ['N/A'])[1]})", color=color, fontsize=9, fontweight='bold')
            ax.plot(df_well[mnemonic], df_well['DEPTH'], color=color, linewidth=0.7)
            
            if track_info.get('log_scale'):
                ax.set_xscale('log')
            if 'range' in track_info:
                ax.set_xlim(track_info['range'])
            
            color_index += 1

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig

def plot_elbow_method(X: pd.DataFrame, title_name: str):
    """Generates an Elbow Method plot (unchanged)."""
    if X is None or X.empty: return None
    ks, inertias = range(1, 10), []
    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X)
        inertias.append(model.inertia_)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ks, inertias, '-o', color='b', markerfacecolor='red', markersize=8)
    ax.set_xlabel('Number of Clusters (k)', fontsize=14); ax.set_ylabel('Inertia', fontsize=14)
    ax.set_title(f'Elbow Method for {title_name}', fontsize=16, fontweight='bold')
    ax.set_xticks(ks); ax.grid(True, linestyle='--', alpha=0.6); return fig

def plot_tsne_visualization(X: pd.DataFrame, y_labels: np.ndarray, title_name: str):
    """Generates a t-SNE visualization (unchanged)."""
    if X is None or X.empty: return None
    perplexity_value = min(30, len(X) - 1) if len(X) > 1 else 1
    model = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=perplexity_value, random_state=42)
    tsne_features = model.fit_transform(X)
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(tsne_features[:,0], tsne_features[:,1], c=y_labels, cmap='viridis', alpha=0.7)
    ax.set_xlabel('t-SNE Component 1', fontsize=14); ax.set_ylabel('t-SNE Component 2', fontsize=14)
    ax.set_title(f't-SNE Visualization for {title_name}', fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    unique_labels = np.unique(y_labels)
    if len(unique_labels) > 0:
        handles, _ = scatter.legend_elements(num=len(unique_labels))
        labels = [f'Cluster {i}' for i in unique_labels]
        ax.legend(handles, labels)
    return fig