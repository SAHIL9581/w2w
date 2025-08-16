# utils/plotting.py

import matplotlib.pyplot as plt, numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import warnings
from .config import LOG_CONFIG, LOG_PLOT_COLORS

warnings.filterwarnings("ignore", category=FutureWarning)

def plot_well_log(df, well_name_to_plot):
    """Generates and returns a matplotlib figure for the well log."""
    # Find which tracks have data to plot
    tracks_with_data = [t for t in LOG_CONFIG + [{'mnemonic': 'CLASSIFICATION'}] if t['mnemonic'] in df.columns]
    num_tracks = len(tracks_with_data)
    if num_tracks == 0: return None # No data to plot

    fig, axs = plt.subplots(nrows=1, ncols=num_tracks, figsize=(num_tracks * 1.7, 15), sharey=True)
    if num_tracks == 1: axs = [axs] # Ensure axs is always a list
    fig.suptitle(f"Well Log: {well_name_to_plot}", fontsize=18, y=1.0)
    min_depth, max_depth = df['DEPTH'].min(), df['DEPTH'].max()
    color_index = 0
    
    for i, track_info in enumerate(tracks_with_data):
        ax = axs[i]
        mnemonic = track_info['mnemonic']
        if i == 0: ax.set_ylabel("Depth (m)")
        ax.set_ylim(max_depth, min_depth); ax.grid(which='major', color='lightgray', linestyle='-')
        ax.xaxis.set_ticks_position('top'); ax.xaxis.set_label_position('top')
        curve_data = df[['DEPTH', mnemonic]].dropna()
        if curve_data.empty: continue
        if mnemonic == 'CLASSIFICATION':
            ax.set_xlabel(mnemonic)
            cmap = plt.get_cmap('viridis', int(df[mnemonic].max() + 1))
            ax.imshow(curve_data[mnemonic].values[:, np.newaxis], aspect='auto', extent=[0, 1, max_depth, min_depth], cmap=cmap)
            ax.set_xticks([])
        else:
            color = LOG_PLOT_COLORS[color_index % len(LOG_PLOT_COLORS)]
            min_val, max_val = track_info['range']
            ax.set_xlabel(f"{mnemonic}\n({min_val}-{max_val})", color=color)
            ax.plot(curve_data[mnemonic], curve_data['DEPTH'], color=color, lw=0.7)
            ax.set_xlim(min_val, max_val)
            if track_info.get('log_scale', False): ax.set_xscale('log')
            color_index += 1
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def plot_elbow_method(X, title_suffix=""):
    """Generates and returns a matplotlib figure for the elbow method."""
    ks = range(2, 11)
    inertias = [KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X).inertia_ for k in ks]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ks, inertias, '-o'); ax.set_xlabel('Number of Clusters (k)'); ax.set_ylabel('Inertia')
    ax.set_title(f'Elbow Method for {title_suffix}'); ax.set_xticks(ks); ax.grid(True)
    return fig

def plot_tsne_visualization(X, y_labels, title_suffix=""):
    """Generates and returns a matplotlib figure for t-SNE visualization."""
    perplexity = min(30, len(X) - 1) if len(X) > 1 else 1
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='random', learning_rate='auto')
    tsne_features = tsne.fit_transform(X)
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(tsne_features[:,0], tsne_features[:,1], c=y_labels, cmap='viridis', alpha=0.7)
    ax.set_xlabel('t-SNE Component 1'); ax.set_ylabel('t-SNE Component 2'); ax.set_title(f't-SNE Visualization for {title_suffix}'); ax.grid(True)
    if len(np.unique(y_labels)) > 1: ax.legend(*scatter.legend_elements(), title='Clusters')
    return fig