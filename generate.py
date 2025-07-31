# generate.py

import os
import zipfile
import io
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import lasio
from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from typing import Literal

warnings.filterwarnings("ignore", category=FutureWarning)

# --- CONFIGURATION ---
LOG_CONFIG = [
    {'mnemonic': 'GR', 'range': (0, 150)}, {'mnemonic': 'SGR', 'range': (0, 300)},
    {'mnemonic': 'RSHA', 'range': (0.2, 2000), 'log_scale': True}, {'mnemonic': 'RMED', 'range': (0.2, 2000), 'log_scale': True},
    {'mnemonic': 'RDEP', 'range': (0.2, 2000), 'log_scale': True}, {'mnemonic': 'RXO', 'range': (0.2, 2000), 'log_scale': True},
    {'mnemonic': 'RMIC', 'range': (0.2, 2000), 'log_scale': True}, {'mnemonic': 'SP', 'range': (-150, 150)},
    {'mnemonic': 'DTC', 'range': (40, 200)}, {'mnemonic': 'DTS', 'range': (80, 300)},
    {'mnemonic': 'RHOB', 'range': (1.95, 2.95)}, {'mnemonic': 'DRHO', 'range': (-0.1, 0.1)},
    {'mnemonic': 'NPHI', 'range': (0, 0.6)}, {'mnemonic': 'PEF', 'range': (0, 10)},
    {'mnemonic': 'CALI', 'range': (6, 17)}, {'mnemonic': 'BS', 'range': (6, 17)},
    {'mnemonic': 'DCAL', 'range': (-1, 1)}, {'mnemonic': 'ROP', 'range': (0, 1000)},
    {'mnemonic': 'ROPA', 'range': (0, 1000)}, {'mnemonic': 'MUDWEIGHT', 'range': (8, 22)},
]
LOG_PLOT_COLORS = ['red', 'black', 'blue']

CLIENT_MAPPING = {
  "CALI": "None", "RSHA": "None", "RMED": "None", "RDEP": "RES", "RHOB": "None",
  "GR": "None", "SGR": "SN18", "NPHI": "None", "PEF": "None", "DTC": "None",
  "SP": "SP", "BS": "None", "ROP": "None", "DTS": "None", "DCAL": "None",
  "DRHO": "None", "MUDWEIGHT": "None", "RMIC": "None", "ROPA": "None", "RXO": "IND"
}

# --- DATA & MAPPING FUNCTIONS ---
def map_las_headers(reference: list, unknown: list) -> dict:
    print("\n--- Mapping LAS Headers to Standard ---")
    final_mapping = {}
    unknown_set = set(unknown)
    for ref_header in reference:
        target_header = CLIENT_MAPPING.get(ref_header)
        if target_header and target_header != "None" and target_header in unknown_set:
            final_mapping[ref_header] = target_header
            print(f"  ✅ Mapped standard '{ref_header}' to found '{target_header}'")
        else:
            final_mapping[ref_header] = None
    return final_mapping

def load_and_combine_las_from_zip(zip_bytes):
    print("\n--- Loading and Combining LAS files from ZIP bytes ---")
    las_dfs = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
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
    except Exception as e:
        print(f"  ❌ CRITICAL ERROR: Could not process ZIP file. Error: {e}")
        return None
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

def find_and_prepare_ml_features(df):
    print("\n--- Preparing Features for Machine Learning ---")
    cols_to_exclude = ['DEPTH', 'WELL', 'CLASSIFICATION']
    potential_features = [col for col in df.columns if col not in cols_to_exclude]
    ml_df_base = df[potential_features].select_dtypes(include=np.number).copy()
    valid_feature_cols = []
    for col in ml_df_base.columns:
        if ml_df_base[col].nunique() > 1:
            valid_feature_cols.append(col)
    if len(valid_feature_cols) < 2:
        return None
    ml_df_valid = ml_df_base[valid_feature_cols].copy()
    ml_df_valid.interpolate(method='linear', inplace=True, limit_direction='both')
    ml_df_valid.bfill(inplace=True)
    ml_df_valid.ffill(inplace=True)
    ml_df_valid.dropna(how='any', inplace=True)
    if ml_df_valid.empty:
        return None
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(ml_df_valid)
    return pd.DataFrame(x_scaled, columns=ml_df_valid.columns, index=ml_df_valid.index)

# --- PLOTTING FUNCTIONS ---
def plot_well_log(df, well_name):
    print(f"\n--- Generating Professional Well Log Plot for well: '{well_name}' ---")
    df_well = df[df['WELL'] == well_name].copy()
    tracks_to_plot = LOG_CONFIG + [{'mnemonic': 'CLASSIFICATION'}]
    fig, axs = plt.subplots(nrows=1, ncols=len(tracks_to_plot), figsize=(len(tracks_to_plot) * 1.5, 15), sharey=True,
                            gridspec_kw={'width_ratios': [1]*len(LOG_CONFIG) + [0.5]})
    fig.suptitle(f"Well Log: {well_name}", fontsize=18, y=1.02)
    min_depth, max_depth = df_well['DEPTH'].min(), df_well['DEPTH'].max()
    ax1 = axs[0]
    ax1.set_ylabel("Depth (m)", fontsize=12, fontweight='bold')
    ax1.set_ylim(max_depth, min_depth)
    ax1.yaxis.set_major_locator(MultipleLocator(250))
    ax1.yaxis.set_minor_locator(MultipleLocator(50))
    ax1.tick_params(axis='y', which='major', labelsize=10, labelcolor='black', width=1.5, length=10)
    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')
    ax1.spines['left'].set_linewidth(1.5)
    color_index = 0
    for i, track_info in enumerate(tracks_to_plot):
        ax = axs[i]
        mnemonic = track_info['mnemonic']
        ax.set_box_aspect(25)
        ax.grid(which='major', color='lightgray', linestyle='-')
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.tick_params(axis='x', labelsize=8)
        if mnemonic not in df_well.columns or df_well[mnemonic].dropna().empty:
            ax.set_xlabel(mnemonic, color='lightgray', fontsize=10, fontweight='bold')
            ax.set_xticks([])
            continue
        if mnemonic == 'CLASSIFICATION':
            ax.set_xlabel(mnemonic, color='black', fontsize=10, fontweight='bold')
            facies_labels = sorted(df_well[mnemonic].dropna().unique())
            num_facies = len(facies_labels)
            cmap = plt.colormaps.get_cmap('viridis')
            colors = [cmap(i / num_facies) for i in range(num_facies)]
            facies_color_map = dict(zip(facies_labels, colors))
            for facies_val, color in facies_color_map.items():
                ax.fill_betweenx(df_well['DEPTH'], 0, 1, where=(df_well[mnemonic] == facies_val), facecolor=color)
            ax.set_xlim(0, 1)
            ax.set_xticks([])
        else:
            color = LOG_PLOT_COLORS[color_index % len(LOG_PLOT_COLORS)]
            min_val, max_val = track_info['range']
            title_str = f"{mnemonic}\n({min_val:g} - {max_val:g})"
            ax.set_xlabel(title_str, color=color, fontsize=9, fontweight='bold', ha='center')
            ax.plot(df_well[mnemonic], df_well['DEPTH'], color=color, linewidth=0.7)
            ax.set_xlim(track_info['range'])
            if track_info.get('log_scale', False):
                ax.set_xscale('log')
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            color_index += 1
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# THIS FUNCTION IS MODIFIED
def plot_elbow_method(X, title_name):
    print(f"\n--- Generating Elbow Method Plot for: {title_name} ---")
    ks = range(1, 10)
    inertias = []
    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        inertias.append(model.inertia_)
    fig = plt.figure(figsize=(10, 6))
    plt.plot(ks, inertias, '-o', color='b', markerfacecolor='red', markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=14)
    plt.ylabel('Inertia (Sum of Squared Distances)', fontsize=14)
    # DYNAMIC TITLE
    plt.title(f'Elbow Method for {title_name}', fontsize=16, fontweight='bold')
    plt.xticks(ks)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig

# THIS FUNCTION IS MODIFIED
def plot_tsne_visualization(X, y_labels, title_name):
    print(f"\n--- Generating t-SNE Visualization for: {title_name} ---")
    perplexity = min(30, len(X) - 1) if len(X) > 1 else 1
    model = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=perplexity, random_state=42)
    tsne_features = model.fit_transform(X)
    fig = plt.figure(figsize=(12, 8))
    scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=y_labels, cmap='viridis', alpha=0.7)
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    # DYNAMIC TITLE
    plt.title(f't-SNE Visualization for {title_name}', fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    if len(np.unique(y_labels)) > 0:
        plt.legend(handles=scatter.legend_elements()[0], labels=[f'Cluster {i}' for i in np.unique(y_labels)])
    plt.tight_layout()
    return fig

# --- FastAPI Web Server ---
app = FastAPI(
    title="Automated Well Log Plotting API",
    description="Upload a ZIP file containing LAS files to generate professional plots."
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Well Log Plotting API. Go to /docs to use the tool."}

@app.post("/generate-plots/{plot_type}", 
         responses={
             200: {"content": {"application/zip": {}}, "description": "Successfully generated a ZIP of plot images."},
             400: {"description": "Could not process the uploaded file."},
         })
async def generate_plots(
    plot_type: Literal['well-log', 'elbow', 'tsne'], 
    file: UploadFile = File(..., description="A ZIP file containing .las well logs.")
):
    zip_bytes = await file.read()
    master_df = load_and_combine_las_from_zip(zip_bytes)
    if master_df is None:
        raise HTTPException(status_code=400, detail="Could not read or process valid LAS files from the ZIP.")

    unique_wells = master_df['WELL'].unique().tolist()
    scopes_to_process = unique_wells if plot_type == 'well-log' else [None] + unique_wells
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        for scope in scopes_to_process:
            scope_name = "all_wells" if scope is None else scope
            df_original = master_df if scope is None else master_df[master_df['WELL'] == scope].copy()
            if df_original.empty: continue

            reference_headers = [log['mnemonic'] for log in LOG_CONFIG]
            header_map = map_las_headers(reference=reference_headers, unknown=df_original.columns.tolist())
            
            df_standardized = df_original[['DEPTH', 'WELL']].copy()
            for std_name, actual_name in header_map.items():
                if actual_name:
                    df_standardized[std_name] = df_original[actual_name]

            safe_scope_name = "".join(c for c in scope_name if c.isalnum() or c in (' ', '.')).rstrip().replace(' ', '_')
            
            fig = None
            if plot_type == 'well-log':
                scaled_X = find_and_prepare_ml_features(df_standardized)
                if scaled_X is not None:
                    kmeans = KMeans(n_clusters=7, random_state=42, n_init=10).fit(scaled_X)
                    labels_df = pd.DataFrame(kmeans.labels_, index=scaled_X.index, columns=['CLASSIFICATION'])
                    df_with_classification = df_standardized.join(labels_df)
                    df_with_classification['CLASSIFICATION'].ffill(inplace=True)
                    df_with_classification['CLASSIFICATION'].bfill(inplace=True)
                    fig = plot_well_log(df_with_classification, scope_name)
                else:
                    fig = plot_well_log(df_standardized, scope_name)
                filename = f"well_log_{safe_scope_name}.png"
            else:
                scaled_X = find_and_prepare_ml_features(df_standardized)
                if scaled_X is None:
                    print(f"Skipping ML plot for {scope_name} as features could not be prepared.")
                    continue
                if plot_type == 'elbow':
                    # PASS THE DYNAMIC TITLE
                    fig = plot_elbow_method(scaled_X, scope_name)
                    filename = f"elbow_method_{safe_scope_name}.png"
                elif plot_type == 'tsne':
                    kmeans = KMeans(n_clusters=7, random_state=42, n_init=10).fit(scaled_X)
                    # PASS THE DYNAMIC TITLE
                    fig = plot_tsne_visualization(scaled_X, kmeans.labels_, scope_name)
                    filename = f"tsne_by_kmeans_{safe_scope_name}.png"

            if fig:
                img_buffer = io.BytesIO()
                fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                plt.close(fig)
                zip_file.writestr(filename, img_buffer.getvalue())

    zip_buffer.seek(0)
    return Response(content=zip_buffer.getvalue(), media_type="application/zip", headers={"Content-Disposition": f"attachment; filename=plots_{plot_type}.zip"})
