# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from typing import Literal
import io
import zipfile
from utils.data_processing import load_and_combine_las_from_zip, map_las_headers, find_and_prepare_ml_features
from utils.plotting import plot_well_log, plot_elbow_method, plot_tsne_visualization
from sklearn.cluster import KMeans
import pandas as pd

app = FastAPI(
    title="Automated Well Log Plotting API",
    description="Upload a ZIP file containing LAS files to generate professional plots."
)

class PlotRequest(BaseModel):
    client_mapping: Dict[str, str]

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
    file: UploadFile = File(..., description="A ZIP file containing .las well logs."),
    body: PlotRequest = Body(...)
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

            from utils.config import LOG_CONFIG
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
                    fig = plot_elbow_method(scaled_X, scope_name)
                    filename = f"elbow_method_{safe_scope_name}.png"
                elif plot_type == 'tsne':
                    kmeans = KMeans(n_clusters=7, random_state=42, n_init=10).fit(scaled_X)
                    fig = plot_tsne_visualization(scaled_X, kmeans.labels_, scope_name)
                    filename = f"tsne_by_kmeans_{safe_scope_name}.png"

            if fig:
                img_buffer = io.BytesIO()
                fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                import matplotlib.pyplot as plt
                plt.close(fig)
                zip_file.writestr(filename, img_buffer.getvalue())

    zip_buffer.seek(0)
    return Response(content=zip_buffer.getvalue(), media_type="application/zip", headers={"Content-Disposition": f"attachment; filename=plots_{plot_type}.zip"})
