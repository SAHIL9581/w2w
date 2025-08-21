# main.py
import io
import json # Import the json library
import zipfile
from typing import Dict, List, Literal

import matplotlib.pyplot as plt
import pandas as pd
from fastapi import (FastAPI, File, Form, HTTPException, Response, # Import Form
                     UploadFile)
from pydantic import BaseModel, Field, ValidationError
from sklearn.cluster import KMeans

# Import utility functions
from utils.data_processing import (find_and_prepare_ml_features,
                                   get_las_headers_from_zip,
                                   load_and_combine_las_from_zip,
                                   map_las_headers)
from utils.generate_all_plots import (plot_elbow_method,
                                      plot_tsne_visualization,
                                      plot_well_log)

app = FastAPI(
    title="Automated Well Log Plotting API",
    description="An API to process well log LAS files and generate plots using a predefined configuration."
)

class CustomPlotRequest(BaseModel):
    client_mapping: Dict[str, str] = Field(
        ...,
        example={
            "GR": "GAMMA", "RDEP": "RES_DEEP", "RHOB": "DENSITY", "SP": "SP_VAL"
        }
    )

@app.get("/")
def read_root():
    return {"message": "Welcome to the Well Log Plotting API. Go to /docs to see the endpoints."}


@app.post("/get-headers/", response_model=Dict[str, List[str]])
async def get_headers(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .zip file.")

    zip_bytes = await file.read()
    try:
        headers_to_exclude = {
            "DEPTH_MD", "FORCE_2020_LITHOFACIES_CONFIDENCE", "FORCE_2020_LITHOFACIES_LITHOLOGY",
            "X_LOC", "Y_LOC", "Z_LOC"
        }
        all_headers = get_las_headers_from_zip(zip_bytes)
        filtered_headers = [h for h in all_headers if h not in headers_to_exclude]
        if not filtered_headers:
            raise HTTPException(status_code=404, detail="No displayable LAS file headers found after filtering.")
        return {"headers": filtered_headers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the file: {e}")


@app.post("/generate-plots/{plot_type}", summary="Generate Plots with Automatic Mapping")
async def generate_plots_auto(
    plot_type: Literal['well-log', 'elbow', 'tsne'],
    file: UploadFile = File(..., description="A ZIP file containing .las well logs."),
):
    """
    Generates plots using the **automatic smart mapping**.
    It first looks for direct curve matches (e.g., GR -> GR) and then uses `config.py` as a fallback.
    """
    zip_bytes = await file.read()
    master_df = load_and_combine_las_from_zip(zip_bytes)
    zip_buffer = await process_and_plot_data(master_df, plot_type)
    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=plots_{plot_type}_auto.zip"}
    )


@app.post("/generate-plots-with-mapping/{plot_type}", summary="Generate Plots with Custom Mapping")
async def generate_plots_custom(
    plot_type: Literal['well-log', 'elbow', 'tsne'],
    # --- THIS IS THE FIX ---
    # We now explicitly tell FastAPI to expect the 'body' as a string from the form data.
    body: str = Form(..., description="A JSON string of the client mapping."),
    # --- END OF FIX ---
    file: UploadFile = File(..., description="A ZIP file containing .las well logs."),
):
    """
    Generates plots using a **custom mapping provided in the request body**.
    This gives you full control over which curves are plotted.
    """
    # --- AND THIS IS THE PARSING LOGIC ---
    try:
        # Manually parse the JSON string from the form data into our Pydantic model
        body_data = CustomPlotRequest.model_validate_json(body)
    except (ValidationError, json.JSONDecodeError) as e:
        # If parsing or validation fails, return a helpful 400 error
        raise HTTPException(status_code=400, detail=f"Invalid JSON format in 'body' field: {e}")
    # --- END OF PARSING LOGIC ---
    
    zip_bytes = await file.read()
    master_df = load_and_combine_las_from_zip(zip_bytes)
    # Now we pass the validated mapping dictionary to our processing function
    zip_buffer = await process_and_plot_data(master_df, plot_type, custom_mapping=body_data.client_mapping)
    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=plots_{plot_type}_custom.zip"}
    )


async def process_and_plot_data(master_df: pd.DataFrame, plot_type: str, custom_mapping: dict = None) -> io.BytesIO:
    if master_df is None:
        raise HTTPException(status_code=404, detail="Could not read or process valid LAS files from the ZIP.")

    unique_wells = master_df['WELL'].unique().tolist()
    scopes_to_process = unique_wells if plot_type == 'well-log' else [None] + unique_wells

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for scope in scopes_to_process:
            scope_name = "all_wells_combined" if scope is None else scope
            df_original = master_df if scope is None else master_df[master_df['WELL'] == scope].copy()
            if df_original.empty: continue
            
            header_map = map_las_headers(unknown=df_original.columns.tolist(), custom_mapping=custom_mapping)
            df_standardized = df_original[['DEPTH', 'WELL']].copy()
            for std_name, actual_name in header_map.items():
                if actual_name and actual_name in df_original.columns:
                    df_standardized[std_name] = df_original[actual_name]
            
            safe_scope_name = "".join(c for c in scope_name if c.isalnum() or c in (' ', '.')).rstrip().replace(' ', '_')
            fig = None
            scaled_X = find_and_prepare_ml_features(df_standardized)

            if plot_type == 'well-log':
                if scaled_X is not None and not scaled_X.empty:
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
                if scaled_X is None or scaled_X.empty: continue
                if plot_type == 'elbow':
                    fig = plot_elbow_method(scaled_X, scope_name)
                    filename = f"elbow_method_{safe_scope_name}.png"
                elif plot_type == 'tsne':
                    if scaled_X.shape[1] < 2: continue
                    kmeans = KMeans(n_clusters=7, random_state=42, n_init=10).fit(scaled_X)
                    fig = plot_tsne_visualization(scaled_X, kmeans.labels_, scope_name)
                    filename = f"tsne_by_kmeans_{safe_scope_name}.png"
            
            if fig:
                img_buffer = io.BytesIO()
                fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                plt.close(fig)
                zip_file.writestr(filename, img_buffer.getvalue())
    
    zip_buffer.seek(0)
    return zip_buffer