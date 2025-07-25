# dataset_plot_generation.py

import os
import pandas as pd
import lasio
import json
import sys
import matplotlib.pyplot as plt
import io
import zipfile
import uuid
from typing import List, Dict, Any
from urllib.parse import quote

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

# --- Create the FastAPI app instance ---
app = FastAPI(
    title="LAS File Processing API",
    description="A two-step API to process LAS files and stream correlation plots.",
    version="3.1.0" # Final, truly working version
)

# --- In-Memory Cache ---
PROCESSED_DATA_CACHE: Dict[str, Any] = {}


# --- Helper functions (Unchanged) ---

def find_column(df_columns: List[str], aliases: List[str]) -> str | None:
    for alias in aliases:
        if alias in df_columns:
            return alias
    return None

def _process_las_in_memory(zip_contents: bytes) -> pd.DataFrame | None:
    las_dfs = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_contents), 'r') as zf:
            for entry in zf.namelist():
                if not entry.lower().endswith('.las') or entry.startswith('__MACOSX'):
                    continue
                
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
                las_dfs.append(df)

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred parsing LAS files: {e}")

    if not las_dfs: return None
    return pd.concat(las_dfs, ignore_index=True)

def _plot_well_pair_to_stream(df: pd.DataFrame, w1: str, w2: str, curve_to_plot: str, curve_label: str) -> io.BytesIO | None:
    df1 = df[df['WELL'] == w1]
    df2 = df[df['WELL'] == w2]
    if df1.empty or df2.empty: return None

    fig, ax = plt.subplots(figsize=(4, 6), dpi=150)
    
    if curve_to_plot in df1.columns and 'DEPTH_MD' in df1.columns:
        ax.plot(df1[curve_to_plot], df1['DEPTH_MD'], label=w1, color='blue')
        ax.plot(df2[curve_to_plot], df2['DEPTH_MD'], label=w2, color='red')
        ax.set_xlabel(curve_label)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Data missing for plotting.", ha='center', va='center')

    ax.invert_yaxis(); ax.set_ylabel('Depth (MD)'); ax.set_title("Well Correlation")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight')
    plt.close(fig)
    img_buffer.seek(0)
    
    return img_buffer


# --- API Endpoints ---

@app.post("/process_las", summary="Step 1: Process ZIP and get plot URLs")
async def process_las_files(file: UploadFile = File(..., description="A ZIP file containing .las files.")):
    df = _process_las_in_memory(await file.read())
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No valid LAS files found in the ZIP.")

    df.columns = [col.upper() for col in df.columns]
    
    DEPTH_ALIASES = ['DEPT', 'DEPTH', 'MD']
    actual_depth_col = find_column(df.columns, DEPTH_ALIASES)
    if not actual_depth_col: raise HTTPException(status_code=400, detail="No valid Depth column found.")
    df.rename(columns={actual_depth_col: 'DEPTH_MD'}, inplace=True)
        
    GR_ALIASES = ['GR', 'GRC', 'CGR', 'SGR', 'GAMMA']
    SP_ALIASES = ['SP', 'SPN']
    curve_to_plot = 'CORRELATION_CURVE'; curve_label = ''
    
    actual_corr_col = find_column(df.columns, GR_ALIASES)
    if actual_corr_col:
        curve_label = f'Gamma Ray ({actual_corr_col})'
        df.rename(columns={actual_corr_col: curve_to_plot}, inplace=True)
    else:
        actual_corr_col = find_column(df.columns, SP_ALIASES)
        if actual_corr_col:
            curve_label = f'Spontaneous Potential ({actual_corr_col})'
            df.rename(columns={actual_corr_col: curve_to_plot}, inplace=True)
        else: raise HTTPException(status_code=400, detail="No suitable correlation curve found (GR or SP).")
    
    job_id = str(uuid.uuid4())
    PROCESSED_DATA_CACHE[job_id] = {
        "dataframe": df,
        "curve_to_plot": curve_to_plot,
        "curve_label": curve_label
    }
    
    wells = sorted(df['WELL'].astype(str).unique())
    plot_urls = []
    if len(wells) >= 2:
        well_pairs = [(wells[i], wells[i+1]) for i in range(len(wells)-1)]
        for w1, w2 in well_pairs:
            # --- THE FINAL BUG FIX ---
            # The safe='' argument forces quote() to also encode the '/' character.
            safe_w1 = quote(w1, safe='')
            safe_w2 = quote(w2, safe='')
            # --- END OF FIX ---
            plot_urls.append(f"/plot?job_id={job_id}&well1={safe_w1}&well2={safe_w2}")
    
    return JSONResponse(content={
        "status": "success",
        "message": f"Processed {len(wells)} wells. Use the URLs below to view plots.",
        "job_id": job_id,
        "available_wells": wells,
        "plot_urls": plot_urls
    })

@app.get("/plot", summary="Step 2: Get a single plot image using query parameters")
async def get_plot(
    job_id: str = Query(..., description="The job_id returned from the /process_las endpoint."),
    well1: str = Query(..., description="The URL-encoded name of the first well."),
    well2: str = Query(..., description="The URL-encoded name of the second well.")
):
    job_data = PROCESSED_DATA_CACHE.get(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail="Job ID not found or expired. Please process a file first.")

    image_buffer = _plot_well_pair_to_stream(
        df=job_data["dataframe"],
        w1=well1, w2=well2,
        curve_to_plot=job_data["curve_to_plot"],
        curve_label=job_data["curve_label"]
    )
    
    if not image_buffer:
        raise HTTPException(status_code=404, detail="Could not generate plot for the specified wells.")
    
    return StreamingResponse(image_buffer, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run("dataset_plot_generation:app", host="0.0.0.0", port=8000, reload=True)