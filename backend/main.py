import json
import io
import os
import hashlib
import shutil
from collections import defaultdict
from typing import Dict, List, Literal, Optional
from pathlib import Path
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fastapi import (
    FastAPI, File, Form, HTTPException, Response, UploadFile
)
from pydantic import BaseModel, ValidationError, Field
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import lasio
from fastapi import Header, Request, Response
from chat.gemini_client import call_gemini_api
from utils.data_processing import (
    find_and_prepare_ml_features,
    load_data_from_directory,
    map_las_headers,
    unzip_and_save_las_files
)
from utils.generate_all_plots import (
    plot_elbow_method,
    plot_tsne_visualization,
    plot_well_log
)
from reporting.pdf_generator import create_well_report_pdf
from reporting.report_prompt import REPORT_SUMMARY_PROMPT

from chat.prompt import SYSTEM_PROMPT as CHAT_SYSTEM_PROMPT

# ================== Clustering API (Port 8000) ==================

app = FastAPI(
    title="Well Log Clustering & Analysis API",
    description="API for traditional well log clustering analysis, plotting, and AI-powered insights."
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Metadata-SHA256"]
)

# Global variables
UPLOAD_DIR = Path("/app/uploaded_data")
OUTPUT_DIR = Path("/app/output")
sha_call_counts = defaultdict(int)

class CustomPlotRequest(BaseModel):
    client_mapping: Dict[str, str]

class ChatRequest(BaseModel):
    question: str
    sha: str

class ReportRequest(BaseModel):
    well_names: List[str]

@app.on_event("startup")
def on_startup():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"🚀 Well Log Clustering API starting on port 8000")
    print(f"📁 Upload directory: {UPLOAD_DIR}")
    print(f"📁 Output directory: {OUTPUT_DIR}")

def cleanup_directories():
    """Clean up old upload and output data to prevent conflicts."""
    try:
        # Clean upload directory
        if UPLOAD_DIR.exists():
            for item in UPLOAD_DIR.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
        
        # Clean output directory
        if OUTPUT_DIR.exists():
            for item in OUTPUT_DIR.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
                    
        print("🧹 Cleaned up old data directories")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Well Log Clustering & Analysis API (Port 8000). Use /docs to see the endpoints."}

@app.post("/upload-data-and-get-info", tags=["Step 1: Upload Data"], summary="Upload ZIP Files and Get Info")
async def upload_data_and_get_info(file: UploadFile = File(...)):
    """Upload ZIP file containing LAS files for traditional clustering analysis."""
    if not file.filename.lower().endswith('.zip'): 
        raise HTTPException(status_code=400, detail="Invalid file type. Only ZIP files accepted.")
    
    # 🧹 CLEAN OLD DATA BEFORE PROCESSING NEW UPLOAD
    cleanup_directories()
    
    zip_bytes = await file.read()
    headers, wells = unzip_and_save_las_files(zip_bytes, UPLOAD_DIR)
    if not wells: 
        raise HTTPException(status_code=400, detail="No valid .las files found in ZIP.")
    
    # Generate unique SHA for this session
    random_data = os.urandom(64)
    full_sha256 = hashlib.sha256(random_data).hexdigest()
    short_sha = full_sha256[:8]
    
    # Initialize rate limit counter for this SHA
    sha_call_counts[short_sha] = 0
    
    print(f"🆔 New session created with SHA: {short_sha}")
    print(f"📊 Initialized rate limit: {sha_call_counts[short_sha]}/5 calls")
    
    response_headers = {
        "X-Metadata-SHA256": short_sha
    }
    
    return Response(
        content=json.dumps({
            "message": "ZIP file uploaded successfully.", 
            "file_type": "zip",
            "headers": headers, 
            "well_names": wells,
            "sha": short_sha,
            "file_count": len(wells),
            "chat_calls_remaining": 5 - sha_call_counts[short_sha]
        }),
        media_type="application/json",
        headers=response_headers
    )

@app.post("/generate-all-plots", tags=["Step 2: Generate Plots"], summary="Generate and Save ALL Plot Types for ALL Wells")
async def generate_all_plots(client_mapping: Optional[str] = Form(None)):
    try:
        mapping_dict = None
        if client_mapping:
            try:
                mapping_dict = json.loads(client_mapping)
                CustomPlotRequest(client_mapping=mapping_dict)
                print(f"🗺️ Client mapping provided: {mapping_dict}")
            except Exception as e: 
                raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
        
        # 🔍 CHECK FOR UPLOADED DATA
        if not UPLOAD_DIR.exists() or not any(UPLOAD_DIR.iterdir()): 
            raise HTTPException(status_code=404, detail="No data on server. Upload a ZIP first.")
        
        print(f"📂 Loading data from directory: {UPLOAD_DIR}")
        
        try:
            master_df = load_data_from_directory(UPLOAD_DIR)
            print(f"✅ Standard loading successful")
        except Exception as load_error:
            print(f"❌ Standard load failed: {load_error}")
            print("🔄 Attempting manual LAS loading for limited dataset...")
            
            # 🔧 MANUAL LAS LOADING FALLBACK FOR LIMITED DATASETS
            try:
                all_well_data = []
                
                # Find all LAS files
                las_files = []
                for pattern in ['*.las', '*.LAS']:
                    las_files.extend(list(UPLOAD_DIR.glob(pattern)))
                for pattern in ['**/*.las', '**/*.LAS']:
                    las_files.extend(list(UPLOAD_DIR.glob(pattern)))
                las_files = list(set(las_files))
                
                if not las_files:
                    raise HTTPException(status_code=400, detail="No LAS files found in upload directory")
                
                print(f"📁 Found {len(las_files)} LAS files: {[f.name for f in las_files]}")
                
                for las_file in las_files:
                    try:
                        print(f"📖 Reading LAS file: {las_file.name}")
                        las = lasio.read(str(las_file))
                        df = las.df().reset_index()
                        
                        # Get well name from file or LAS header
                        well_name = las_file.stem
                        try:
                            if hasattr(las.well, 'WELL') and las.well.WELL.value:
                                well_name = str(las.well.WELL.value)
                        except:
                            pass
                        
                        # Add WELL column
                        df['WELL'] = well_name
                        
                        print(f"   📊 Well: {well_name}, Shape: {df.shape}, Columns: {list(df.columns)}")
                        all_well_data.append(df)
                        
                    except Exception as file_error:
                        print(f"   ❌ Failed to read {las_file.name}: {file_error}")
                        continue
                
                if not all_well_data:
                    raise HTTPException(status_code=400, detail="Could not read any LAS files")
                
                # Combine all wells
                master_df = pd.concat(all_well_data, ignore_index=True, sort=False)
                print(f"✅ Manual loading successful: {len(master_df)} rows, {len(master_df.columns)} columns")
                
            except Exception as manual_error:
                print(f"❌ Manual loading also failed: {manual_error}")
                raise HTTPException(status_code=500, detail=f"Could not load LAS data: {str(manual_error)}")
        
        if master_df is None or master_df.empty: 
            raise HTTPException(status_code=400, detail="No valid data found after loading")
        
        print(f"📊 Final dataset: {len(master_df)} rows and {len(master_df.columns)} columns")
        
        # 🔧 GET ACTUAL WELL NAMES FROM THE DATA
        if 'WELL' not in master_df.columns:
            raise HTTPException(status_code=400, detail="No WELL column found in data. Cannot identify individual wells.")
        
        actual_well_names = master_df['WELL'].unique().tolist()
        print(f"🏷️ Actual wells in dataset: {actual_well_names}")
        print(f"📋 Available columns: {list(master_df.columns)}")
        
        wells_processed = []
        wells_with_details = []
        processing_errors = []
        
        for well_name in actual_well_names:
            try:
                print(f"🔄 Processing well: {well_name}")
                
                # 🔧 SAFE FOLDER NAME GENERATION
                safe_folder_name = str(well_name).replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")
                well_output_dir = OUTPUT_DIR / safe_folder_name
                well_output_dir.mkdir(exist_ok=True)
                
                df_well = master_df[master_df['WELL'] == well_name].copy()
                
                if df_well.empty:
                    print(f"⚠️ No data found for well: {well_name}")
                    continue
                
                print(f"   📋 Well data shape: {df_well.shape}")
                print(f"   📋 Well columns: {list(df_well.columns)}")
                
                # 🔧 UNIVERSAL DEPTH HANDLING - Try all possible depth column names
                depth_column = None
                depth_alternatives = ['DEPTH', 'DEPT', 'MD', 'DEPTH_MD', 'TVD', 'MEASURED_DEPTH']
                for alt in depth_alternatives:
                    if alt in df_well.columns:
                        depth_column = alt
                        if alt != 'DEPTH':
                            try:
                                df_well['DEPTH'] = pd.to_numeric(df_well[alt], errors='coerce')
                            except Exception as depth_error:
                                print(f"   ⚠️ Error converting {alt} to numeric: {depth_error}")
                                continue
                        print(f"   ✅ Using {alt} as DEPTH column")
                        break
                
                if not depth_column:
                    print(f"   ❌ No depth column found for {well_name}")
                    processing_errors.append(f"No depth column found for {well_name}")
                    continue
                
                # 🔧 UNIVERSAL FEATURE EXTRACTION - Use ALL available numeric columns
                try:
                    # Get all column names and try to convert each to numeric
                    all_columns = df_well.columns.tolist()
                    numeric_columns = []
                    
                    for col in all_columns:
                        if col in ['WELL']:  # Skip non-numeric identifier columns
                            continue
                        try:
                            # Try to convert to numeric
                            pd.to_numeric(df_well[col], errors='raise')
                            numeric_columns.append(col)
                        except:
                            print(f"   📝 Column {col} is not numeric, skipping")
                    
                    print(f"   🔢 Found numeric columns: {numeric_columns}")
                except Exception as numeric_error:
                    print(f"   ❌ Error detecting numeric columns: {numeric_error}")
                    continue
                
                # Remove depth and well identifier columns from features
                exclude_cols = ['DEPTH', 'DEPT', 'MD', 'DEPTH_MD', 'TVD', 'MEASURED_DEPTH', 'WELL']
                feature_columns = [col for col in numeric_columns if col not in exclude_cols]
                
                if len(feature_columns) == 0:
                    print(f"   ❌ No numeric feature columns found for {well_name}")
                    print(f"   📋 Available columns were: {list(df_well.columns)}")
                    print(f"   📋 Detected numeric columns were: {numeric_columns}")
                    processing_errors.append(f"No numeric feature columns for {well_name}")
                    continue
                
                print(f"   🎯 Using {len(feature_columns)} feature columns: {feature_columns}")
                
                # 🔧 CREATE CLEAN DATASET WITH DEPTH, WELL, AND ALL FEATURES
                try:
                    # Start with essential columns
                    df_clean = pd.DataFrame()
                    df_clean['DEPTH'] = pd.to_numeric(df_well['DEPTH'], errors='coerce')
                    df_clean['WELL'] = df_well['WELL']
                    
                    # Add all feature columns with proper handling
                    for col in feature_columns:
                        try:
                            # Convert to numeric and fill NaN with 0
                            df_clean[col] = pd.to_numeric(df_well[col], errors='coerce').fillna(0)
                        except Exception as col_error:
                            print(f"   ⚠️ Issue with column {col}: {col_error}")
                            df_clean[col] = 0  # Fill with zeros if conversion fails
                    
                    # Remove rows where DEPTH is NaN
                    df_clean = df_clean.dropna(subset=['DEPTH'])
                    
                    print(f"   📊 Clean dataset shape: {df_clean.shape}")
                    print(f"   📊 Features for ML: {feature_columns}")
                    
                    if len(df_clean) == 0:
                        print(f"   ❌ No valid data after cleaning for {well_name}")
                        processing_errors.append(f"No valid data after cleaning for {well_name}")
                        continue
                    
                except Exception as clean_error:
                    print(f"   ❌ Error creating clean dataset: {clean_error}")
                    processing_errors.append(f"Data cleaning failed for {well_name}")
                    continue
                
                # 🔧 UNIVERSAL ML PREPARATION - Works with ANY number of features
                try:
                    # Extract feature data (exclude DEPTH and WELL)
                    feature_data = df_clean[feature_columns].copy()
                    
                    # Remove rows where all features are zero or NaN
                    valid_rows = (feature_data != 0).any(axis=1) & feature_data.notna().any(axis=1)
                    if not valid_rows.any():
                        print(f"   ⚠️ All feature data is zero or NaN, using all rows")
                        valid_rows = pd.Series([True] * len(feature_data), index=feature_data.index)
                    
                    feature_data_clean = feature_data[valid_rows].fillna(0)
                    df_clean_final = df_clean[valid_rows].copy()
                    
                    if len(feature_data_clean) < 2:
                        print(f"   ❌ Insufficient valid data points ({len(feature_data_clean)}) for {well_name}")
                        processing_errors.append(f"Insufficient valid data points for {well_name}")
                        continue
                    
                    print(f"   📏 Valid data points: {len(feature_data_clean)}")
                    
                    # Scale the features using StandardScaler
                    scaler = StandardScaler()
                    scaled_features = scaler.fit_transform(feature_data_clean)
                    
                    # Create scaled DataFrame
                    scaled_X = pd.DataFrame(
                        scaled_features,
                        index=feature_data_clean.index,
                        columns=feature_columns
                    )
                    
                    print(f"   🔢 Scaled features shape: {scaled_X.shape}")
                    
                except Exception as scaling_error:
                    print(f"   ❌ Feature scaling failed: {scaling_error}")
                    processing_errors.append(f"Feature scaling failed for {well_name}: {scaling_error}")
                    continue
                
                # 🔧 UNIVERSAL CLUSTERING - Adaptive to any number of features and samples
                n_samples = len(scaled_X)
                n_features = len(scaled_X.columns)
                
                # Smart cluster count calculation
                if n_samples < 10:
                    n_clusters = min(2, n_samples - 1) if n_samples > 1 else 2
                else:
                    n_clusters = min(max(2, min(n_samples // 5, n_features + 1)), 7)
                
                print(f"   🎯 Using {n_clusters} clusters for {n_samples} samples, {n_features} features")
                
                try:
                    # Adaptive KMeans parameters
                    if n_samples < 50:
                        kmeans = KMeans(
                            n_clusters=n_clusters, 
                            random_state=42, 
                            n_init=3, 
                            max_iter=100,
                            tol=1e-3
                        )
                    else:
                        kmeans = KMeans(
                            n_clusters=n_clusters, 
                            random_state=42, 
                            n_init=10
                        )
                    
                    kmeans_labels = kmeans.fit_predict(scaled_X)
                    
                    # Add classification to the final clean data
                    df_classified = df_clean_final.copy()
                    df_classified['CLASSIFICATION'] = kmeans_labels
                    df_classified['CLASSIFICATION'] = df_classified['CLASSIFICATION'].astype(int)
                    
                    # Fill any gaps in classification
                    df_classified['CLASSIFICATION'].ffill(inplace=True)
                    df_classified['CLASSIFICATION'].bfill(inplace=True)
                    
                    unique_clusters = len(set(kmeans_labels))
                    print(f"   ✅ Clustering completed with {unique_clusters} unique clusters")
                    
                except Exception as cluster_error:
                    print(f"   ❌ Clustering failed: {cluster_error}")
                    processing_errors.append(f"Clustering failed for {well_name}: {cluster_error}")
                    continue
                
                # 🔧 UNIVERSAL PLOT GENERATION - Works with any feature set
                plots_generated = 0
                plot_types_generated = []
                
                for plot_type_str in ["cluster", "elbow"]:
                    try:
                        fig = None
                        
                        if plot_type_str == "cluster": 
                            fig = plot_well_log(df_classified, well_name)
                        elif plot_type_str == "elbow": 
                            fig = plot_elbow_method(scaled_X, well_name)
                        
                        if fig:
                            filename = f"well_log_cluster.png" if plot_type_str == "cluster" else f"{plot_type_str}.png"
                            fig.savefig(well_output_dir / filename, dpi=300, bbox_inches='tight')
                            plt.close(fig)
                            plots_generated += 1
                            plot_types_generated.append(plot_type_str)
                            print(f"   ✅ Generated {plot_type_str} plot")
                        
                    except Exception as plot_error:
                        print(f"   ⚠️ Failed to generate {plot_type_str} plot: {plot_error}")
                        # Continue with other plots even if one fails
                
                # Record successful processing
                wells_processed.append(well_name)
                wells_with_details.append({
                    "well_name": well_name,
                    "safe_folder_name": safe_folder_name,
                    "data_points": n_samples,
                    "features_used": feature_columns,
                    "plots_generated": plot_types_generated,
                    "clusters_found": unique_clusters if 'unique_clusters' in locals() else 0
                })
                
                print(f"   ✅ Successfully processed {well_name} ({plots_generated} plots, {n_features} features)")
                
            except Exception as well_error:
                error_msg = f"Error processing well {well_name}: {str(well_error)}"
                print(f"   ❌ {error_msg}")
                processing_errors.append(error_msg)
                continue
        
        # 🎯 FINAL RESULTS
        if not wells_processed:
            error_detail = f"No wells processed successfully. Available wells: {actual_well_names}. Available columns: {list(master_df.columns)}. First 3 errors: {'; '.join(processing_errors[:3])}"
            raise HTTPException(status_code=422, detail=error_detail)
        
        # 🎯 SUCCESS RESPONSE WITH ACTUAL WELL NAMES
        response = {
            "status": "success", 
            "message": f"All plots saved for wells: {wells_processed}",
            "wells_processed": wells_processed,  # ACTUAL well names from uploaded data
            "total_wells": len(wells_processed),
            "dataset_info": {
                "original_well_names": actual_well_names,
                "total_columns": len(master_df.columns),
                "available_features": len(master_df.columns) - 2,  # Exclude WELL and DEPTH
                "sample_columns": list(master_df.columns)
            },
            "processing_details": wells_with_details
        }
        
        if processing_errors:
            response["warnings"] = f"Some issues occurred: {len(processing_errors)} warnings"
            response["warning_details"] = processing_errors[:3]
        
        print(f"✅ COMPLETED: Processed {len(wells_processed)} wells successfully")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error in generate_all_plots: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Plot generation failed: {str(e)}")


@app.post("/get-plot", tags=["Step 3: Retrieve Plots"], summary="Retrieve a SPECIFIC, pre-generated plot for a well")
async def get_plot(
    plot_type: Literal['well-log', 'elbow', 'tsne'],
    well_name: str = Form(...),
):
    filename_map = {
        'well-log': 'well_log_cluster.png',
        'elbow': 'elbow.png',
        'tsne': 'tsne.png'
    }
    filename = filename_map[plot_type]
    
    # Clean well name for safe folder lookup
    safe_folder_name = str(well_name).replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")
    image_path = OUTPUT_DIR / safe_folder_name / filename
    
    if not image_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Plot '{filename}' not found for well '{well_name}'. Available wells: {[d.name for d in OUTPUT_DIR.iterdir() if d.is_dir()]}"
        )
    
    def iterfile():
        with open(image_path, mode="rb") as file_like:
            yield from file_like
    return StreamingResponse(iterfile(), media_type="image/png")

@app.post("/chat", tags=["Step 4: AI Analysis"], summary="Analyze ALL Generated Plots with AI (5 Calls Per SHA)")
async def chat_with_all_plots(request: ChatRequest):
    """
    AI-powered analysis of generated plots with rate limiting.
    Each SHA gets exactly 5 chat calls.
    """
    sha_value = request.sha
    
    # 🔒 RATE LIMIT CHECK - Exactly 5 calls per SHA
    current_calls = sha_call_counts.get(sha_value, 0)
    
    if current_calls >= 5:
        raise HTTPException(
            status_code=429,
            detail=f"🚫 Rate limit exceeded for SHA: {sha_value}. Maximum 5 chat calls allowed per upload session. You have used all {current_calls}/5 calls."
        )
    
    # Check if plots exist
    image_paths = [str(p) for p in OUTPUT_DIR.glob("**/*.png")]
    if not image_paths:
        raise HTTPException(
            status_code=404, 
            detail="❌ No plot images found. Please generate plots first using the /generate-all-plots endpoint."
        )
    
    # 📞 INCREMENT CALL COUNT BEFORE API CALL
    sha_call_counts[sha_value] = current_calls + 1
    new_call_count = sha_call_counts[sha_value]
    calls_remaining = 5 - new_call_count
    
    print(f"🔍 Chat request for SHA: {sha_value}")
    print(f"📊 Call count: {new_call_count}/5 (Remaining: {calls_remaining})")
    print(f"🖼️  Analyzing {len(image_paths)} plot images")
    
    try:
        # 🤖 CALL GEMINI API
        answer = await call_gemini_api(request.question, image_paths, CHAT_SYSTEM_PROMPT)
        
        success_response = {
            "answer": answer,
            "sha": sha_value,
            "call_number": new_call_count,
            "calls_remaining": calls_remaining,
            "rate_limit_status": f"{new_call_count}/5 calls used",
            "images_analyzed": len(image_paths)
        }
        
        if calls_remaining == 0:
            success_response["rate_limit_warning"] = "⚠️ This was your final chat call for this SHA. Upload new data to get 5 more calls."
        
        return success_response
        
    except Exception as e:
        error_response = {
            "error": f"AI analysis failed: {str(e)}",
            "sha": sha_value,
            "call_number": new_call_count,
            "calls_remaining": calls_remaining,
            "note": "Call count was still incremented due to rate limiting policy"
        }
        
        raise HTTPException(status_code=500, detail=error_response)

@app.get("/rate-limit-status", tags=["Helper"], summary="Check Rate Limit Status for All SHAs")
def get_rate_limit_status():
    """
    Check the current rate limit status for all active SHA sessions.
    Shows how many chat calls have been made for each SHA.
    """
    active_sessions = {}
    for sha, calls_made in sha_call_counts.items():
        active_sessions[sha] = {
            "calls_made": calls_made,
            "calls_remaining": max(0, 5 - calls_made),
            "limit_reached": calls_made >= 5,
            "status": "EXHAUSTED" if calls_made >= 5 else "ACTIVE"
        }
    
    return {
        "rate_limit_policy": "5 chat calls per SHA session",
        "total_active_sessions": len(active_sessions),
        "sessions": active_sessions,
        "note": "Upload new data to get a fresh SHA with 5 new chat calls"
    }

@app.get("/rate-limit-status/{sha}", tags=["Helper"], summary="Check Rate Limit Status for Specific SHA")
def get_rate_limit_status_for_sha(sha: str):
    """
    Check the rate limit status for a specific SHA.
    """
    calls_made = sha_call_counts.get(sha, 0)
    calls_remaining = max(0, 5 - calls_made)
    
    if sha not in sha_call_counts:
        raise HTTPException(
            status_code=404, 
            detail=f"SHA '{sha}' not found. This SHA may not exist or may have expired."
        )
    
    return {
        "sha": sha,
        "calls_made": calls_made,
        "calls_remaining": calls_remaining,
        "limit_reached": calls_made >= 5,
        "status": "EXHAUSTED" if calls_made >= 5 else "ACTIVE",
        "max_calls_per_session": 5
    }

@app.get("/debug-uploaded-data")
async def debug_uploaded_data():
    """Debug endpoint to inspect the loaded data structure."""
    try:
        if not UPLOAD_DIR.exists() or not any(UPLOAD_DIR.iterdir()):
            return {"error": "No upload directory or files found"}
        
        # Try standard loading
        try:
            master_df = load_data_from_directory(UPLOAD_DIR)
            standard_load_success = True
            standard_error = None
        except Exception as e:
            standard_load_success = False
            standard_error = str(e)
            master_df = None
        
        # Try manual loading
        try:
            all_well_data = []
            las_files = []
            for pattern in ['*.las', '*.LAS']:
                las_files.extend(list(UPLOAD_DIR.glob(pattern)))
            for pattern in ['**/*.las', '**/*.LAS']:
                las_files.extend(list(UPLOAD_DIR.glob(pattern)))
            las_files = list(set(las_files))
            
            manual_details = []
            for las_file in las_files:
                try:
                    las = lasio.read(str(las_file))
                    df = las.df().reset_index()
                    well_name = las_file.stem
                    try:
                        if hasattr(las.well, 'WELL') and las.well.WELL.value:
                            well_name = str(las.well.WELL.value)
                    except:
                        pass
                    
                    df['WELL'] = well_name
                    all_well_data.append(df)
                    
                    manual_details.append({
                        "file": las_file.name,
                        "well_name": well_name,
                        "shape": df.shape,
                        "columns": list(df.columns),
                        "dtypes": df.dtypes.to_dict(),
                        "sample_data": df.head(2).to_dict()
                    })
                except Exception as file_error:
                    manual_details.append({
                        "file": las_file.name,
                        "error": str(file_error)
                    })
            
            if all_well_data:
                manual_df = pd.concat(all_well_data, ignore_index=True, sort=False)
                manual_load_success = True
                manual_error = None
            else:
                manual_df = None
                manual_load_success = False
                manual_error = "No data files processed"
                
        except Exception as e:
            manual_load_success = False
            manual_error = str(e)
            manual_df = None
            manual_details = []
        
        debug_info = {
            "upload_directory": str(UPLOAD_DIR),
            "files_found": [f.name for f in UPLOAD_DIR.rglob("*") if f.is_file()],
            "standard_loading": {
                "success": standard_load_success,
                "error": standard_error,
                "data_info": None
            },
            "manual_loading": {
                "success": manual_load_success,
                "error": manual_error,
                "files_processed": manual_details
            }
        }
        
        # Add standard loading info if successful
        if standard_load_success and master_df is not None:
            debug_info["standard_loading"]["data_info"] = {
                "shape": master_df.shape,
                "columns": list(master_df.columns),
                "dtypes": master_df.dtypes.to_dict(),
                "well_names": master_df.get('WELL', pd.Series()).unique().tolist() if 'WELL' in master_df.columns else "No WELL column",
                "sample_data": master_df.head(2).to_dict() if not master_df.empty else "Empty DataFrame"
            }
        
        # Add manual loading info if successful
        if manual_load_success and manual_df is not None:
            debug_info["manual_loading"]["combined_data_info"] = {
                "shape": manual_df.shape,
                "columns": list(manual_df.columns),
                "dtypes": manual_df.dtypes.to_dict(),
                "well_names": manual_df.get('WELL', pd.Series()).unique().tolist() if 'WELL' in manual_df.columns else "No WELL column",
                "sample_data": manual_df.head(2).to_dict() if not manual_df.empty else "Empty DataFrame"
            }
        
        return debug_info
        
    except Exception as e:
        return {"error": f"Debug failed: {str(e)}"}


@app.get("/debug-env")
async def debug_env():
    """A temporary endpoint to check the live environment variable."""
    key = os.getenv("GEMINI_API_KEY", "GEMINI_API_KEY was NOT FOUND")
    
    # Return only the last 4 characters for security
    key_display = f"...{key[-4:]}" if "NOT FOUND" not in key else key
    
    return {
        "message": "Reading environment variable from the live server.",
        "GEMINI_API_KEY": key_display
    }

@app.post("/generate-pdf-report", tags=["Step 5: Reporting"], summary="Generate a PDF Report with AI Summary")
async def generate_pdf_report(request: ReportRequest):
    """
    Generate a comprehensive PDF report for specified wells.
    This endpoint does not count against the chat rate limit.
    """
    try:
        if not request.well_names:
            raise HTTPException(status_code=400, detail="well_names cannot be empty")
        
        image_paths_for_ai = []
        missing_wells = []
        
        for well_name in request.well_names:
            safe_folder_name = str(well_name).replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")
            well_dir = OUTPUT_DIR / safe_folder_name
            
            if not well_dir.is_dir():
                missing_wells.append(well_name)
                continue
                
            well_images = list(well_dir.glob("*.png"))
            if not well_images:
                missing_wells.append(well_name)
                continue
                
            image_paths_for_ai.extend(str(p) for p in well_images)
        
        if missing_wells:
            raise HTTPException(
                status_code=404, 
                detail=f"Output/images not found for wells: {', '.join(missing_wells)}. Please generate plots first."
            )
        
        if not image_paths_for_ai:
            raise HTTPException(
                status_code=404, 
                detail="No images found for any of the specified wells. Please generate plots first."
            )
        
        print(f"📄 Generating PDF report for {len(request.well_names)} wells")
        print(f"🖼️  Using {len(image_paths_for_ai)} plot images")
        
        # Generate AI summary for report (this doesn't count against rate limit)
        try:
            ai_summary = await call_gemini_api(
                question="Provide a comprehensive summary and comparison for all the wells based on these plots.",
                image_paths=image_paths_for_ai,
                system_prompt=REPORT_SUMMARY_PROMPT
            )
            print("🤖 AI summary generated successfully for PDF report")
        except Exception as e:
            print(f"⚠️ AI summary failed, using fallback: {e}")
            ai_summary = f"""
Executive Summary

This report presents machine learning-based analysis results for {len(request.well_names)} wells: {', '.join(request.well_names)}. The analysis employed K-means clustering and visualization techniques to automatically identify different rock types and assess reservoir potential.

The generated plots include well log classifications showing different lithofacies, elbow method analysis for optimal cluster determination, and data pattern recognition.
            """
        
        try:
            pdf_buffer = create_well_report_pdf(request.well_names, ai_summary)
            print("✅ PDF report generated successfully")
            
            return Response(
                content=pdf_buffer.getvalue(),
                media_type="application/pdf",
                headers={"Content-Disposition": "attachment; filename=well_analysis_report.pdf"}
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to generate PDF report: {str(e)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Unexpected error generating report: {str(e)}"
        )

# 🚀 MAIN ENTRY POINT - CRITICAL FOR PORT BINDING
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Well Log Clustering API manually...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=False,
        access_log=True
    )
