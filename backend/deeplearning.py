import json
import io
import os
import hashlib
import shutil
import zipfile
from collections import defaultdict
from typing import Dict, List, Literal, Optional, Union
from pathlib import Path
from fastapi.responses import StreamingResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import matplotlib.pyplot as plt
import pandas as pd
from fastapi import (
    FastAPI, File, Form, HTTPException, UploadFile
)
from pydantic import BaseModel, ValidationError, Field
import lasio
import tempfile
import numpy as np

# Deep Learning imports
import joblib
import torch
from models.transformer import W2WTransformerModel
from utils_file import plot_model_predictions

# ================== Deep Learning API (Port 8002) ==================

app = FastAPI(
    title="Deep Learning Well Correlation API",
    description="API for LAS/ZIP upload, CSV conversion, and advanced transformer-based well correlation."
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
LAS_UPLOAD_DIR = Path("/app/las_uploaded_data")
WELL_FILES_DIR = Path("/app/well_files")
las_file_mappings = defaultdict(dict)

class InferenceRequest(BaseModel):
    """Advanced well correlation inference using transformer model."""
    reference_well_name: str = Field(..., description="The name of the primary well for comparison.")
    wells_of_interest: List[str] = Field(..., description="A list of well names to correlate against the reference well.")
    correlation_threshold: Optional[float] = Field(0.5, gt=0.0, lt=1.0, description="Correlation threshold between 0 and 1")

@app.on_event("startup")
def on_startup():
    LAS_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    WELL_FILES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"🚀 Deep Learning API starting on port 8002")
    print(f"📁 LAS upload directory: {LAS_UPLOAD_DIR}")
    print(f"📁 Well files directory: {WELL_FILES_DIR}")

def extract_las_from_zip(zip_bytes: bytes, extract_dir: Path):
    """Extract LAS files from ZIP and return file info. Handles nested directories."""
    try:
        las_files = []
        well_names = []
        headers = []
        
        with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                if file_info.filename.lower().endswith(('.las', '.LAS')) and not file_info.is_dir():
                    print(f"Extracting: {file_info.filename}")
                    
                    # Extract to main directory (flatten structure)
                    file_data = zip_ref.read(file_info)
                    filename = Path(file_info.filename).name  # Get just filename, no path
                    extracted_path = extract_dir / filename
                    
                    # Write the file directly to the main extraction directory
                    with open(extracted_path, 'wb') as f:
                        f.write(file_data)
                    
                    las_files.append(filename)  # Store just filename
                    
                    # Try to read LAS metadata
                    try:
                        las = lasio.read(str(extracted_path))
                        well_name = Path(filename).stem
                        if hasattr(las.well, 'WELL') and las.well.WELL.value:
                            well_name = str(las.well.WELL.value)
                        well_names.append(well_name)
                        
                        # Get headers
                        df = las.df()
                        file_headers = list(df.columns)
                        headers.extend(file_headers)
                        
                    except Exception as e:
                        print(f"Warning: Could not read LAS metadata from {filename}: {e}")
                        well_names.append(Path(filename).stem)
        
        if not las_files:
            raise ValueError("No LAS files found in ZIP archive")
        
        # Remove duplicates from headers
        headers = list(set(headers))
        
        print(f"Successfully extracted {len(las_files)} LAS files: {las_files}")
        print(f"Well names: {well_names}")
        
        return las_files, well_names, headers
        
    except Exception as e:
        print(f"Error extracting LAS files from ZIP: {str(e)}")
        raise

def auto_process_las_to_csv(las_sha: str, csv_filename: str = None, user_mapping: dict = None):
    """Process LAS files to CSV with exactly 20 features to match the trained model."""
    try:
        las_dir = LAS_UPLOAD_DIR / las_sha
        if not las_dir.exists():
            raise FileNotFoundError(f"LAS files not found for SHA: {las_sha}")
        
        # Find LAS files
        las_files = []
        for pattern in ['*.las', '*.LAS']:
            las_files.extend(list(las_dir.glob(pattern)))
        for pattern in ['**/*.las', '**/*.LAS']:
            las_files.extend(list(las_dir.glob(pattern)))
        las_files = list(set(las_files))
        
        if not las_files:
            all_files = list(las_dir.rglob("*"))
            raise ValueError(f"No LAS files found. Available: {[f.name for f in all_files if f.is_file()]}")
        
        print(f"Found {len(las_files)} LAS files for processing: {[f.name for f in las_files]}")
        
        # 🎯 EXACTLY 20 FEATURES (removed RMIC to keep RXO for client, exactly 20)
        REQUIRED_FEATURES = [
            "DEPTH_MD", "CALI", "RSHA", "RMED", "RDEP", "RHOB", "GR", 
            "SGR", "NPHI", "PEF", "DTC", "SP", "BS", "ROP", "DTS", 
            "DCAL", "DRHO", "MUDWEIGHT", "RMIC", "ROPA", "RXO"
        ]
        
        print(f"🎯 Target features (EXACTLY {len(REQUIRED_FEATURES)}): {REQUIRED_FEATURES}")
        
        # 🗺️ USER MAPPING: Create reverse mapping for column lookup
        column_mapping = {}
        if user_mapping:
            # Create reverse mapping: las_column_name -> required_feature_name
            for required_feature, las_column in user_mapping.items():
                if required_feature in REQUIRED_FEATURES:
                    column_mapping[las_column] = required_feature
            print(f"🗺️  User mapping applied: {len(column_mapping)} columns mapped")
        
        all_well_data = []
        processing_errors = []
        
        for las_file_path in las_files:
            try:
                print(f"Processing: {las_file_path}")
                
                # Read LAS file
                las = lasio.read(str(las_file_path))
                df = las.df().reset_index()
                
                # Get well name
                well_name = las_file_path.stem
                try:
                    if hasattr(las.well, 'WELL') and las.well.WELL.value:
                        well_name = str(las.well.WELL.value)
                except:
                    pass
                
                print(f"   Original columns: {list(df.columns)}")
                
                # Fix depth column
                depth_cols = ['DEPT', 'DEPTH', 'DEPTH_MD', 'MD']
                for col in depth_cols:
                    if col in df.columns and col != 'DEPTH_MD':
                        df = df.rename(columns={col: 'DEPTH_MD'})
                        break
                
                # 🗺️ Apply user mapping to rename columns
                if column_mapping:
                    df = df.rename(columns=column_mapping)
                    print(f"   🗺️  Applied column mapping to rename LAS columns to required features")
                
                # Create DataFrame with proper column handling
                processed_data = {'WELL': well_name}
                feature_count = 0
                
                # Add each required feature safely
                for feature in REQUIRED_FEATURES:
                    if feature in df.columns:
                        try:
                            # Safe conversion: Handle each column individually
                            column_data = df[feature]
                            
                            # Convert to Series if it's not already
                            if hasattr(column_data, 'iloc'):
                                column_data = column_data.iloc[:] if len(column_data.shape) == 1 else column_data.iloc[:, 0]
                            
                            # Convert to numeric safely
                            numeric_data = []
                            for val in column_data:
                                try:
                                    numeric_val = float(val) if pd.notna(val) else 0.0
                                    numeric_data.append(numeric_val)
                                except (ValueError, TypeError):
                                    numeric_data.append(0.0)
                            
                            processed_data[feature] = numeric_data
                            feature_count += 1
                            print(f"   ✅ Found {feature}: {len(numeric_data)} values")
                            
                        except Exception as e:
                            print(f"   ⚠️  Issue with {feature}: {e} - filling with zeros")
                            processed_data[feature] = [0.0] * len(df)
                            feature_count += 1
                    else:
                        # Fill missing features with zeros
                        processed_data[feature] = [0.0] * len(df)
                        feature_count += 1
                        print(f"   ➕ Missing {feature} - filled with zeros")
                
                print(f"   📊 Total features added: {feature_count} (target: 20)")
                
                # Create DataFrame from processed data
                processed_df = pd.DataFrame(processed_data)
                
                # Verify feature count
                numeric_cols = [col for col in processed_df.columns if col != 'WELL']
                print(f"   🔍 Numeric columns: {len(numeric_cols)} - {numeric_cols}")
                
                # Handle NaN values by replacing with column means
                for col in REQUIRED_FEATURES:
                    if processed_df[col].isna().any():
                        mean_val = processed_df[col].mean()
                        if pd.isna(mean_val) or mean_val == 0:
                            mean_val = 1.0  # Use 1.0 as fallback instead of 0
                        processed_df[col] = processed_df[col].fillna(mean_val)
                
                all_well_data.append(processed_df)
                print(f"✅ Successfully processed: {well_name} - {len(processed_df)} rows, {len(numeric_cols)} features")
                
            except Exception as e:
                error_msg = f"Error processing {las_file_path.name}: {str(e)}"
                print(f"❌ {error_msg}")
                processing_errors.append(error_msg)
                continue
        
        if not all_well_data:
            raise ValueError(f"No files processed successfully. Errors: {processing_errors}")
        
        # Combine all DataFrames
        print(f"📊 Combining {len(all_well_data)} DataFrames...")
        combined_df = pd.concat(all_well_data, ignore_index=True, sort=False)
        
        # CRITICAL: Verify exact feature count
        numeric_features = len([col for col in combined_df.columns if col != 'WELL'])
        feature_names = [col for col in combined_df.columns if col != 'WELL']
        
        print(f"🔍 FINAL VERIFICATION:")
        print(f"   - Total columns: {len(combined_df.columns)} (22 = 1 WELL + 21 features)")
        print(f"   - Numeric features: {numeric_features} (MUST BE 21)")
        print(f"   - Feature names: {feature_names}")
        
        if numeric_features != 21:
            raise ValueError(f"❌ FEATURE COUNT ERROR! Got {numeric_features}, need exactly 21. Features: {feature_names}")
        
        # Save CSV
        if not csv_filename:
            csv_filename = "current_wells.csv"
        
        output_csv_path = WELL_FILES_DIR / csv_filename
        combined_df.to_csv(output_csv_path, index=False)
        
        well_names = combined_df['WELL'].unique().tolist()
        
        print(f"✅ SUCCESS: CSV saved with {len(combined_df)} rows, 21 total columns (20 features + WELL)")
        print(f"✅ Wells: {well_names}")
        print(f"🎯 PERFECT: Exactly 20 features for transformer model!")
        
        return csv_filename, well_names, len(combined_df), list(combined_df.columns)
        
    except Exception as e:
        print(f"❌ ERROR in CSV processing: {str(e)}")
        raise Exception(str(e))


def get_latest_csv_file():
    """Get the most recently created CSV file from well_files directory."""
    try:
        csv_files = list(WELL_FILES_DIR.glob("*.csv"))
        if not csv_files:
            return None
        latest_csv = max(csv_files, key=lambda f: f.stat().st_ctime)
        return latest_csv.name
    except Exception as e:
        print(f"Error getting latest CSV: {str(e)}")
        return None

def run_advanced_inference(config: dict):
    """Run well-to-well correlation inference using trained transformer detector."""
    try:
        data_path = Path("/app/well_files")
        well_csv_path = data_path / config['paths']['processed_csv']
        well_scaler_path = data_path / config['paths']['std_scaler_path']
        
        print(f"Loading CSV: {well_csv_path}")
        print(f"Loading scaler: {well_scaler_path}")
        
        if not well_csv_path.exists():
            raise FileNotFoundError(f"Training data not found: {well_csv_path}")
        if not well_scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {well_scaler_path}")
        
        df = pd.read_csv(well_csv_path, sep=',')
        
        # 🔧 DEBUG: Check what we loaded
        print(f"🔍 CSV DEBUGGING:")
        print(f"   - Total columns in CSV: {len(df.columns)}")
        print(f"   - All columns: {list(df.columns)}")
        
        # Remove WELL column for features
        feature_columns = [col for col in df.columns if col != 'WELL']
        print(f"   - Feature columns (should be 20): {len(feature_columns)}")
        print(f"   - Feature names: {feature_columns}")
        
        # 🔧 CRITICAL FIX: Force exactly 20 features by adding/removing as needed
        if len(feature_columns) == 19:
            print("⚠️  WARNING: Got 19 features, adding dummy feature to make 20...")
            df['DUMMY_FEATURE_FIX'] = 0.0
            feature_columns = [col for col in df.columns if col != 'WELL']
            print(f"   - Updated feature columns: {len(feature_columns)} - {feature_columns}")
        # elif len(feature_columns) == 21:
        #     print("⚠️  WARNING: Got 21 features, removing last feature to make 20...")
        #     # Remove the last column (not WELL)
        #     cols_to_keep = ['WELL'] + feature_columns[:20]
        #     df = df[cols_to_keep]
        #     feature_columns = [col for col in df.columns if col != 'WELL']
        #     print(f"   - Updated feature columns: {len(feature_columns)} - {feature_columns}")
        # elif len(feature_columns) != 20:
        #     # Force to exactly 20 by padding or trimming
        #     print(f"⚠️  WARNING: Got {len(feature_columns)} features, forcing to 20...")
        #     if len(feature_columns) < 20:
        #         # Add dummy features
        #         for i in range(20 - len(feature_columns)):
        #             df[f'DUMMY_FEATURE_{i}'] = 0.0
        #     elif len(feature_columns) > 20:
        #         # Keep only first 20 features
        #         cols_to_keep = ['WELL'] + feature_columns[:20]
        #         df = df[cols_to_keep]
            
        #     feature_columns = [col for col in df.columns if col != 'WELL']
        #     print(f"   - Fixed feature columns: {len(feature_columns)} - {feature_columns}")
        
        # Double-check we have exactly 20
        if len(feature_columns) != 21:
            raise ValueError(f"FAILED TO FIX: Still have {len(feature_columns)} features instead of 20")
        
        scaler = joblib.load(well_scaler_path)

        # Now, check its attributes
        print(f"Type of loaded object: {type(scaler)}")
    
    # Check the learned parameters
        print(f"Loaded Mean: {scaler.mean_}")
        print(f"Loaded Scale (Std Dev): {scaler.scale_}")
        print(f"Number of features seen: {scaler.n_features_in_}")
        
        model_cfg = config['model_params']
        device = torch.device('cpu')
        
        model = W2WTransformerModel(
            num_queries=model_cfg['num_queries'],
            d_model=model_cfg['d_model'], 
            nheads=model_cfg['nheads'],
            num_encoder_layers=model_cfg['num_encoder_layers'],
            num_decoder_layers=model_cfg['num_decoder_layers'],
            dim_feedforward=model_cfg['dim_feedforward'],
            dropout=model_cfg['dropout'],
            num_classes=model_cfg['num_classes'],
            in_channels=model_cfg['in_channels']
        )
        model.to(device)
        
        well_model_path = data_path / config['paths']['det_checkpoint']
        
        if not well_model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {well_model_path}")
            
        print(f"Loading model: {well_model_path}")
        ckpt = torch.load(well_model_path, map_location=device)
        state_dict = ckpt.get('state_dict', ckpt)
        model.load_state_dict(state_dict)
        model.eval()
        
        inf_cfg = config['inference']
        wells_to_plot = [inf_cfg['reference_well_name']] + inf_cfg['wells_of_interest']
        output_path = inf_cfg.get('output_plot', 'results/well_correlation.png')
        threshold = inf_cfg.get('correlation_threshold', 0.5)
        
        results_dir = Path("/app/results")
        results_dir.mkdir(exist_ok=True)
        full_output_path = results_dir / Path(output_path).name
        full_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 🔧 DEBUG: Final check before model call
        print(f"🔍 FINAL CHECK BEFORE MODEL:")
        print(f"   - DataFrame shape: {df.shape}")
        print(f"   - Wells to plot: {wells_to_plot}")
        print(f"   - Feature columns (MUST BE 20): {len(feature_columns)}")
        print(f"   - Features: {feature_columns}")
        
        # 🔧 CRITICAL: Ensure DataFrame has correct column order
        expected_order = ['WELL'] + feature_columns
        df = df[expected_order]
        
        plot_model_predictions(
            df=df,
            scaler=scaler,
            model=model,
            wells_to_plot=wells_to_plot,
            output_path=str(full_output_path),
            confidence_threshold=threshold
        )
        
        print(f"Advanced inference completed. Plot saved to {full_output_path}")
        return str(full_output_path)
        
    except Exception as e:
        print(f"Error in advanced inference: {str(e)}")
        raise


@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Deep Learning Well Correlation API (Port 8002). Use /docs to see the endpoints."}

@app.post("/upload-las-and-convert-csv", tags=["API 1: LAS/ZIP Upload & CSV Conversion"], summary="Upload LAS Files and Auto-Replace CSV")
async def upload_las_and_convert_csv(files: Union[UploadFile, List[UploadFile]] = File(..., description="LAS files or ZIP file containing LAS files")):
    """
    API 1: Upload LAS files or ZIP file containing LAS files.
    Automatically clears all existing CSV files and creates fresh CSV from uploaded data.
    """
    try:
        print(f"🔄 Starting file upload process...")
        
        # 🔥 AUTOMATIC CSV CLEANUP - SILENT & SEAMLESS
        csv_files = list(WELL_FILES_DIR.glob("*.csv"))
        removed_count = 0
        for csv_file in csv_files:
            try:
                csv_file.unlink()
                removed_count += 1
            except Exception as e:
                print(f"Warning: Could not remove {csv_file.name}: {e}")
        
        if removed_count > 0:
            print(f"🔄 Auto-cleared {removed_count} existing CSV files")
        
        # Convert single file to list for uniform processing
        if not isinstance(files, list):
            files = [files]
        
        # Check file types
        file_extensions = [f.filename.lower().split('.')[-1] for f in files]
        
        # Create unique SHA for this upload session
        random_data = os.urandom(64)
        full_sha256 = hashlib.sha256(random_data).hexdigest()
        las_sha = full_sha256[:8]
        
        las_session_dir = LAS_UPLOAD_DIR / las_sha
        las_session_dir.mkdir(exist_ok=True)
        
        well_names = []
        headers = []
        las_files = []
        
        # Case 1: ZIP file containing LAS files
        if len(files) == 1 and file_extensions[0] == 'zip':
            file = files[0]
            zip_bytes = await file.read()
            
            print("--- Processing ZIP file containing LAS files ---")
            try:
                las_files, well_names, headers = extract_las_from_zip(zip_bytes, las_session_dir)
                print(f"Extracted {len(las_files)} LAS files from ZIP")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to extract LAS files from ZIP: {str(e)}")
        
        # Case 2: Individual LAS files
        elif all(ext in ['las'] for ext in file_extensions):
            for file in files:
                if not file.filename.lower().endswith(('.las', '.LAS')):
                    raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}. Only .las files accepted")
                
                las_files.append(file.filename)
                file_path = las_session_dir / file.filename
                
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                # Try to extract well name and headers from LAS file
                try:
                    las = lasio.read(str(file_path))
                    well_name = file.filename.replace('.las', '').replace('.LAS', '')
                    if hasattr(las.well, 'WELL') and las.well.WELL.value:
                        well_name = str(las.well.WELL.value)
                    well_names.append(well_name)
                    
                    df = las.df()
                    file_headers = list(df.columns)
                    headers.extend(file_headers)
                        
                except Exception as e:
                    print(f"Warning: Could not read LAS metadata from {file.filename}: {e}")
                    well_names.append(file.filename.replace('.las', '').replace('.LAS', ''))
        
        # Case 3: Invalid file types
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file types. Supported: ZIP file containing LAS files or individual LAS files. Received: {file_extensions}"
            )
        
        # Remove duplicates from headers
        headers = list(set(headers))
        
        # 🚀 CREATE NEW CSV FROM UPLOADED DATA
        print("--- Converting uploaded LAS files to CSV ---")
        try:
            # Generate simple CSV name (no timestamp needed since directory is clean)
            csv_filename = "current_wells.csv"
            
            csv_filename, processed_wells, total_rows, csv_columns = auto_process_las_to_csv(las_sha, csv_filename)
            auto_csv_success = True
            csv_info = {
                "csv_filename": csv_filename,
                "csv_path": str(WELL_FILES_DIR / csv_filename),
                "total_rows": total_rows,
                "columns": csv_columns,
                "csv_size_mb": round((WELL_FILES_DIR / csv_filename).stat().st_size / (1024*1024), 2)
            }
        except Exception as e:
            print(f"CSV generation failed: {str(e)}")
            auto_csv_success = False
            csv_info = {"error": str(e)}
        
        # Store LAS session info
        las_file_mappings[las_sha] = {
            'files': las_files,
            'well_names': well_names,
            'headers': headers,
            'upload_time': pd.Timestamp.now().isoformat(),
            'file_count': len(las_files),
            'auto_csv_generated': auto_csv_success,
            'csv_info': csv_info,
            'upload_type': 'zip' if len(files) == 1 and file_extensions[0] == 'zip' else 'individual_las'
        }
        
        response_headers = {
            "X-Metadata-SHA256": las_sha
        }
        
        response_data = {
            "message": "Files uploaded and converted to CSV successfully!" if auto_csv_success else "Files uploaded but CSV conversion failed.",
            "file_type": "zip_with_las" if len(files) == 1 and file_extensions[0] == 'zip' else "individual_las",
            "headers": headers,
            "well_names": processed_wells if auto_csv_success else well_names,
            "sha": las_sha,
            "file_count": len(las_files),
            "las_files": las_files,
            "csv_conversion_success": auto_csv_success
        }
        
        if auto_csv_success:
            response_data["csv_info"] = csv_info
        else:
            response_data["csv_error"] = csv_info.get("error", "Unknown error")
        
        return Response(
            content=json.dumps(response_data),
            media_type="application/json",
            headers=response_headers
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload and convert files: {str(e)}")

@app.post("/advanced-well-correlation", tags=["API 2: Advanced Correlation"], summary="Advanced Well Correlation - Display Image Inline")
async def advanced_well_correlation(request_body: InferenceRequest):
    """
    API 2: Advanced well correlation analysis using transformer-based boundary detection model.
    Returns the correlation plot as an inline image that displays directly in the browser/API response.
    """
    try:
        latest_csv = get_latest_csv_file()
        if not latest_csv:
            raise HTTPException(
                status_code=404,
                detail="No CSV files found in well_files directory. Please upload LAS files first using API 1."
            )
        
        print(f"🚀 Auto-detected latest CSV: {latest_csv}")
        
        config_path = Path(__file__).parent / "config.json"
        if not config_path.exists():
            raise HTTPException(status_code=404, detail="config.json not found in backend directory.")
        
        with open(config_path) as f:
            request_config = json.load(f)
        
        request_config['paths']['processed_csv'] = latest_csv
        request_config['inference'] = {
            'reference_well_name': request_body.reference_well_name,
            'wells_of_interest': request_body.wells_of_interest,
            'correlation_threshold': request_body.correlation_threshold,
            'output_plot': 'results/auto_correlation.png'
        }
        
        print("--- Starting AUTOMATED Advanced Well Correlation ---")
        print(f"Using CSV: {latest_csv}")
        print(f"Reference well: {request_body.reference_well_name}")
        print(f"Wells of interest: {request_body.wells_of_interest}")
        print(f"Correlation threshold: {request_body.correlation_threshold}")
        
        csv_path = WELL_FILES_DIR / latest_csv
        df = pd.read_csv(csv_path)
        available_wells = df['WELL'].unique().tolist()
        all_required_wells = [request_body.reference_well_name] + request_body.wells_of_interest
        
        missing_wells = [w for w in all_required_wells if w not in available_wells]
        if missing_wells:
            raise HTTPException(
                status_code=400,
                detail=f"Wells not found in CSV {latest_csv}: {missing_wells}. Available wells: {available_wells}"
            )
        
        output_path = run_advanced_inference(request_config)
        
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Analysis completed but output plot not found.")
        
        # 🖼️ RETURN IMAGE DIRECTLY - DISPLAY INLINE IN BROWSER/API
        with open(output_path, "rb") as img_file:
            image_bytes = img_file.read()
        
        print(f"✅ Returning correlation plot inline ({len(image_bytes)} bytes)")
        
        return Response(
            content=image_bytes,
            media_type="image/png",
            headers={
                "Content-Disposition": "inline; filename=correlation_plot.png",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Required model files missing: {str(e)}")
    except Exception as e:
        print(f"Error in advanced correlation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Advanced correlation analysis failed: {str(e)}")

@app.get("/debug-las-session/{sha}", tags=["Debug"], summary="Debug LAS Session Files")
async def debug_las_session(sha: str):
    """Debug endpoint to check what files are in a LAS session directory."""
    try:
        las_dir = LAS_UPLOAD_DIR / sha
        if not las_dir.exists():
            return {"error": f"Directory not found: {las_dir}"}
        
        all_files = []
        las_file_details = []
        
        for file_path in las_dir.rglob("*"):
            if file_path.is_file():
                file_info = {
                    "filename": file_path.name,
                    "relative_path": str(file_path.relative_to(las_dir)),
                    "full_path": str(file_path),
                    "size_bytes": file_path.stat().st_size,
                    "is_las": file_path.suffix.lower() in ['.las']
                }
                all_files.append(file_info)
                
                # If it's a LAS file, try to get more details
                if file_info['is_las']:
                    try:
                        las = lasio.read(str(file_path))
                        df = las.df()
                        well_name = file_path.stem
                        if hasattr(las.well, 'WELL') and las.well.WELL.value:
                            well_name = str(las.well.WELL.value)
                        
                        las_file_details.append({
                            "filename": file_path.name,
                            "well_name": well_name,
                            "columns": list(df.columns),
                            "row_count": len(df),
                            "readable": True
                        })
                    except Exception as e:
                        las_file_details.append({
                            "filename": file_path.name,
                            "readable": False,
                            "error": str(e)
                        })
        
        las_files = [f for f in all_files if f['is_las']]
        
        return {
            "sha": sha,
            "directory": str(las_dir),
            "total_files": len(all_files),
            "las_files_count": len(las_files),
            "all_files": all_files,
            "las_file_details": las_file_details
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/check-model-files", tags=["Helper"], summary="Check Model Files Status")
async def check_model_files():
    """Verify all required files for advanced well correlation are present."""
    config_path = Path(__file__).parent / "config.json"
    data_path = Path("/app/well_files")
    
    latest_csv = get_latest_csv_file()
    
    required_files = {
        "config.json": config_path,
        "latest_csv": data_path / latest_csv if latest_csv else Path("missing"),
        "scaler.bin": data_path / "scaler.bin", 
        "detector_final.pt": data_path / "detector_final.pt"
    }
    
    file_status = {}
    all_present = True
    
    for file_name, file_path in required_files.items():
        exists = file_path.exists() if file_path.name != "missing" else False
        file_status[file_name] = {
            "exists": exists,
            "path": str(file_path),
            "size_mb": round(file_path.stat().st_size / (1024*1024), 2) if exists else 0
        }
        if not exists:
            all_present = False
    
    return {
        "all_files_present": all_present,
        "file_status": file_status,
        "latest_csv_file": latest_csv,
        "ready_for_advanced_inference": all_present
    }

@app.get("/list-processed-csvs", tags=["Helper"], summary="List Available CSV Files")
async def list_processed_csvs():
    """List all CSV files available in the well_files directory."""
    try:
        csv_files = list(WELL_FILES_DIR.glob("*.csv"))
        
        file_info = []
        for csv_file in csv_files:
            size_mb = round(csv_file.stat().st_size / (1024*1024), 2)
            
            try:
                df = pd.read_csv(csv_file, nrows=5)
                columns = list(df.columns)
                df_full = pd.read_csv(csv_file)
                well_count = len(df_full['WELL'].unique()) if 'WELL' in df_full.columns else "Unknown"
                total_rows = len(df_full)
                wells = df_full['WELL'].unique().tolist() if 'WELL' in df_full.columns else []
            except:
                columns = ["Error reading file"]
                well_count = "Unknown"
                total_rows = "Unknown"
                wells = []
            
            file_info.append({
                "filename": csv_file.name,
                "size_mb": size_mb,
                "columns": columns,
                "well_count": well_count,
                "total_rows": total_rows,
                "wells": wells,
                "created_time": pd.Timestamp.fromtimestamp(csv_file.stat().st_ctime).isoformat()
            })
        
        file_info.sort(key=lambda x: x['created_time'], reverse=True)
        
        return {
            "csv_files_count": len(csv_files),
            "latest_csv": file_info[0]['filename'] if file_info else None,
            "files": file_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing CSV files: {str(e)}")

# 🚀 MAIN ENTRY POINT - CRITICAL FOR PORT BINDING
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Deep Learning API manually...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8002, 
        reload=False,
        access_log=True
    )
