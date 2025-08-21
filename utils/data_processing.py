# utils/data_processing.py
import io
import os
import zipfile
from typing import Dict, List, Optional

import lasio
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Import the configuration directly
from .config import CLIENT_MAPPING, LOG_CONFIG


def get_las_headers_from_zip(zip_bytes: bytes) -> list:
    """Reads a zip file in bytes and returns a unique list of all headers."""
    all_headers = set()
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
            for entry in zf.namelist():
                if not entry.lower().endswith('.las') or entry.startswith('__MACOSX'):
                    continue
                try:
                    raw_bytes = zf.read(entry)
                    text = raw_bytes.decode('utf-8', errors='replace')
                    las = lasio.read(io.StringIO(text))
                    for key in las.keys():
                        all_headers.add(key)
                except Exception as e:
                    print(f"  ⚠️ Could not parse headers from '{entry}'. Error: {e}")
    except zipfile.BadZipFile:
        print("  ❌ CRITICAL ERROR: The provided file is not a valid ZIP archive.")
        return []
    return sorted(list(all_headers))


# --- UPDATED MAPPING FUNCTION TO HANDLE BOTH CASES ---
def map_las_headers(unknown: List[str], custom_mapping: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Handles mapping logic.
    - If a `custom_mapping` dictionary is provided, it uses that exclusively.
    - If not, it falls back to the "smart" mapping (direct match -> config alias).
    """
    final_mapping = {}
    reference_headers = [log['mnemonic'] for log in LOG_CONFIG]
    unknown_lookup = {col.upper(): col for col in unknown}

    if custom_mapping:
        # --- Case 1: Use the explicit mapping from the API request body ---
        print("\n--- Mapping LAS Headers using custom client-provided map ---")
        client_mapping_upper = {k.upper(): v.upper() for k, v in custom_mapping.items()}
        for ref_header in reference_headers:
            target_header = client_mapping_upper.get(ref_header.upper())
            if target_header and target_header in unknown_lookup:
                original_case_header = unknown_lookup[target_header]
                final_mapping[ref_header] = original_case_header
                print(f"  ✅ Mapped '{ref_header}' to '{original_case_header}' via custom map")
            else:
                final_mapping[ref_header] = None
    else:
        # --- Case 2: Use the default "smart" mapping logic ---
        print("\n--- Mapping LAS Headers (Smart Mapping) ---")
        for ref_header in reference_headers:
            found_match = None
            ref_header_upper = ref_header.upper()
            if ref_header_upper in unknown_lookup:
                found_match = unknown_lookup[ref_header_upper]
                print(f"  ✅ Found direct match for '{ref_header}' -> '{found_match}'")
            else:
                target_alias = CLIENT_MAPPING.get(ref_header)
                if target_alias and target_alias != "None" and target_alias.upper() in unknown_lookup:
                    found_match = unknown_lookup[target_alias.upper()]
                    print(f"  ✅ Mapped '{ref_header}' to alias '{found_match}' via config.py")
            final_mapping[ref_header] = found_match

    return final_mapping


def load_and_combine_las_from_zip(zip_bytes: bytes):
    """Loads all LAS files from zip bytes and combines them into a single DataFrame."""
    # This function remains unchanged.
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
    depth_col = next((col for col in combined_df.columns if col.upper() in ['DEPT', 'DEPTH']), None)
    if depth_col:
        combined_df.rename(columns={depth_col: 'DEPTH'}, inplace=True)
    else:
        print("❌ CRITICAL ERROR: No 'DEPT' or 'DEPTH' column found in any LAS file.")
        return None
    return combined_df


def find_and_prepare_ml_features(df: pd.DataFrame):
    """Prepares numerical features from a DataFrame for machine learning tasks."""
    # This function remains unchanged.
    print("\n--- Preparing Features for Machine Learning ---")
    cols_to_exclude = ['DEPTH', 'WELL', 'CLASSIFICATION']
    potential_features = [col for col in df.columns if col not in cols_to_exclude]
    ml_df_base = df[potential_features].select_dtypes(include=np.number).copy()

    valid_feature_cols = [col for col in ml_df_base.columns if ml_df_base[col].nunique() > 1]
    
    if len(valid_feature_cols) < 1:
        print("  ⚠️ No valid non-constant features found for ML. Skipping classification.")
        return None
        
    print(f"  ✅ Using {len(valid_feature_cols)} feature(s) for clustering: {valid_feature_cols}")

    ml_df_valid = ml_df_base[valid_feature_cols].copy()
    ml_df_valid.interpolate(method='linear', inplace=True, limit_direction='both')
    ml_df_valid.bfill(inplace=True)
    ml_df_valid.ffill(inplace=True)
    ml_df_valid.dropna(how='any', inplace=True)

    if ml_df_valid.empty:
        print("  ❌ Data is empty after cleaning. Skipping classification.")
        return None

    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(ml_df_valid)
    return pd.DataFrame(x_scaled, columns=ml_df_valid.columns, index=ml_df_valid.index)