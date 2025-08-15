import os
import io
import zipfile
import lasio
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils.config import CLIENT_MAPPING

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
                    print(f"     📈 Curves: {list(df.columns)}")
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
    valid_feature_cols = [col for col in ml_df_base.columns if ml_df_base[col].nunique() > 1]
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
