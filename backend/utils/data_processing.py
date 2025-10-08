# utils/data_processing.py
import io
import zipfile
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import lasio
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .config import CLIENT_MAPPING, LOG_CONFIG

def unzip_and_save_las_files(zip_bytes: bytes, target_dir: Path) -> Tuple[List[str], List[str]]:
    """
    Cleans the target directory, unzips a file in memory, saves all .las files,
    and returns the headers and well names found.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    all_headers, all_well_names = set(), set()

    # Clean the directory before adding new files
    for f in target_dir.glob('*.las'):
        f.unlink()

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
            for entry in zf.namelist():
                if entry.endswith('/') or not entry.lower().endswith('.las'):
                    continue
                
                filename = Path(entry).name
                well_name = filename.split('.')[0]
                all_well_names.add(well_name)
                
                raw_bytes = zf.read(entry)
                (target_dir / filename).write_bytes(raw_bytes)

                text_content = raw_bytes.decode('utf-8', errors='replace')
                las = lasio.read(io.StringIO(text_content), ignore_data=True)
                for key in las.keys():
                    all_headers.add(key)
    except zipfile.BadZipFile:
        return [], []
    
    return sorted(list(all_headers)), sorted(list(all_well_names))

def load_data_from_directory(source_dir: Path) -> Optional[pd.DataFrame]:
    """Loads all .las files from a given directory into a single DataFrame."""
    las_dfs = []
    for las_file in source_dir.glob('*.las'):
        try:
            las = lasio.read(str(las_file))
            df = las.df().reset_index()
            df['WELL'] = las_file.stem
            las_dfs.append(df)
        except Exception as e:
            print(f"  ⚠️ Could not parse '{las_file.name}'. Error: {e}")

    if not las_dfs:
        return None

    combined_df = pd.concat(las_dfs, ignore_index=True)
    depth_col = next((col for col in combined_df.columns if col.upper() in ['DEPT', 'DEPTH']), None)
    if depth_col:
        combined_df.rename(columns={depth_col: 'DEPTH'}, inplace=True)
    else:
        return None
    return combined_df

def map_las_headers(unknown: List[str], custom_mapping: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Handles both custom mapping and the default 'smart' mapping."""
    final_mapping = {}
    reference_headers = [log['mnemonic'] for log in LOG_CONFIG]
    unknown_lookup = {col.upper(): col for col in unknown}
    if custom_mapping is not None:
        custom_mapping_upper = {k.upper(): v for k, v in custom_mapping.items()}
        for ref_header in reference_headers:
            target_header = custom_mapping_upper.get(ref_header.upper())
            if target_header and target_header.upper() != "NONE" and target_header.upper() in unknown_lookup:
                final_mapping[ref_header] = unknown_lookup[target_header.upper()]
            else:
                final_mapping[ref_header] = None
    else:
        for ref_header in reference_headers:
            if ref_header.upper() in unknown_lookup:
                final_mapping[ref_header] = unknown_lookup[ref_header.upper()]
            else:
                alias = CLIENT_MAPPING.get(ref_header)
                if alias and alias != "None" and alias.upper() in unknown_lookup:
                    final_mapping[ref_header] = unknown_lookup[alias.upper()]
                else:
                    final_mapping[ref_header] = None
    return final_mapping

def find_and_prepare_ml_features(df: pd.DataFrame):
    """Prepares numerical features from a DataFrame for machine learning tasks."""
    cols_to_exclude = ['DEPTH', 'WELL', 'CLASSIFICATION']
    potential_features = [col for col in df.columns if col not in cols_to_exclude]
    ml_df_base = df[potential_features].select_dtypes(include=np.number).copy()
    valid_feature_cols = [col for col in ml_df_base.columns if ml_df_base[col].nunique() > 1]
    if not valid_feature_cols: return None
    ml_df_valid = ml_df_base[valid_feature_cols].copy()
    ml_df_valid.interpolate(method='linear', inplace=True, limit_direction='both')
    ml_df_valid.bfill(inplace=True); ml_df_valid.ffill(inplace=True)
    ml_df_valid.dropna(how='any', inplace=True)
    if ml_df_valid.empty: return None
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(ml_df_valid)
    return pd.DataFrame(x_scaled, columns=ml_df_valid.columns, index=ml_df_valid.index)