import json
import argparse
import numpy as np
from datasets.utils import load_csv, encode_labels, scale_features, generate_windows, save_test_patches

def main():
    parser = argparse.ArgumentParser(
        description="Convert a processed CSV into test patches via a JSON config file."
    )
    parser.add_argument(
        '--config', required=True,
        help='Path to JSON config file with keys: csv, output_dir, window_size, stride, label_column, delimiter'
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config) as cf:
        config = json.load(cf)
    cfg = config.get('patch', config)

    # Extract parameters
    csv_path = cfg['csv']
    output_dir = cfg['output_dir']
    window_size = cfg.get('window_size', 700)
    stride = cfg.get('stride', 20)
    label_col = cfg.get('label_column', 'GROUP')
    delimiter = cfg.get('delimiter', ';')

    # Load CSV
    df = load_csv(csv_path, sep=delimiter)
    print(f"[DEBUG] Loaded CSV from '{csv_path}' with columns: {df.columns.tolist()}")

    # Validate label column
    if label_col not in df.columns:
        raise KeyError(f"Expected '{label_col}' column for labels in CSV but found: {df.columns.tolist()}")

    # Encode labels (including 'UNKNOWN')
    raw_labels = df[label_col]
    print(f"[DEBUG] Encoding {len(raw_labels)} labels from column '{label_col}'.")
    labels, _ = encode_labels(raw_labels)

    # Drop label column from features
    df = df.drop(columns=[label_col])

    # Scale numeric features (coercing non-numeric to 0.0)
    df_scaled, _ = scale_features(df)

    # Generate sliding-window patches
    X_patches, Y_patches = generate_windows(df_scaled, labels, window_size, stride)

    # Save to disk
    save_test_patches(X_patches, Y_patches, output_dir)
    print(f"Saved {len(X_patches)} test patches to '{output_dir}'")

if __name__ == '__main__':
    main()
