# main.py
"""
Inference script for the W2W model pipeline using a JSON config file.
Usage:
    python main.py --config path/to/config.json
"""
import argparse
import json
import torch
from torch.utils.data import DataLoader

# adjust imports to your project structure
from datasets.W2W import W2WDataset
from utils.utils import collate_fn
from models.W2WModel import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='W2W Inference with JSON config')
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to JSON config file')
    return parser.parse_args()


def load_config(path: str) -> dict:
    """Load JSON config into a dict."""
    with open(path, 'r') as f:
        return json.load(f)


def main():
    args = parse_args()
    config = load_config(args.config)

    # Determine device
    device = config['inference'].get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare dataset and DataLoader
    dataset = W2WDataset(
        zip_folder=config['paths']['zip_folder'],
        raw_las_dir=config['paths']['raw_las_dir'],
        csv_file=config['paths']['processed_csv_path'],
        std_scaler_bin_path=config['paths']['std_scaler_path']
    )
    loader = DataLoader(
        dataset,
        batch_size=config['inference'].get('batch_size', 8),
        collate_fn=collate_fn,
        shuffle=False
    )

    # Build model and load checkpoint
    model = build_model(config)
    ckpt_path = config['paths']['resume']
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # Run inference
    outputs = []
    with torch.no_grad():
        for inputs, meta in loader:
            inputs = inputs.to(device)
            preds = model(inputs)
            outputs.append(preds.cpu())

    # Save outputs
    out_path = config['inference'].get('output_path', 'outputs.pt')
    torch.save(outputs, out_path)
    print(f"Inference complete. Saved to {out_path}")


if __name__ == '__main__':
    main()