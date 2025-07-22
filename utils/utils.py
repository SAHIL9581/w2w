# utils/utils.py
"""
Utility functions for data loading and collation.
"""
import pandas as pd
import torch


def load_data(path: str, delimiter: str = ';') -> pd.DataFrame:
    """Read processed CSV."""
    return pd.read_csv(path, delimiter=delimiter)


def collate_fn(batch):
    """
    Collate batch of (input_tensor, metadata).
    """
    inputs = torch.stack([item[0] for item in batch])
    if isinstance(batch[0][1], dict):
        meta = {k: torch.stack([item[1][k] for item in batch])
                for k in batch[0][1]}
    else:
        meta = None
    return inputs, meta
