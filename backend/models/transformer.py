import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_activation(name: str = 'relu') -> nn.Module:
    """
    Returns a PyTorch activation module by name.
    Supported: 'relu', 'prelu', 'gelu'.
    """
    name = name.lower()
    if name == 'prelu':
        return nn.PReLU()
    if name == 'relu':
        return nn.ReLU()
    if name == 'gelu':
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self._max_len = max_len
        self.pe = self._build_pe(max_len)

    def _build_pe(self, length):
        pe = torch.zeros(length, self.d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(1)  # (L, 1, D)

    def forward(self, x: torch.Tensor):
        seq_len = x.size(0)
        if seq_len > self.pe.size(0):
            print(f"[INFO] Rebuilding positional encoding for seq_len={seq_len}")
            self.pe = self._build_pe(seq_len).to(x.device)

        pe_slice = self.pe[:seq_len]
        return x + pe_slice




class W2WTransformerModel(nn.Module):
    """
    A DETR-style Transformer model for well-boundary detection.

    Args:
      in_channels: number of input feature channels (C).
      num_classes: number of target classes (excluding 'no-object').
      num_queries: number of learned object queries (N).
      d_model:      transformer feature dimension.
      nheads:       number of attention heads.
      num_encoder_layers: number of encoder layers.
      num_decoder_layers: number of decoder layers.
      dim_feedforward:    feedforward network dimension.
      dropout:            dropout rate.
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_queries: int,
        d_model: int,
        nheads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ):
        super().__init__()
        # Input projection: (batch, C, L) -> (batch, d_model, L)
        self.input_proj = nn.Conv1d(in_channels, d_model, kernel_size=1)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # Learned object queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Prediction heads
        self.class_embed = nn.Linear(d_model, num_classes + 1)  # +1 for 'no-object'
        self.bbox_embed = nn.Linear(d_model, 4)

        # Initialize weights
        self._reset_parameters()
        self.d_model = d_model
        self.num_queries = num_queries

    def _reset_parameters(self):
        nn.init.normal_(self.query_embed.weight, std=0.02)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> dict:
        bs, C, L = x.shape

        x = self.input_proj(x)        # (B, d_model, L)
        x = x.permute(2, 0, 1)         # (L, B, d_model)
        pos = self.pos_encoder(x)     # (L, B, d_model)

        memory = self.transformer.encoder(pos)

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)  # (num_queries, B, d_model)
        tgt = torch.zeros_like(query_embed)                                 # (num_queries, B, d_model)

        # Decoder returns (num_queries, B, d_model)
        hs = self.transformer.decoder(tgt, memory)

        # Permute to (B, num_queries, d_model)
        hs = hs.permute(1, 0, 2)

        # Prediction heads
        outputs_class = self.class_embed(hs)
        outputs_bbox  = self.bbox_embed(hs).sigmoid()

        return {'pred_logits': outputs_class, 'pred_boxes': outputs_bbox}