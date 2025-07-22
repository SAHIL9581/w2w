import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat
from models.utils import get_activation


class Project(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.adapt_avg_pool = nn.AdaptiveAvgPool1d(in_features)
        self.run = nn.Sequential(nn.Linear(in_features, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2)
        x = self.adapt_avg_pool(x)
        return self.run(x)


class Attention(nn.Module):
    def __init__(self, in_features, num_heads):
        super().__init__()
        self.expand_context = nn.Linear(in_features, in_features*2*num_heads, bias=False)
        self.decompose_context = Rearrange('batch seq (dim chunks heads) -> chunks batch heads seq dim', dim=in_features, heads=num_heads, chunks=2)
        self.expand_query = nn.Linear(in_features, in_features*num_heads, bias=False)
        self.decompose_query = Rearrange('batch seq (dim heads) -> batch heads seq dim', dim=in_features, heads=num_heads)
        self.scale_factor = in_features**(-0.5)
        self.recompose = nn.Sequential(Rearrange('b h t d -> b t (h d)'), nn.Linear(num_heads*in_features, in_features, bias=False))

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        expanded_context = self.expand_context(context)
        key, value = torch.chunk(self.decompose_context(expanded_context), 2, dim=0)
        refined = self.decompose_query(self.expand_query(query))
        scaled_dot_prod = torch.einsum('b h i d , b h j d -> b h i j', refined, key[0]) * self.scale_factor
        attention = torch.softmax(scaled_dot_prod, dim=-1)
        return self.recompose(torch.einsum('b h i j , b h j d -> b h i d', attention, value[0]))


class Transformer(nn.Module):
    def __init__(self, in_features, num_heads, dropout, expansion_factor, act_name):
        super().__init__()
        self.mhsa = Attention(in_features=in_features, num_heads=num_heads)
        self.drop = nn.Dropout(dropout)
        self.norm0 = nn.LayerNorm(in_features)
        self.norm1 = nn.LayerNorm(in_features)
        self.project = nn.Sequential(nn.Linear(in_features, in_features*expansion_factor),
                                     get_activation(act_name),
                                     nn.Dropout(dropout),
                                     nn.Linear(in_features*expansion_factor, in_features),
                                     nn.Dropout(dropout))

    def forward(self, query, context):
        y = self.norm0(self.drop(self.mhsa(query, context))+query)
        return self.norm1(self.project(y)+y)


class Query(nn.Module):
    def __init__(self, seq_len, dim):
        super().__init__()
        # TODO: Initiate the tensor using dataset statistics instead of random values
        self.learnable_query = torch.nn.Parameter(torch.randn(seq_len, dim))
        self.learnable_query.requires_grad = True

    def forward(self, context):
        batch = context.shape[0]
        return repeat(self.learnable_query, 'seq dim -> batch seq dim', batch=batch)