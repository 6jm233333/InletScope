import numpy as np
import torch
from torch import nn


def _check_input(x):
    if x.ndim != 3 or tuple(x.shape[1:]) != (100, 1):
        raise ValueError('Expected single-sensor input with shape (B, 100, 1).')

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(float(np.log(10000.0)) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiScalePatchEmbedding(nn.Module):

    def __init__(self, seq_len, patch_sizes, d_model, input_size):
        super().__init__()
        self.seq_len = seq_len
        self.patch_sizes = tuple(patch_sizes)
        self.projections = nn.ModuleList([
            nn.Linear(patch_size * input_size, d_model)
            for patch_size in self.patch_sizes
        ])

    def forward(self, x):
        features = []
        for patch_size, projection in zip(self.patch_sizes, self.projections):
            num_patches = self.seq_len // patch_size
            patches = x[:, :num_patches * patch_size, :]
            patches = patches.reshape(x.size(0), num_patches,
                                      patch_size * x.size(2))
            features.append(projection(patches))
        return features


class UniformScaleWeights(nn.Module):

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x.new_full((len(x), self.n), 1 / self.n)


class InletScope(nn.Module):

    def __init__(self, patch_sizes=(4, 10, 20)):
        super().__init__()
        self.seq_len = 100
        self.d_model = 64
        self.num_class = 2
        self.enc_in = 1
        self.patch_sizes = list(patch_sizes)
        if (not self.patch_sizes or any(p <= 0 or p > self.seq_len
                                       or self.seq_len % p
                                       for p in self.patch_sizes)):
            raise ValueError('Patch lengths must be positive divisors of 100.')

        self.patch_embedding = MultiScalePatchEmbedding(
            self.seq_len, self.patch_sizes, self.d_model, self.enc_in)
        self.pos_embedding = PositionalEmbedding(self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, dropout=0.1,
            batch_first=True, norm_first=True)
        self.transformers = nn.ModuleList([
            nn.TransformerEncoder(encoder_layer, num_layers=2,
                                  enable_nested_tensor=False)
            for _ in self.patch_sizes
        ])
        self.scale_attention = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, len(self.patch_sizes)), nn.Softmax(dim=-1))
        self.classifier = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 2))

    def forward(self, x):
        _check_input(x)
        encoded = [
            encoder(self.pos_embedding(patches)).mean(dim=1)
            for patches, encoder in zip(self.patch_embedding(x), self.transformers)
        ]
        scales = torch.stack(encoded, dim=1)
        weights = self.scale_attention(scales.mean(dim=1))
        fused = (scales * weights.unsqueeze(-1)).sum(dim=1)
        return self.classifier(fused)


class CausalBlock(nn.Module):

    def __init__(self, cin, cout, dilation):
        super().__init__()
        self.pad = 4 * dilation
        self.conv = nn.Conv1d(cin, cout, 5, dilation=dilation)
        self.skip = nn.Conv1d(cin, cout, 1)

    def forward(self, x):
        padded = nn.functional.pad(x, (self.pad, 0))
        return torch.relu(self.conv(padded) + self.skip(x))


class TCN(nn.Module):

    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            CausalBlock(1, 64, 1), CausalBlock(64, 64, 2),
            CausalBlock(64, 64, 4))
        self.head = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        _check_input(x)
        return self.head(self.blocks(x.transpose(1, 2)).mean(-1))


def build_model(name):

    if name == 'tcn':
        return TCN()
    if name not in ('inletscope', 'single_scale', 'uniform_fusion'):
        raise ValueError(f'Unknown neural model: {name!r}.')
    model = InletScope((10,) if name == 'single_scale' else (4, 10, 20))
    if name == 'uniform_fusion':
        model.scale_attention = UniformScaleWeights(3)
    return model


def statistical_features(x):

    x = np.asarray(x)
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.ndim != 2 or x.shape[1] != 100:
        raise ValueError('Expected pressure windows with shape (B, 100).')
    return np.column_stack([
        x.mean(1), x.std(1), x.min(1), x.max(1), np.median(x, axis=1),
        x[:, -1], x[:, -1] - x[:, 0], np.abs(np.diff(x, axis=1)).mean(1),
    ])
