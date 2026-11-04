"""Inference-only mass regression network used by this release."""

import torch
from torch import nn


class MassModel(nn.Module):
    def __init__(self, feature_dim: int = 512, residual_scale: float = 0.5):
        super().__init__()
        self.image_attention = nn.Sequential(nn.Linear(feature_dim, 128), nn.Tanh(), nn.Linear(128, 1))
        self.view_metadata_encoder = nn.Sequential(nn.Linear(3, feature_dim), nn.LayerNorm(feature_dim), nn.GELU())
        self.text_attention = nn.Sequential(nn.Linear(feature_dim, 64), nn.Tanh(), nn.Linear(64, 1))
        self.geometry_encoder = nn.Sequential(nn.Linear(6, 128), nn.LayerNorm(128), nn.GELU())
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2 + 128, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.GELU(),
        )
        self.density_head = nn.Linear(128, 1)
        self.residual_head = nn.Linear(128, 1)
        self.residual_scale = residual_scale

    def forward(self, images, global_text, component_text, geometry, view_metadata):
        images = images.float() + self.view_metadata_encoder(view_metadata)
        view_weights = torch.softmax(self.image_attention(images).squeeze(-1), dim=1)
        image_feature = (images * view_weights.unsqueeze(-1)).sum(dim=1)
        texts = torch.stack([global_text.float(), component_text.float()], dim=1)
        text_weights = torch.softmax(self.text_attention(texts).squeeze(-1), dim=1)
        text_feature = (texts * text_weights.unsqueeze(-1)).sum(dim=1)
        geometry_feature = self.geometry_encoder(geometry)
        hidden = self.fusion(torch.cat([image_feature, text_feature, geometry_feature], dim=-1))
        density = torch.nn.functional.softplus(self.density_head(hidden)) + 1e-4
        residual = self.residual_scale * torch.tanh(self.residual_head(hidden))
        mass = torch.exp(geometry[:, 4]).unsqueeze(-1) * density * torch.exp(residual)
        return mass.squeeze(-1)
