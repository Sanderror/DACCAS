"""
Full CAPTCHA attack model implementing:
  "Transformer-based end-to-end attack on text CAPTCHAs with triplet deep attention"
  (Zhang et al., Computers & Security 146, 2024)

Architecture:
  1. Feature Extraction:
     - ResNet45 with TDA module
     - 3-layer Transformer Encoder (8 heads, d_model=512, d_ff=2048, ReLU)
  2. Character Recognition (Decoder):
     - Key: mini U-Net applied to encoder feature map
     - Value: identity mapping of encoder feature map
     - Query: sinusoidal positional encoding -> Query Enhancement (3-layer
       masked self-attention Transformer)
     - Cross-attention: Output = softmax(Q·K^T / sqrt(d_k)) · V
     - Linear classifier -> character probabilities
"""
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet_tda import ResNet45TDA


# ============================================================================
# Positional Encoding (from ABINet, matches paper Eq. 13-15)
# ============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# ============================================================================
# Transformer Encoder (Section 3.1.2)
# ============================================================================
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                                 key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=3,
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()
        layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                          src_key_padding_mask=src_key_padding_mask)
        return output


# ============================================================================
# Feature Extraction Backbone: ResNet45+TDA + Transformer Encoder
# ============================================================================
class FeatureExtractor(nn.Module):
    """Section 3.1: ResNet45 with TDA + 3-layer Transformer Encoder."""
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=3,
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.resnet = ResNet45TDA()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=8 * 32)
        self.transformer_encoder = TransformerEncoder(
            d_model=d_model, nhead=nhead, num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout
        )

    def forward(self, images):
        # images: (B, 3, 32, 128)
        feature = self.resnet(images)  # (B, 512, H', W') = (B, 512, 8, 32)
        n, c, h, w = feature.shape
        # Reshape to sequence: (H'*W', B, C)
        feature = feature.view(n, c, -1).permute(2, 0, 1)  # (H'*W', B, C)
        feature = self.pos_encoder(feature)
        feature = self.transformer_encoder(feature)
        # Reshape back: (B, C, H', W')
        feature = feature.permute(1, 2, 0).view(n, c, h, w)
        return feature


# ============================================================================
# Mini U-Net for Key generation (Section 3.2.1)
# ============================================================================
def _encoder_layer(in_c, out_c, k=3, s=2, p=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, s, p),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True)
    )


def _decoder_layer(in_c, out_c, k=3, s=1, p=1, scale_factor=None, size=None,
                   mode='nearest'):
    align_corners = None if mode == 'nearest' else True
    return nn.Sequential(
        nn.Upsample(size=size, scale_factor=scale_factor,
                     mode=mode, align_corners=align_corners),
        nn.Conv2d(in_c, out_c, k, s, p),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True)
    )


class MiniUNet(nn.Module):
    """Mini U-Net for generating key in the decoder.
    
    From Section 3.2.1:
    - Encoder: 4 layers. First layer kernel=3, stride=1, padding=2.
      Remaining layers kernel=3, stride=2, padding=2.
    - Decoder: 4 layers with nearest-neighbor upsampling (2x).
    - Skip connections (standard U-Net design).
    
    Uses num_channels=64 as internal channel count (from ABINet's
    PositionAttention default).
    """
    def __init__(self, in_channels=512, num_channels=64, h=8, w=32):
        super().__init__()
        # Encoder: 4 layers
        # Paper: first layer k=3, s=1, p=2; rest k=3, s=2, p=2
        # However, ABINet uses s=(1,2) for first and s=(2,2) for rest.
        # We follow ABINet's PositionAttention which matches the paper's intent.
        self.k_encoder = nn.ModuleList([
            _encoder_layer(in_channels, num_channels, k=3, s=(1, 2), p=1),
            _encoder_layer(num_channels, num_channels, k=3, s=(2, 2), p=1),
            _encoder_layer(num_channels, num_channels, k=3, s=(2, 2), p=1),
            _encoder_layer(num_channels, num_channels, k=3, s=(2, 2), p=1),
        ])
        # Decoder: 4 layers with upsample
        self.k_decoder = nn.ModuleList([
            _decoder_layer(num_channels, num_channels, scale_factor=2),
            _decoder_layer(num_channels, num_channels, scale_factor=2),
            _decoder_layer(num_channels, num_channels, scale_factor=2),
            _decoder_layer(num_channels, in_channels, size=(h, w)),
        ])

    def forward(self, x):
        # Encoder with skip connections
        features = []
        k = x
        for enc in self.k_encoder:
            k = enc(k)
            features.append(k)
        # Decoder with skip connections
        for i in range(len(self.k_decoder) - 1):
            k = self.k_decoder[i](k)
            k = k + features[len(self.k_decoder) - 2 - i]  # skip connection
        k = self.k_decoder[-1](k)
        return k


# ============================================================================
# Query Enhancement Module (Section 3.2.2)
# ============================================================================
class QueryEnhancement(nn.Module):
    """Query Enhancement module using masked self-attention.
    
    From Section 3.2.2 and Fig. 4:
    - 3 Transformer decoder layers with upper triangular mask
    - Prevents the model from seeing future information
    - Uses sinusoidal positional encoding for query initialization
    
    This is essentially a stack of self-attention blocks with causal masking.
    """
    def __init__(self, d_model=512, nhead=8, num_layers=3,
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(QELayer(d_model, nhead, dim_feedforward, dropout))

    def forward(self, q, mask=None):
        """
        Args:
            q: (seq_len, batch, d_model) - positional encoding initialized query
            mask: upper triangular mask (seq_len, seq_len)
        """
        for layer in self.layers:
            q = layer(q, mask)
        return q


class QELayer(nn.Module):
    """Single layer of Query Enhancement: masked self-attention + FFN."""
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, q, mask=None):
        q2, _ = self.self_attn(q, q, q, attn_mask=mask)
        q = q + self.dropout1(q2)
        q = self.norm1(q)
        q2 = self.linear2(self.dropout(F.relu(self.linear1(q))))
        q = q + self.dropout2(q2)
        q = self.norm2(q)
        return q


# ============================================================================
# Full Model
# ============================================================================
class CaptchaTDAModel(nn.Module):
    """Complete model for text CAPTCHA recognition.
    
    Args:
        num_classes: number of character classes (including null/padding token)
        max_length: maximum sequence length for prediction (including EOS token)
        d_model: model dimension (default 512)
        nhead: number of attention heads (default 8)
        num_encoder_layers: Transformer encoder layers (default 3)
        num_qe_layers: Query Enhancement layers (default 3)
        dim_feedforward: FFN hidden dimension (default 2048)
        dropout: dropout rate (default 0.1)
    """
    def __init__(self, num_classes, max_length=26, d_model=512, nhead=8,
                 num_encoder_layers=3, num_qe_layers=3,
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.num_classes = num_classes

        # Feature extraction: ResNet45+TDA + Transformer Encoder
        self.feature_extractor = FeatureExtractor(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout
        )

        # Key generation: mini U-Net
        self.key_net = MiniUNet(in_channels=d_model, num_channels=64, h=8, w=32)

        # Query initialization: sinusoidal positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=max_length)

        # Query Enhancement module
        self.query_enhancement = QueryEnhancement(
            d_model=d_model, nhead=nhead, num_layers=num_qe_layers,
            dim_feedforward=dim_feedforward, dropout=dropout
        )

        # Query projection (from ABINet's PositionAttention)
        self.query_project = nn.Linear(d_model, d_model)

        # Classification head
        self.cls = nn.Linear(d_model, num_classes)

    def _get_upper_triangular_mask(self, sz, device):
        """Generate upper triangular mask for query enhancement.
        Prevents attending to future positions.
        """
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, 0.0)
        return mask

    def forward(self, images):
        """
        Args:
            images: (B, 3, 32, 128) input CAPTCHA images
        Returns:
            logits: (B, max_length, num_classes) character probabilities
        """
        B = images.size(0)
        device = images.device

        # 1. Feature Extraction
        feature_map = self.feature_extractor(images)  # (B, C, H, W) = (B, 512, 8, 32)
        N, C, H, W = feature_map.shape

        # 2. Generate Key via mini U-Net (Eq. 11)
        key = self.key_net(feature_map)  # (B, C, H, W)
        key = key.view(N, C, -1).permute(0, 2, 1)  # (B, H*W, C)

        # 3. Value is identity mapping of feature map (Eq. 12)
        value = feature_map.view(N, C, -1).permute(0, 2, 1)  # (B, H*W, C)

        # 4. Initialize Query with sinusoidal positional encoding (Eq. 13-16)
        zeros = feature_map.new_zeros((self.max_length, N, self.d_model))
        q = self.pos_encoder(zeros)  # (max_length, B, d_model)

        # 5. Query Enhancement with upper triangular mask
        qe_mask = self._get_upper_triangular_mask(self.max_length, device)
        q = self.query_enhancement(q, mask=qe_mask)  # (max_length, B, d_model)

        # Reshape query: (max_length, B, d_model) -> (B, max_length, d_model)
        q = q.permute(1, 0, 2)  # (B, max_length, d_model)
        q = self.query_project(q)  # (B, max_length, d_model)

        # 6. Cross-attention: Output = softmax(Q·K^T / sqrt(d_k)) · V  (Eq. 17)
        attn_scores = torch.bmm(q, key.transpose(1, 2))  # (B, max_length, H*W)
        attn_scores = attn_scores / (self.d_model ** 0.5)
        attn_scores = torch.softmax(attn_scores, dim=-1)
        attn_vecs = torch.bmm(attn_scores, value)  # (B, max_length, d_model)

        # 7. Classification
        logits = self.cls(attn_vecs)  # (B, max_length, num_classes)

        return logits
