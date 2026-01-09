"""Video Encoder with Dense Vision Transformer + Temporal Conv + Attention Pooling

CRITICAL: Sign language requires temporal modeling (motion between frames).
This encoder uses a dense Vision Transformer (ViT) for spatial feature extraction,
combined with 1D temporal convolutions to capture motion patterns crucial for 
sign language understanding.

Architecture Flow:
    Video Frames (B, T, C, H, W)
           ↓
    Dense Vision Transformer (ViT) (per-frame features)
           ↓
    Frame Features (B, T, 1024)
           ↓
    Temporal Convolutions (motion modeling) - DENSE
           ↓
    Attention Pooling (compress to fixed tokens) - DENSE
           ↓
    Video Embedding (B, num_queries, output_dim)

Configuration: DENSE ViT optimized for 6000 video dataset
- Allows overfitting to training data
- Large hidden dimensions for rich feature representation
- Deep feed-forward networks
- Multiple transformer layers for temporal modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
try:
    from timm.models import vision_transformer
except ImportError:
    vision_transformer = None


class TemporalConvBlock(nn.Module):
    """1D depthwise separable convolutions over time to capture motion patterns.
    
    Uses depthwise separable convolutions for efficiency (critical for mobile).
    Includes residual connection and pre-normalization for stable training.
    """
    
    def __init__(self, dim: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        # Pre-norm for stable training
        self.norm = nn.LayerNorm(dim)
        
        # Depthwise conv (operates on each channel independently)
        self.conv_depthwise = nn.Conv1d(
            dim, dim, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2, 
            groups=dim,  # Depthwise: each channel processed separately
            bias=False
        )
        
        # Pointwise conv (mixes channels)
        self.conv_pointwise = nn.Conv1d(dim, dim, kernel_size=1, bias=True)
        
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) - temporal sequence of features
        Returns:
            out: (B, T, D) - processed features with residual
        """
        residual = x
        
        # Pre-norm
        x = self.norm(x)
        
        # Conv expects (B, C, T) format
        x = x.transpose(1, 2)
        
        # Depthwise separable conv
        x = self.conv_depthwise(x)
        x = self.act(x)
        x = self.conv_pointwise(x)
        
        # Back to (B, T, D)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        
        return x + residual




class DenseVisionTransformer(nn.Module):
    """Dense Vision Transformer backbone for sign language video encoding.
    
    A dense ViT configuration designed for overfitting on 6000 sign language videos.
    Uses large hidden dimensions and deep networks to capture fine-grained details.
    
    Architecture:
        - Patch embedding: Converts image to sequence of patches
        - Deep transformer blocks with large hidden dims
        - Rich attention mechanisms
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        hidden_dim: int = 1024,  # DENSE: Increased from 768
        num_heads: int = 16,     # DENSE: Increased from 12
        num_layers: int = 24,    # DENSE: Increased from 12
        mlp_dim: int = 4096,     # DENSE: Increased from 3072
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        pretrained: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, hidden_dim)
        )
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Dense transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self._init_weights()
        
        print(f"[INFO] Dense ViT created: {num_layers} layers, {hidden_dim} hidden dim, {num_heads} heads")
    
    def _init_weights(self):
        """Initialize model weights."""
        # Patch embedding
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        nn.init.zeros_(self.patch_embed.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) - image frames
        
        Returns:
            features: (B, num_patches + 1, hidden_dim) - patch embeddings + cls token
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, hidden_dim, num_patches_h, num_patches_w)
        x = x.flatten(2)  # (B, hidden_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, hidden_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, hidden_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, hidden_dim)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        x = self.transformer(x)
        
        # Layer norm
        x = self.norm(x)
        
        return x  # (B, num_patches + 1, hidden_dim)


class VideoEncoder(nn.Module):
    """Dense Vision Transformer backbone + Temporal Conv + Attention Pooling.
    
    This encoder is optimized for:
    1. Sign language with temporal modeling (temporal convolutions capture motion)
    2. Fixed-length output (attention pooling compresses any length video)
    3. Overfitting to 6000 video dataset (dense architecture with large capacity)
    
    Architecture: Dense ViT backbone (~86M params) + Temporal layers
    Designed to enable full overfitting to training data for maximum accuracy.
    """
    
    def __init__(
        self,
        output_dim: int = 512,
        num_queries: int = 32,
        pretrained: bool = True,
        freeze_epochs: int = 0,  # DENSE: Don't freeze, train all from start
        num_temporal_conv: int = 4,  # DENSE: Increased from 2
        kernel_size: int = 3,
        dropout: float = 0.2,  # DENSE: Slightly higher for regularization
        # ViT parameters
        vit_hidden_dim: int = 1024,  # DENSE
        vit_num_heads: int = 16,     # DENSE
        vit_num_layers: int = 24,    # DENSE
        vit_mlp_dim: int = 4096      # DENSE
    ):
        super().__init__()
        self.freeze_epochs = freeze_epochs
        self._frozen = False  # Don't freeze for dense training
        self.num_queries = num_queries
        
        # Dense Vision Transformer backbone
        self.backbone = DenseVisionTransformer(
            img_size=224,
            patch_size=16,
            hidden_dim=vit_hidden_dim,
            num_heads=vit_num_heads,
            num_layers=vit_num_layers,
            mlp_dim=vit_mlp_dim,
            dropout=dropout,
            attention_dropout=dropout,
            pretrained=pretrained
        )
        
        self.backbone_dim = vit_hidden_dim  # Dense ViT output dimension
        
        # More powerful temporal attention pooling for DENSE architecture
        self.temporal_attn = DenseTemporalAttentionPooling(
            input_dim=self.backbone_dim,
            output_dim=output_dim,
            num_queries=num_queries,
            num_heads=16,  # DENSE: More heads
            num_temporal_conv=num_temporal_conv,
            kernel_size=kernel_size,
            dropout=dropout,
            mlp_dim=output_dim * 4  # DENSE: Larger FFN
        )
        
        print(f"[INFO] Dense VideoEncoder initialized with {self._count_params()}M parameters")
    
    def _count_params(self) -> float:
        """Count total parameters in millions."""
        return sum(p.numel() for p in self.parameters()) / 1e6
    
    def _freeze_backbone(self):
        """Freeze backbone parameters for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self._frozen = True
        print("[INFO] Backbone frozen - only temporal attention is trainable")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self._frozen = False
        print("[INFO] Backbone unfrozen - all parameters trainable")
    
    def forward(
        self, 
        video_frames: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            video_frames: (B, T, C, H, W) - video frames
                B = batch size
                T = number of frames (e.g., 16)
                C = channels (3 for RGB)
                H, W = height, width (e.g., 224)
        
        Returns:
            features: (B, num_queries, output_dim) - video embedding
            lengths: (B,) - output lengths (all same = num_queries)
        """
        B, T, C, H, W = video_frames.shape
        
        # Process all frames through ViT backbone
        # Reshape: (B, T, C, H, W) -> (B*T, C, H, W)
        frames = video_frames.view(B * T, C, H, W)
        
        # Extract ViT features (includes all patch embeddings + cls token)
        vit_out = self.backbone(frames)  # (B*T, num_patches + 1, hidden_dim)
        
        # Use cls token or pool all patches
        # We use mean pooling of all patches for better representation
        features = vit_out.mean(dim=1)  # (B*T, hidden_dim)
        
        # Reshape back: (B*T, hidden_dim) -> (B, T, hidden_dim)
        features = features.view(B, T, -1)
        
        # DENSE temporal modeling + attention pooling
        features = self.temporal_attn(features)  # (B, num_queries, output_dim)
        
        # All videos output same length (num_queries)
        lengths = torch.full(
            (B,), 
            self.num_queries, 
            dtype=torch.long, 
            device=features.device
        )
        
        return features, lengths
    
    @property
    def is_frozen(self) -> bool:
        """Check if backbone is frozen."""
        return self._frozen
    
    def get_param_count(self) -> dict:
        """Get parameter counts for each component."""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        temporal_params = sum(p.numel() for p in self.temporal_attn.parameters())
        
        return {
            'backbone': backbone_params,
            'temporal': temporal_params,
            'total': backbone_params + temporal_params
        }


class DenseTemporalAttentionPooling(nn.Module):
    """DENSE Temporal attention pooling with deeper temporal convolutions.
    
    Optimized for 6000 video dataset to enable overfitting.
    Uses more temporal layers and larger feed-forward networks.
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 512,
        num_queries: int = 32,
        num_heads: int = 16,
        num_temporal_conv: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
        mlp_dim: int = 2048
    ):
        super().__init__()
        self.num_queries = num_queries
        self.output_dim = output_dim
        
        # Learnable query tokens (learned during training)
        self.queries = nn.Parameter(
            torch.randn(num_queries, output_dim) * 0.02
        )
        
        # Project input features to output dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # DENSE: More temporal convolution blocks
        self.temporal_convs = nn.Sequential(
            *[TemporalConvBlock(output_dim, kernel_size, dropout)
              for _ in range(num_temporal_conv)]
        )
        
        # Cross-attention: queries attend to frame features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Post-attention layers
        self.attn_norm = nn.LayerNorm(output_dim)
        
        # DENSE: Larger FFN
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, output_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(output_dim)
        
        # Additional dense refinement layers
        self.refinement = nn.Sequential(
            nn.Linear(output_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) - temporal sequence of features
        Returns:
            out: (B, num_queries, output_dim) - compressed video representation
        """
        B = x.size(0)
        
        # Project input to output dimension
        x = self.input_proj(x)  # (B, T, output_dim)
        
        # Apply temporal convolutions to capture motion patterns
        x = self.temporal_convs(x)  # (B, T, output_dim)
        
        # Expand queries for batch
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, Q, D)
        
        # Cross-attention: queries attend to frame features
        attn_out, _ = self.cross_attn(queries, x, x)
        queries = self.attn_norm(queries + self.dropout(attn_out))
        
        # Feed-forward network
        ffn_out = self.ffn(queries)
        queries = self.ffn_norm(queries + ffn_out)
        
        # Additional refinement for DENSE architecture
        ref_out = self.refinement(queries)
        out = queries + ref_out
        
        return out
