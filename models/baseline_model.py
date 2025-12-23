#!/usr/bin/env python3
"""
Dual-Head Baseline VQA Model: ResNet-34 + BERT-base
Based on train_dual_head_v6.py architecture applied to baseline model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import BertModel
from typing import Dict


class AttentionFusion(nn.Module):
    """Attention-based fusion of visual and textual features."""

    def __init__(
        self,
        visual_dim: int = 512,       # ResNet-34 pooled output
        text_dim: int = 768,         # BERT-base hidden size
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.35,
    ):
        super().__init__()

        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Project visual and text features to same dimension
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Multi-head self-attention over {visual, text}
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, visual_features, text_features):
        """
        Args:
            visual_features: (batch, visual_dim)
            text_features:   (batch, text_dim)
        Returns:
            fused_features:  (batch, hidden_dim)
            attn_weights:    (batch, num_heads, seq_len, seq_len) with seq_len=2
        """
        v = self.visual_proj(visual_features).unsqueeze(1)  # (B, 1, H)
        t = self.text_proj(text_features).unsqueeze(1)      # (B, 1, H)

        # Sequence of 2 tokens: [vision, text]
        combined = torch.cat([v, t], dim=1)  # (B, 2, H)

        # Self-attention
        attn_out, attn_weights = self.multihead_attn(
            combined, combined, combined
        )

        # Residual + norm
        combined = self.norm1(combined + attn_out)

        # FFN + residual + norm
        ffn_out = self.ffn(combined)
        combined = self.norm2(combined + ffn_out)

        # Global average pooling along sequence dimension
        fused = combined.mean(dim=1)  # (B, H)

        return fused, attn_weights


class DualHeadBaselineVQAModel(nn.Module):
    """
    Dual-Head Baseline VQA Model with ResNet-34 + BERT-base.
    
    Based on v6 dual-head architecture:
    - Shared encoders (ResNet-34 for vision, BERT-base for text)
    - Shared attention fusion module
    - Separate specialized heads:
        * Binary head: Optimized for yes/no questions
        * Open-ended head: For other answer types
    
    Key improvements from v6:
    - Increased capacity (fusion_hidden_dim=512)
    - Higher dropout (0.35)
    - Better attention mechanism (8 heads)
    """

    def __init__(
        self,
        num_open_ended_classes: int,
        visual_feature_dim: int = 512,     # ResNet-34 output
        text_feature_dim: int = 768,       # BERT-base hidden size
        fusion_hidden_dim: int = 512,      # Increased from v6
        num_attention_heads: int = 8,      # Same as v6
        dropout: float = 0.35,             # Increased from v6
        freeze_vision_encoder: bool = False,
        freeze_text_encoder: bool = False,
    ):
        super().__init__()

        self.num_open_ended_classes = num_open_ended_classes
        self.fusion_hidden_dim = fusion_hidden_dim

        # ============= Vision Encoder: ResNet-34 =============
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        # Remove the final FC layer, keep up to global pooling
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-1])  # (B, 512, 1, 1)

        if freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        # ============= Text Encoder: BERT-base =============
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")

        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # ============= Multimodal Fusion (Shared) =============
        self.fusion = AttentionFusion(
            visual_dim=visual_feature_dim,
            text_dim=text_feature_dim,
            hidden_dim=fusion_hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
        )

        # ============= Binary Head (Yes/No) =============
        # Lightweight but effective for binary classification
        self.binary_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 2),  # Yes/No
        )

        # ============= Open-Ended Head =============
        # More capacity for diverse answer classification
        self.open_ended_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.LayerNorm(fusion_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(fusion_hidden_dim // 2, num_open_ended_classes),
        )

        self._init_classifier_weights()

    def _init_classifier_weights(self):
        """Initialize classifier heads with Xavier initialization."""
        for head in [self.binary_head, self.open_ended_head]:
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

    def extract_visual_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features from ResNet-34.
        
        Args:
            images: (batch, 3, 224, 224)
        Returns:
            features: (batch, 512)
        """
        features = self.vision_encoder(images)     # (B, 512, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 512)
        return features

    def extract_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract text features from BERT-base.
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
        Returns:
            text_features: (batch, 768) from [CLS]
        """
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # [CLS] representation
        text_features = outputs.last_hidden_state[:, 0, :]  # (B, 768)
        return text_features

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through dual-head model.

        Args:
            images: (batch, 3, 224, 224)
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            return_attention: whether to return attention weights

        Returns:
            Dictionary with:
                - 'binary': (batch, 2) logits for yes/no
                - 'open_ended': (batch, num_classes) logits for open-ended
                - 'features': (batch, fusion_hidden_dim) shared features
                - 'attention' (optional): attention weights
        """
        # Extract features
        visual_features = self.extract_visual_features(images)
        text_features = self.extract_text_features(input_ids, attention_mask)

        # Fuse features
        fused_features, attention_weights = self.fusion(
            visual_features, text_features
        )

        # Get predictions from both heads
        binary_logits = self.binary_head(fused_features)
        open_ended_logits = self.open_ended_head(fused_features)

        outputs = {
            "binary": binary_logits,
            "open_ended": open_ended_logits,
            "features": fused_features,
        }

        if return_attention:
            outputs["attention"] = attention_weights

        return outputs

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and total parameters for each component."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())

        vision_params = sum(p.numel() for p in self.vision_encoder.parameters())
        text_params = sum(p.numel() for p in self.text_encoder.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        binary_params = sum(p.numel() for p in self.binary_head.parameters())
        open_ended_params = sum(p.numel() for p in self.open_ended_head.parameters())

        return {
            "total": total,
            "trainable": trainable,
            "vision_encoder": vision_params,
            "text_encoder": text_params,
            "fusion": fusion_params,
            "binary_head": binary_params,
            "open_ended_head": open_ended_params,
        }

    def get_model_size_mb(self) -> float:
        """Approximate model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb


def print_model_summary(model: nn.Module):
    """Print detailed model summary."""
    print("=" * 80)
    print("DUAL-HEAD BASELINE MODEL SUMMARY (ResNet-34 + BERT-base)")
    print("=" * 80)

    if hasattr(model, "count_parameters"):
        params = model.count_parameters()
        print(f"\nParameter Counts:")
        print(f"  Vision Encoder (ResNet-34):  {params['vision_encoder']:>12,} params")
        print(f"  Text Encoder (BERT-base):    {params['text_encoder']:>12,} params")
        print(f"  Fusion Module:               {params['fusion']:>12,} params")
        print(f"  Binary Head:                 {params['binary_head']:>12,} params")
        print(f"  Open-Ended Head:             {params['open_ended_head']:>12,} params")
        print("  " + "-" * 50)
        print(f"  Total:                       {params['total']:>12,} params")
        print(f"  Trainable:                   {params['trainable']:>12,} params")
        trainable_pct = (params["trainable"] / params["total"]) * 100
        print(f"  Trainable %:                 {trainable_pct:>12.2f}%")

    if hasattr(model, "get_model_size_mb"):
        size_mb = model.get_model_size_mb()
        print(f"\nModel Size: {size_mb:.2f} MB")

    print("\n" + "=" * 80)


# ================= Example usage and testing =================
if __name__ == "__main__":
    print("Testing Dual-Head Baseline VQA Model (ResNet-34 + BERT-base)\n")

    # Create model
    model = DualHeadBaselineVQAModel(
        num_open_ended_classes=120,
        fusion_hidden_dim=512,
        num_attention_heads=8,
        dropout=0.35,
        freeze_vision_encoder=False,
        freeze_text_encoder=False,
    )
    
    print_model_summary(model)

    # Dummy input
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_input_ids = torch.randint(0, 30522, (batch_size, 40))  # BERT vocab size
    dummy_attention_mask = torch.ones(batch_size, 40, dtype=torch.long)

    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        outputs = model(dummy_images, dummy_input_ids, dummy_attention_mask, return_attention=True)
        
        print(f"Binary head output shape: {outputs['binary'].shape}")
        print(f"Expected: (batch_size={batch_size}, num_classes=2)")
        
        print(f"Open-ended head output shape: {outputs['open_ended'].shape}")
        print(f"Expected: (batch_size={batch_size}, num_classes=120)")
        
        print(f"Shared features shape: {outputs['features'].shape}")
        print(f"Expected: (batch_size={batch_size}, fusion_hidden_dim=512)")
        
        print(f"Attention weights shape: {outputs['attention'].shape}")

    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)

