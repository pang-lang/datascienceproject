#!/usr/bin/env python3
"""
Dual-Head VQA Model with Shared Multimodal Backbone
Handles both binary (yes/no) and open-ended questions with a unified architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import DistilBertModel
from typing import Dict, Optional, Tuple, List
import numpy as np


class AttentionFusion(nn.Module):
    """Efficient attention-based fusion of visual and textual features."""
    
    def __init__(self, 
                 visual_dim: int = 576,
                 text_dim: int = 768,
                 hidden_dim: int = 256,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Project visual and text features to same dimension
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
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
            nn.Dropout(dropout)
        )
        
    def forward(self, visual_features, text_features):
        """
        Args:
            visual_features: (batch, visual_dim)
            text_features: (batch, text_dim)
        Returns:
            fused_features: (batch, hidden_dim)
        """
        # Project to same dimension
        v = self.visual_proj(visual_features)  # (batch, hidden_dim)
        t = self.text_proj(text_features)      # (batch, hidden_dim)
        
        # Add sequence dimension for attention
        v = v.unsqueeze(1)  # (batch, 1, hidden_dim)
        t = t.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Concatenate for attention
        combined = torch.cat([v, t], dim=1)  # (batch, 2, hidden_dim)
        
        # Multi-head attention
        attn_out, attn_weights = self.multihead_attn(
            combined, combined, combined
        )
        
        # Residual connection and normalization
        combined = self.norm1(combined + attn_out)
        
        # Feed-forward network
        ffn_out = self.ffn(combined)
        combined = self.norm2(combined + ffn_out)
        
        # Global average pooling across sequence
        fused = combined.mean(dim=1)  # (batch, hidden_dim)
        
        return fused, attn_weights


class DualHeadVQAModel(nn.Module):
    """
    Dual-Head VQA Model with Shared Multimodal Backbone.
    
    Architecture:
    - Shared Vision Encoder: MobileNetV3-Small (pretrained)
    - Shared Text Encoder: DistilBERT (pretrained)
    - Shared Multimodal Fusion: Attention-based fusion
    - Binary Head: Yes/No classification (2 classes)
    - Open-Ended Head: Multi-class classification (N classes)
    
    This architecture allows efficient training on both question types
    while sharing the heavy lifting of feature extraction and fusion.
    """
    
    def __init__(self,
                 num_open_ended_classes: int,
                 visual_feature_dim: int = 576,
                 text_feature_dim: int = 768,
                 fusion_hidden_dim: int = 256,
                 num_attention_heads: int = 4,
                 dropout: float = 0.3,
                 freeze_vision_encoder: bool = False,
                 freeze_text_encoder: bool = False):
        """
        Args:
            num_open_ended_classes: Number of classes for open-ended questions
            visual_feature_dim: Dimension of visual features (576 for MobileNetV3-Small)
            text_feature_dim: Dimension of text features (768 for DistilBERT)
            fusion_hidden_dim: Hidden dimension for fusion module
            num_attention_heads: Number of attention heads
            dropout: Dropout probability
            freeze_vision_encoder: Whether to freeze vision encoder
            freeze_text_encoder: Whether to freeze text encoder
        """
        super().__init__()
        
        self.num_open_ended_classes = num_open_ended_classes
        self.fusion_hidden_dim = fusion_hidden_dim
        
        # ============= Shared Vision Encoder: MobileNetV3-Small =============
        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        
        # Remove classifier, keep feature extractor
        self.vision_encoder = nn.Sequential(*list(mobilenet.children())[:-1])
        
        if freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        
        # ============= Shared Text Encoder: DistilBERT =============
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # ============= Shared Multimodal Fusion =============
        self.fusion = AttentionFusion(
            visual_dim=visual_feature_dim,
            text_dim=text_feature_dim,
            hidden_dim=fusion_hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # ============= Task-Specific Heads =============
        
        # Binary Head (Yes/No questions)
        self.binary_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.LayerNorm(fusion_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim // 2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # Yes/No
        )
        
        # Open-Ended Head (Multi-class classification)
        self.open_ended_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.LayerNorm(fusion_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim // 2, num_open_ended_classes)
        )
        
        # Initialize heads
        self._init_heads()
    
    def _init_heads(self):
        """Initialize classification heads with Xavier initialization."""
        for head in [self.binary_head, self.open_ended_head]:
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
    
    def extract_visual_features(self, images):
        """Extract visual features from images."""
        features = self.vision_encoder(images)
        
        # Global average pooling
        if len(features.shape) == 4:  # (batch, channels, h, w)
            features = F.adaptive_avg_pool2d(features, 1)
            features = features.view(features.size(0), -1)
        
        return features
    
    def extract_text_features(self, input_ids, attention_mask):
        """Extract text features from questions."""
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        text_features = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        
        return text_features
    
    def get_shared_features(self, images, input_ids, attention_mask):
        """
        Extract and fuse multimodal features (shared backbone).
        
        Args:
            images: (batch, 3, 224, 224)
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
        
        Returns:
            fused_features: (batch, fusion_hidden_dim)
            attention_weights: Attention weights from fusion
        """
        visual_features = self.extract_visual_features(images)
        text_features = self.extract_text_features(input_ids, attention_mask)
        fused_features, attention_weights = self.fusion(visual_features, text_features)
        
        return fused_features, attention_weights
    
    def forward(self, 
                images: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                question_types: Optional[List[str]] = None,
                return_features: bool = False):
        """
        Forward pass through the dual-head model.
        
        Args:
            images: (batch, 3, 224, 224)
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            question_types: List of 'binary' or 'open-ended' for each sample
            return_features: Whether to return intermediate features
        
        Returns:
            If question_types is None:
                Dictionary with 'binary' and 'open_ended' logits
            If question_types is provided:
                Routed logits based on question type
        """
        # Get shared multimodal features
        fused_features, attention_weights = self.get_shared_features(
            images, input_ids, attention_mask
        )
        
        if question_types is None:
            # Return predictions from both heads
            binary_logits = self.binary_head(fused_features)
            open_ended_logits = self.open_ended_head(fused_features)
            
            result = {
                'binary': binary_logits,
                'open_ended': open_ended_logits,
            }
            
            if return_features:
                result['features'] = fused_features
                result['attention_weights'] = attention_weights
            
            return result
        
        else:
            # Route to appropriate head based on question type
            batch_size = fused_features.size(0)
            
            # Separate binary and open-ended samples
            binary_mask = torch.tensor(
                [qt == 'binary' for qt in question_types],
                dtype=torch.bool,
                device=fused_features.device
            )
            open_ended_mask = ~binary_mask
            
            # Initialize output logits
            # We need to know the max dimension between 2 (binary) and num_open_ended_classes
            max_classes = max(2, self.num_open_ended_classes)
            output_logits = torch.zeros(batch_size, max_classes, device=fused_features.device)
            
            # Process binary questions
            if binary_mask.any():
                binary_features = fused_features[binary_mask]
                binary_preds = self.binary_head(binary_features)
                output_logits[binary_mask, :2] = binary_preds
            
            # Process open-ended questions
            if open_ended_mask.any():
                open_ended_features = fused_features[open_ended_mask]
                open_ended_preds = self.open_ended_head(open_ended_features)
                output_logits[open_ended_mask, :self.num_open_ended_classes] = open_ended_preds
            
            if return_features:
                return output_logits, fused_features, attention_weights
            
            return output_logits
    
    def predict_binary(self, images, input_ids, attention_mask):
        """Predict for binary (yes/no) questions only."""
        fused_features, _ = self.get_shared_features(images, input_ids, attention_mask)
        return self.binary_head(fused_features)
    
    def predict_open_ended(self, images, input_ids, attention_mask):
        """Predict for open-ended questions only."""
        fused_features, _ = self.get_shared_features(images, input_ids, attention_mask)
        return self.open_ended_head(fused_features)
    
    def count_parameters(self):
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Count by component
        vision_params = sum(p.numel() for p in self.vision_encoder.parameters())
        text_params = sum(p.numel() for p in self.text_encoder.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        binary_head_params = sum(p.numel() for p in self.binary_head.parameters())
        open_ended_head_params = sum(p.numel() for p in self.open_ended_head.parameters())
        
        return {
            'total': total,
            'trainable': trainable,
            'vision_encoder': vision_params,
            'text_encoder': text_params,
            'fusion': fusion_params,
            'binary_head': binary_head_params,
            'open_ended_head': open_ended_head_params,
            'shared_backbone': vision_params + text_params + fusion_params
        }
    
    def get_model_size_mb(self):
        """Get model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb


def create_dual_head_model(
    num_open_ended_classes: int,
    fusion_dim: int = 256,
    num_heads: int = 4,
    dropout: float = 0.3,
    freeze_encoders: bool = False,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> DualHeadVQAModel:
    """
    Factory function to create dual-head VQA model.
    
    Args:
        num_open_ended_classes: Number of classes for open-ended questions
        fusion_dim: Hidden dimension for fusion
        num_heads: Number of attention heads
        dropout: Dropout probability
        freeze_encoders: Whether to freeze pretrained encoders
        device: Device to load model on
    
    Returns:
        model: Initialized dual-head VQA model
    """
    model = DualHeadVQAModel(
        num_open_ended_classes=num_open_ended_classes,
        fusion_hidden_dim=fusion_dim,
        num_attention_heads=num_heads,
        dropout=dropout,
        freeze_vision_encoder=freeze_encoders,
        freeze_text_encoder=freeze_encoders
    )
    
    model = model.to(device)
    return model


def print_model_summary(model: DualHeadVQAModel):
    """Print detailed model summary."""
    print("=" * 80)
    print("DUAL-HEAD VQA MODEL SUMMARY")
    print("=" * 80)
    
    params = model.count_parameters()
    print(f"\nParameter Counts:")
    print(f"  Shared Backbone:")
    print(f"    Vision Encoder:    {params['vision_encoder']:>12,} params")
    print(f"    Text Encoder:      {params['text_encoder']:>12,} params")
    print(f"    Fusion Module:     {params['fusion']:>12,} params")
    print(f"    Subtotal:          {params['shared_backbone']:>12,} params")
    print(f"\n  Task-Specific Heads:")
    print(f"    Binary Head:       {params['binary_head']:>12,} params")
    print(f"    Open-Ended Head:   {params['open_ended_head']:>12,} params")
    print(f"  " + "-" * 50)
    print(f"  Total:               {params['total']:>12,} params")
    print(f"  Trainable:           {params['trainable']:>12,} params")
    
    trainable_pct = (params['trainable'] / params['total']) * 100
    print(f"  Trainable %:         {trainable_pct:>12.2f}%")
    
    backbone_pct = (params['shared_backbone'] / params['total']) * 100
    print(f"  Shared Backbone %:   {backbone_pct:>12.2f}%")
    
    size_mb = model.get_model_size_mb()
    print(f"\nModel Size: {size_mb:.2f} MB")
    
    print("=" * 80)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Dual-Head VQA Model Architecture\n")
    
    # Create model
    print("Creating Dual-Head Model")
    print("-" * 80)
    model = create_dual_head_model(
        num_open_ended_classes=500,  # VQA-RAD typical vocab size
        fusion_dim=256,
        num_heads=4,
        dropout=0.3,
        device='cpu'
    )
    print_model_summary(model)
    
    # Test with dummy input
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_input_ids = torch.randint(0, 30522, (batch_size, 40))
    dummy_attention_mask = torch.ones(batch_size, 40)
    
    print("\n\nTest 1: Dual Prediction (both heads)")
    print("-" * 80)
    with torch.no_grad():
        outputs = model(dummy_images, dummy_input_ids, dummy_attention_mask)
        print(f"Binary output shape:      {outputs['binary'].shape}")
        print(f"Open-ended output shape:  {outputs['open_ended'].shape}")
    
    print("\n\nTest 2: Mixed Batch Routing")
    print("-" * 80)
    question_types = ['binary', 'open-ended', 'binary', 'open-ended']
    with torch.no_grad():
        routed_output = model(
            dummy_images, dummy_input_ids, dummy_attention_mask,
            question_types=question_types
        )
        print(f"Routed output shape: {routed_output.shape}")
    
    print("\n\nTest 3: Binary-Only Prediction")
    print("-" * 80)
    with torch.no_grad():
        binary_output = model.predict_binary(dummy_images, dummy_input_ids, dummy_attention_mask)
        print(f"Binary prediction shape: {binary_output.shape}")
    
    print("\n\nTest 4: Open-Ended-Only Prediction")
    print("-" * 80)
    with torch.no_grad():
        open_ended_output = model.predict_open_ended(dummy_images, dummy_input_ids, dummy_attention_mask)
        print(f"Open-ended prediction shape: {open_ended_output.shape}")
    
    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)

