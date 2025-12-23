#!/usr/bin/env python3
"""
FIXED Training Script for Dual-Head VQA Model
Immediate Action Plan Implementation

Key Fixes:
1. Conservative medical-safe augmentation
2. Reset binary_loss_weight to 1.0
3. Removed class weighting for balanced binary
4. Reduced focal_gamma and label_smoothing
5. Increased model capacity
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import json

from preprocessing.combined_preprocessing import create_combined_data_loaders, check_unk_rate
from models.lightweight_model import DualHeadVQAModel, print_model_summary


class SimplifiedFocalLoss(nn.Module):
    """Focal loss WITHOUT alpha weighting for balanced classes."""
    
    def __init__(self, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        num_classes = inputs.size(-1)
        
        # Label smoothing
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.zeros_like(inputs)
                smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            
            log_probs = F.log_softmax(inputs, dim=-1)
            ce_loss = -(smooth_targets * log_probs).sum(dim=-1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Focal component (NO alpha weighting)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing."""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-7, max_lr=5e-5):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.current_epoch = 0
    
    def step(self, epoch=None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            lr = self.min_lr + (self.max_lr - self.min_lr) * (self.current_epoch / self.warmup_epochs)
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class EarlyStopping:
    """Enhanced early stopping."""
    
    def __init__(self, patience=15, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, val_metric, epoch):
        score = val_metric if self.mode == 'max' else -val_metric
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        
        return self.early_stop


class FixedDualHeadTrainer:
    """Fixed trainer with all improvements applied."""
    
    def __init__(self,
                 num_open_ended_classes: int,
                 batch_size: int = 32,
                 learning_rate: float = 5e-5,
                 num_epochs: int = 50,
                 device: str = None,
                 checkpoint_dir: str = 'checkpoints/dual_head_fixed',
                 fusion_hidden_dim: int = 512,
                 num_attention_heads: int = 8,
                 dropout: float = 0.35,
                 focal_gamma: float = 2.0,
                 weight_decay: float = 0.01,
                 early_stopping_patience: int = 15,
                 binary_loss_weight: float = 1.0,
                 open_ended_loss_weight: float = 1.0,
                 label_smoothing: float = 0.05,
                 warmup_epochs: int = 5):
        
        self.num_open_ended_classes = num_open_ended_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        
        self.fusion_hidden_dim = fusion_hidden_dim
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.focal_gamma = focal_gamma
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.warmup_epochs = warmup_epochs
        
        self.binary_loss_weight = binary_loss_weight
        self.open_ended_loss_weight = open_ended_loss_weight
        
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.setup_logging()
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.warmup_scheduler = None
        self.binary_criterion = None
        self.open_ended_criterion = None
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        self.history = {
            'train_loss': [], 'train_binary_loss': [], 'train_open_ended_loss': [],
            'train_binary_acc': [], 'train_open_ended_acc': [], 'train_overall_acc': [],
            'val_loss': [], 'val_binary_loss': [], 'val_open_ended_loss': [],
            'val_binary_acc': [], 'val_open_ended_acc': [], 'val_overall_acc': [],
            'learning_rates': []
        }
        
        self.best_val_acc = 0.0
        self.best_binary_acc = 0.0
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = os.path.join(self.checkpoint_dir, 'training.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ],
            force=True
        )
        self.logger = logging.getLogger(__name__)
    
    def diagnose_binary_distribution(self, train_loader: DataLoader):
        """Analyze binary question distribution."""
        self.logger.info("\n" + "="*80)
        self.logger.info("BINARY DISTRIBUTION ANALYSIS")
        self.logger.info("="*80)
        
        binary_answers = {'yes', 'no'}
        yes_count = 0
        no_count = 0
        total_binary = 0
        
        for batch in train_loader:
            answers = batch['answer']['text']
            for ans in answers:
                if ans in binary_answers:
                    total_binary += 1
                    if ans == 'yes':
                        yes_count += 1
                    else:
                        no_count += 1
        
        yes_pct = 100 * yes_count / total_binary if total_binary > 0 else 0
        no_pct = 100 * no_count / total_binary if total_binary > 0 else 0
        
        self.logger.info(f"Total binary questions: {total_binary}")
        self.logger.info(f"  Yes: {yes_count} ({yes_pct:.1f}%)")
        self.logger.info(f"  No: {no_count} ({no_pct:.1f}%)")
        
        if abs(yes_pct - 50) < 5:
            self.logger.info("✅ Distribution is balanced - NO class weighting needed!")
        else:
            self.logger.info("⚠️ Distribution is imbalanced - consider class weighting")
        
        self.logger.info("="*80)
    
    def initialize_model(self, train_loader: DataLoader, answer_vocab: dict):
        """Initialize model, optimizer, and loss functions."""
        self.logger.info("\n" + "="*80)
        self.logger.info("FIXED DUAL-HEAD VQA MODEL TRAINING")
        self.logger.info("="*80)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Binary loss weight: {self.binary_loss_weight} (RESET TO EQUAL)")
        self.logger.info(f"Using SIMPLIFIED focal loss (NO class weights)")
        self.logger.info("="*80)
        
        # Analyze binary distribution
        self.diagnose_binary_distribution(train_loader)
        
        # Create model with INCREASED capacity
        self.model = DualHeadVQAModel(
            num_open_ended_classes=self.num_open_ended_classes,
            fusion_hidden_dim=self.fusion_hidden_dim,
            num_attention_heads=self.num_attention_heads,
            dropout=self.dropout,
            freeze_vision_encoder=False,
            freeze_text_encoder=False
        ).to(self.device)
        
        print_model_summary(self.model)
        
        # SIMPLIFIED loss functions - NO class weights for balanced binary
        self.binary_criterion = SimplifiedFocalLoss(
            gamma=self.focal_gamma,
            label_smoothing=self.label_smoothing
        )
        
        self.open_ended_criterion = SimplifiedFocalLoss(
            gamma=self.focal_gamma,
            label_smoothing=self.label_smoothing
        )
        
        # Optimizer with REDUCED learning rate
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay,
            eps=1e-8
        )
        
        # Warmup + Cosine scheduler
        self.warmup_scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=self.warmup_epochs,
            total_epochs=self.num_epochs,
            min_lr=1e-7,
            max_lr=self.learning_rate
        )
        
        # Plateau scheduler as backup
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        self.logger.info("✅ Model initialized successfully!")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        binary_loss_sum = 0.0
        open_ended_loss_sum = 0.0
        binary_correct = 0
        binary_total = 0
        open_ended_correct = 0
        open_ended_total = 0
        
        binary_answers = {'yes', 'no'}
        
        # Update learning rate
        if epoch <= self.warmup_epochs:
            current_lr = self.warmup_scheduler.step(epoch)
        else:
            current_lr = self.optimizer.param_groups[0]['lr']
        
        progress = tqdm(train_loader, desc=f'Epoch {epoch} Training')
        
        for batch in progress:
            images = batch['image'].to(self.device)
            input_ids = batch['question']['input_ids'].to(self.device)
            attention_mask = batch['question']['attention_mask'].to(self.device)
            answer_texts = batch['answer']['text']
            answer_indices = batch['answer_idx'].to(self.device)
            batch_size = images.size(0)
            
            is_binary = torch.tensor(
                [ans in binary_answers for ans in answer_texts],
                dtype=torch.bool,
                device=self.device
            )
            is_open_ended = ~is_binary
            
            self.optimizer.zero_grad()
            outputs = self.model(images, input_ids, attention_mask)
            
            loss = 0
            batch_binary_loss = 0
            batch_open_ended_loss = 0
            
            # Binary loss
            if is_binary.any():
                binary_logits = outputs['binary'][is_binary]
                binary_targets = torch.tensor(
                    [1 if answer_texts[i] == 'yes' else 0 
                     for i in range(batch_size) if is_binary[i]],
                    dtype=torch.long,
                    device=self.device
                )
                binary_loss = self.binary_criterion(binary_logits, binary_targets)
                loss = loss + self.binary_loss_weight * binary_loss
                batch_binary_loss = binary_loss.item()
                
                binary_preds = torch.argmax(binary_logits, dim=1)
                binary_correct += (binary_preds == binary_targets).sum().item()
                binary_total += binary_targets.size(0)
            
            # Open-ended loss
            if is_open_ended.any():
                open_ended_logits = outputs['open_ended'][is_open_ended]
                open_ended_targets = answer_indices[is_open_ended]
                open_ended_loss = self.open_ended_criterion(open_ended_logits, open_ended_targets)
                loss = loss + self.open_ended_loss_weight * open_ended_loss
                batch_open_ended_loss = open_ended_loss.item()
                
                open_ended_preds = torch.argmax(open_ended_logits, dim=1)
                open_ended_correct += (open_ended_preds == open_ended_targets).sum().item()
                open_ended_total += open_ended_targets.size(0)
            
            if loss != 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                binary_loss_sum += batch_binary_loss
                open_ended_loss_sum += batch_open_ended_loss
            
            progress.set_postfix({
                'loss': f'{loss.item():.4f}' if loss != 0 else '0.0000',
                'bin': f'{100 * binary_correct / max(binary_total, 1):.1f}%',
                'oe': f'{100 * open_ended_correct / max(open_ended_total, 1):.1f}%',
                'lr': f'{current_lr:.2e}'
            })
        
        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches
        avg_binary_loss = binary_loss_sum / num_batches
        avg_open_ended_loss = open_ended_loss_sum / num_batches
        binary_acc = 100 * binary_correct / max(binary_total, 1)
        open_ended_acc = 100 * open_ended_correct / max(open_ended_total, 1)
        overall_acc = 100 * (binary_correct + open_ended_correct) / max(binary_total + open_ended_total, 1)
        
        return {
            'loss': avg_loss,
            'binary_loss': avg_binary_loss,
            'open_ended_loss': avg_open_ended_loss,
            'binary_acc': binary_acc,
            'open_ended_acc': open_ended_acc,
            'overall_acc': overall_acc
        }
    
    def validate(self, val_loader: DataLoader):
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        binary_loss_sum = 0.0
        open_ended_loss_sum = 0.0
        binary_correct = 0
        binary_total = 0
        open_ended_correct = 0
        open_ended_total = 0
        
        binary_answers = {'yes', 'no'}
        
        with torch.no_grad():
            progress = tqdm(val_loader, desc='Validation')
            for batch in progress:
                images = batch['image'].to(self.device)
                input_ids = batch['question']['input_ids'].to(self.device)
                attention_mask = batch['question']['attention_mask'].to(self.device)
                answer_texts = batch['answer']['text']
                answer_indices = batch['answer_idx'].to(self.device)
                batch_size = images.size(0)
                
                is_binary = torch.tensor(
                    [ans in binary_answers for ans in answer_texts],
                    dtype=torch.bool,
                    device=self.device
                )
                is_open_ended = ~is_binary
                
                outputs = self.model(images, input_ids, attention_mask)
                
                loss = 0
                batch_binary_loss = 0
                batch_open_ended_loss = 0
                
                if is_binary.any():
                    binary_logits = outputs['binary'][is_binary]
                    binary_targets = torch.tensor(
                        [1 if answer_texts[i] == 'yes' else 0 
                         for i in range(batch_size) if is_binary[i]],
                        dtype=torch.long,
                        device=self.device
                    )
                    binary_loss = self.binary_criterion(binary_logits, binary_targets)
                    loss = loss + self.binary_loss_weight * binary_loss
                    batch_binary_loss = binary_loss.item()
                    
                    binary_preds = torch.argmax(binary_logits, dim=1)
                    binary_correct += (binary_preds == binary_targets).sum().item()
                    binary_total += binary_targets.size(0)
                
                if is_open_ended.any():
                    open_ended_logits = outputs['open_ended'][is_open_ended]
                    open_ended_targets = answer_indices[is_open_ended]
                    open_ended_loss = self.open_ended_criterion(open_ended_logits, open_ended_targets)
                    loss = loss + self.open_ended_loss_weight * open_ended_loss
                    batch_open_ended_loss = open_ended_loss.item()
                    
                    open_ended_preds = torch.argmax(open_ended_logits, dim=1)
                    open_ended_correct += (open_ended_preds == open_ended_targets).sum().item()
                    open_ended_total += open_ended_targets.size(0)
                
                if loss != 0:
                    total_loss += loss.item()
                    binary_loss_sum += batch_binary_loss
                    open_ended_loss_sum += batch_open_ended_loss
                
                progress.set_postfix({
                    'loss': f'{loss.item():.4f}' if loss != 0 else '0.0000',
                    'bin': f'{100 * binary_correct / max(binary_total, 1):.1f}%',
                    'oe': f'{100 * open_ended_correct / max(open_ended_total, 1):.1f}%'
                })
        
        num_batches = len(val_loader)
        avg_loss = total_loss / num_batches
        avg_binary_loss = binary_loss_sum / num_batches
        avg_open_ended_loss = open_ended_loss_sum / num_batches
        binary_acc = 100 * binary_correct / max(binary_total, 1)
        open_ended_acc = 100 * open_ended_correct / max(open_ended_total, 1)
        overall_acc = 100 * (binary_correct + open_ended_correct) / max(binary_total + open_ended_total, 1)
        
        return {
            'loss': avg_loss,
            'binary_loss': avg_binary_loss,
            'open_ended_loss': avg_open_ended_loss,
            'binary_acc': binary_acc,
            'open_ended_acc': open_ended_acc,
            'overall_acc': overall_acc
        }
    
    def save_checkpoint(self, epoch: int, val_metrics: dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metrics': val_metrics,
            'history': self.history,
            'config': {
                'num_open_ended_classes': self.num_open_ended_classes,
                'fusion_hidden_dim': self.fusion_hidden_dim,
                'num_attention_heads': self.num_attention_heads,
                'dropout': self.dropout,
                'focal_gamma': self.focal_gamma,
                'label_smoothing': self.label_smoothing,
                'weight_decay': self.weight_decay,
                'binary_loss_weight': self.binary_loss_weight,
                'open_ended_loss_weight': self.open_ended_loss_weight
            }
        }
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"🌟 NEW BEST! Overall: {val_metrics['overall_acc']:.2f}%, Binary: {val_metrics['binary_acc']:.2f}%")
        
        latest_path = os.path.join(self.checkpoint_dir, 'latest_model.pt')
        torch.save(checkpoint, latest_path)
    
    def plot_training_history(self):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Overall loss
        axes[0, 0].plot(epochs, self.history['train_loss'], label='Train', linewidth=2, marker='o', markersize=3)
        axes[0, 0].plot(epochs, self.history['val_loss'], label='Val', linewidth=2, marker='s', markersize=3)
        axes[0, 0].set_title('Overall Loss', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlabel('Epoch')
        
        # Binary loss
        axes[0, 1].plot(epochs, self.history['train_binary_loss'], label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_binary_loss'], label='Val', linewidth=2)
        axes[0, 1].set_title('Binary Loss', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlabel('Epoch')
        
        # Open-ended loss
        axes[0, 2].plot(epochs, self.history['train_open_ended_loss'], label='Train', linewidth=2)
        axes[0, 2].plot(epochs, self.history['val_open_ended_loss'], label='Val', linewidth=2)
        axes[0, 2].set_title('Open-Ended Loss', fontsize=12, fontweight='bold')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_xlabel('Epoch')
        
        # Overall accuracy
        axes[1, 0].plot(epochs, self.history['train_overall_acc'], label='Train', linewidth=2)
        axes[1, 0].plot(epochs, self.history['val_overall_acc'], label='Val', linewidth=2)
        axes[1, 0].set_title('Overall Accuracy (%)', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlabel('Epoch')
        
        # Binary accuracy - HIGHLIGHTED
        axes[1, 1].plot(epochs, self.history['train_binary_acc'], label='Train', linewidth=2.5, color='green')
        axes[1, 1].plot(epochs, self.history['val_binary_acc'], label='Val', linewidth=2.5, color='red')
        axes[1, 1].axhline(y=70, color='blue', linestyle='--', alpha=0.5, label='Target 70%')
        axes[1, 1].set_title('Binary Accuracy (%) - PRIORITY', fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlabel('Epoch')
        
        # Open-ended accuracy
        axes[1, 2].plot(epochs, self.history['train_open_ended_acc'], label='Train', linewidth=2)
        axes[1, 2].plot(epochs, self.history['val_open_ended_acc'], label='Val', linewidth=2)
        axes[1, 2].set_title('Open-Ended Accuracy (%)', fontsize=12, fontweight='bold')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_xlabel('Epoch')
        
        plt.tight_layout()
        plot_path = os.path.join(self.checkpoint_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📊 Training curves saved to: {plot_path}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop."""
        start_time = datetime.now()
        
        self.logger.info("\n" + "="*80)
        self.logger.info("TRAINING START - FIXED VERSION")
        self.logger.info("="*80)
        
        for epoch in range(1, self.num_epochs + 1):
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"EPOCH {epoch}/{self.num_epochs}")
            self.logger.info(f"{'='*80}")
            
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            self.logger.info(f"✓ Train - Loss: {train_metrics['loss']:.4f}")
            self.logger.info(f"  Overall: {train_metrics['overall_acc']:.2f}%")
            self.logger.info(f"  🎯 Binary: {train_metrics['binary_acc']:.2f}%")
            self.logger.info(f"  Open-Ended: {train_metrics['open_ended_acc']:.2f}%")
            
            # Validation
            val_metrics = self.validate(val_loader)
            self.logger.info(f"✓ Val - Loss: {val_metrics['loss']:.4f}")
            self.logger.info(f"  Overall: {val_metrics['overall_acc']:.2f}%")
            self.logger.info(f"  🎯 Binary: {val_metrics['binary_acc']:.2f}%")
            self.logger.info(f"  Open-Ended: {val_metrics['open_ended_acc']:.2f}%")
            
            # Update learning rate
            if epoch > self.warmup_epochs:
                self.scheduler.step(val_metrics['overall_acc'])
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"📉 Learning rate: {current_lr:.2e}")
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_binary_loss'].append(train_metrics['binary_loss'])
            self.history['train_open_ended_loss'].append(train_metrics['open_ended_loss'])
            self.history['train_binary_acc'].append(train_metrics['binary_acc'])
            self.history['train_open_ended_acc'].append(train_metrics['open_ended_acc'])
            self.history['train_overall_acc'].append(train_metrics['overall_acc'])
            
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_binary_loss'].append(val_metrics['binary_loss'])
            self.history['val_open_ended_loss'].append(val_metrics['open_ended_loss'])
            self.history['val_binary_acc'].append(val_metrics['binary_acc'])
            self.history['val_open_ended_acc'].append(val_metrics['open_ended_acc'])
            self.history['val_overall_acc'].append(val_metrics['overall_acc'])
            self.history['learning_rates'].append(current_lr)
            
            # Save best model
            is_best = val_metrics['overall_acc'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['overall_acc']
                self.best_binary_acc = val_metrics['binary_acc']
                self.save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                self.save_checkpoint(epoch, val_metrics, is_best=False)
            
            # Early stopping
            if self.early_stopping(val_metrics['overall_acc'], epoch):
                self.logger.info(f"\n⚠️ Early stopping at epoch {epoch}")
                break
        
        # Training completed
        duration = (datetime.now() - start_time).total_seconds()
        self.plot_training_history()
        
        self.logger.info("\n" + "="*80)
        self.logger.info("🎉 TRAINING COMPLETED!")
        self.logger.info("="*80)
        self.logger.info(f"Time: {duration/60:.2f} minutes")
        self.logger.info(f"Best Overall Val Acc: {self.best_val_acc:.2f}%")
        self.logger.info(f"Best Binary Val Acc: {self.best_binary_acc:.2f}%")
        self.logger.info("="*80)


def main():
    """Main training function with improved parameters."""
    
    params = {
        'batch_size': 32,
        'learning_rate': 5e-5,
        'num_epochs': 50,                  
        'max_answer_vocab_size': 120,       
        'fusion_hidden_dim': 512,
        'num_attention_heads': 8,
        'dropout': 0.35,             
        'focal_gamma': 2.0,       
        'weight_decay': 0.01,              
        'early_stopping_patience': 15,       
        'binary_loss_weight': 1.0,         
        'open_ended_loss_weight': 1.0,
        'label_smoothing': 0.05,              
        'warmup_epochs': 5                   
    }
    
    print("=" * 80)
    print("IMPROVED DUAL-HEAD VQA MODEL TRAINING")
    print("Focus: Improve Binary Accuracy to 75%+")
    print("=" * 80)
    print("\n📋 Configuration:")
    for key, value in params.items():
        marker = "🎯" if key == 'binary_loss_weight' else "  "
        print(f"{marker} {key}: {value}")
    print("=" * 80)
    
    # Validate normalization first
    print("\n🔍 Validating answer normalization...")
    train_stats = check_unk_rate(split='train')
    
    if train_stats['unk_rate'] > 1.0:
        print("❌ WARNING: High <unk> rate detected!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Load data
    print("\n📦 Loading data...")
    train_loader, val_loader, test_loader, answer_vocab, _ = create_combined_data_loaders(
        batch_size=params['batch_size'],
        max_samples=None,
        num_workers=0,
        max_answer_vocab_size=params['max_answer_vocab_size']  # Reduced vocab
    )
    
    num_answers = len(answer_vocab)
    print(f"✓ Vocabulary: {num_answers} answers (reduced from 120)")
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    # Initialize trainer
    trainer = FixedDualHeadTrainer(
        num_open_ended_classes=num_answers,
        **{k: v for k, v in params.items() if k not in ['max_answer_vocab_size']}
    )
    
    # Initialize model
    trainer.initialize_model(train_loader, answer_vocab)
    
    # Train
    trainer.train(train_loader, val_loader)
    
    print("\n✅ Training complete! Evaluate with:")
    print("  python evaluate_dual_head.py --checkpoint checkpoints/dual_head_v2/best_model.pt")


if __name__ == "__main__":
    main()