#!/usr/bin/env python3
"""
Evaluation Script for Dual-Head VQA Model
Evaluates both binary and open-ended question answering performance.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.combined_preprocessing import create_combined_data_loaders
from models.lightweight_model import DualHeadVQAModel
from models.baseline_model import DualHeadBaselineVQAModel


class DualHeadEvaluator:
    """Evaluator for dual-head VQA model (supports both lightweight and baseline)."""
    
    def __init__(self,
                 checkpoint_path: str,
                 test_loader,
                 answer_vocab: dict,
                 device='cuda',
                 output_dir='evaluation_results/dual_head',
                 model_type='auto'):
        self.checkpoint_path = Path(checkpoint_path)
        self.test_loader = test_loader
        self.answer_vocab = answer_vocab
        self.idx_to_answer = {idx: ans for ans, idx in answer_vocab.items()}
        self.device = device
        self.model_type = model_type  # 'auto', 'lightweight', or 'baseline'
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store config and history
        self.config = {}
        self.history = {}
        
        self.model = self._load_model()
        
        print(f"✓ Loaded dual-head model from {self.checkpoint_path}")
        print(f"✓ Output directory: {self.output_dir}")
    
    def _load_model(self):
        """Load model from checkpoint (auto-detects or uses specified model type)."""
        print(f"\nLoading checkpoint: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # Store config and history
        self.config = checkpoint.get('config', {})
        self.history = checkpoint.get('history', {})
        
        # Detect model type if auto
        if self.model_type == 'auto':
            # Check checkpoint path or config for hints
            checkpoint_str = str(self.checkpoint_path).lower()
            if 'baseline' in checkpoint_str:
                self.model_type = 'baseline'
                print("  🔍 Auto-detected model type: BASELINE (ResNet-34 + BERT-base)")
            elif 'lightweight' in checkpoint_str:
                self.model_type = 'lightweight'
                print("  🔍 Auto-detected model type: LIGHTWEIGHT (MobileNetV3 + DistilBERT)")
            else:
                # Default to lightweight for backward compatibility
                self.model_type = 'lightweight'
                print("  🔍 Model type not specified, defaulting to: LIGHTWEIGHT")
        else:
            model_name = "BASELINE (ResNet-34 + BERT-base)" if self.model_type == 'baseline' else "LIGHTWEIGHT (MobileNetV3 + DistilBERT)"
            print(f"  📌 Using specified model type: {model_name}")
        
        # Get model parameters
        num_classes = self.config.get('num_open_ended_classes', len(self.answer_vocab))
        fusion_hidden_dim = self.config.get('fusion_hidden_dim', 512 if self.model_type == 'baseline' else 256)
        num_attention_heads = self.config.get('num_attention_heads', 8 if self.model_type == 'baseline' else 4)
        dropout = self.config.get('dropout', 0.35 if self.model_type == 'baseline' else 0.3)
        
        # Create appropriate model
        if self.model_type == 'baseline':
            model = DualHeadBaselineVQAModel(
                num_open_ended_classes=num_classes,
                fusion_hidden_dim=fusion_hidden_dim,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
                freeze_vision_encoder=False,
                freeze_text_encoder=False
            )
        else:  # lightweight
            model = DualHeadVQAModel(
                num_open_ended_classes=num_classes,
                fusion_hidden_dim=fusion_hidden_dim,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
                freeze_vision_encoder=False,
                freeze_text_encoder=False
            )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        params = model.count_parameters()
        print(f"  Architecture: Dual-Head (Binary + Open-Ended)")
        print(f"  Model Type: {self.model_type.upper()}")
        print(f"  Total parameters: {params['total']:,}")
        
        # Handle different parameter naming
        if 'shared_backbone' in params:
            print(f"  Shared backbone: {params['shared_backbone']:,}")
        else:
            # For baseline model which has separate vision/text encoders
            vision_params = params.get('vision_encoder', 0)
            text_params = params.get('text_encoder', 0)
            fusion_params = params.get('fusion', 0)
            print(f"  Vision encoder: {vision_params:,}")
            print(f"  Text encoder: {text_params:,}")
            print(f"  Fusion module: {fusion_params:,}")
        
        print(f"  Binary head: {params['binary_head']:,}")
        print(f"  Open-ended head: {params['open_ended_head']:,}")
        print(f"  Model size: {model.get_model_size_mb():.2f} MB")
        
        return model
    
    @torch.no_grad()
    def evaluate(self, data_loader, split_name: str = "test"):
        """Evaluate the model on a dataset split."""
        print("\n" + "=" * 80)
        print(f"EVALUATING ON {split_name.upper()} SET")
        print("=" * 80)
        
        binary_answers = {'yes', 'no'}
        
        # Storage for all predictions
        all_data = {
            'binary': {
                'predictions': [],
                'targets': [],
                'questions': [],
                'true_answers': [],
                'pred_answers': []
            },
            'open_ended': {
                'predictions': [],
                'targets': [],
                'questions': [],
                'question_types': [],
                'true_answers': [],
                'pred_answers': []
            }
        }
        
        for batch in tqdm(data_loader, desc=f'Evaluating {split_name}'):
            images = batch['image'].to(self.device)
            input_ids = batch['question']['input_ids'].to(self.device)
            attention_mask = batch['question']['attention_mask'].to(self.device)
            answer_texts = batch['answer']['text']
            answer_indices = batch['answer_idx'].numpy()
            question_texts = batch['question']['text']
            question_types = batch['question_type']
            
            # Forward pass
            outputs = self.model(images, input_ids, attention_mask)
            
            # Process each sample
            for i in range(len(answer_texts)):
                ans_text = answer_texts[i]
                ans_idx = answer_indices[i]
                q_text = question_texts[i]
                q_type = question_types[i]
                
                if ans_text in binary_answers:
                    # Binary question
                    binary_logits = outputs['binary'][i]
                    binary_pred = torch.argmax(binary_logits).item()
                    binary_target = 1 if ans_text == 'yes' else 0
                    
                    all_data['binary']['predictions'].append(binary_pred)
                    all_data['binary']['targets'].append(binary_target)
                    all_data['binary']['questions'].append(q_text)
                    all_data['binary']['true_answers'].append(ans_text)
                    all_data['binary']['pred_answers'].append('yes' if binary_pred == 1 else 'no')
                
                else:
                    # Open-ended question
                    oe_logits = outputs['open_ended'][i]
                    oe_pred = torch.argmax(oe_logits).item()
                    
                    all_data['open_ended']['predictions'].append(oe_pred)
                    all_data['open_ended']['targets'].append(ans_idx)
                    all_data['open_ended']['questions'].append(q_text)
                    all_data['open_ended']['question_types'].append(q_type)
                    all_data['open_ended']['true_answers'].append(ans_text)
                    all_data['open_ended']['pred_answers'].append(
                        self.idx_to_answer.get(oe_pred, '<unk>')
                    )
        
        # Calculate metrics
        results = self._calculate_metrics(all_data)
        
        # Save artifacts
        self._save_detailed_results(all_data, split_name)
        self._plot_confusion_matrices(all_data, split_name)
        self._plot_comparison_charts(results, split_name)
        self._print_summary(results)
        
        return results
    
    def _calculate_metrics(self, all_data):
        """Calculate comprehensive metrics for both heads."""
        results = {}
        
        # Binary metrics
        if len(all_data['binary']['targets']) > 0:
            binary_preds = all_data['binary']['predictions']
            binary_targets = all_data['binary']['targets']
            
            results['binary'] = {
                'accuracy': accuracy_score(binary_targets, binary_preds),
                'samples': len(binary_targets)
            }
            
            if len(set(binary_targets)) > 1:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    binary_targets, binary_preds, average='binary', zero_division=0
                )
                results['binary']['precision'] = precision
                results['binary']['recall'] = recall
                results['binary']['f1'] = f1
        else:
            results['binary'] = {'accuracy': 0, 'samples': 0}
        
        # Open-ended metrics
        if len(all_data['open_ended']['targets']) > 0:
            oe_preds = all_data['open_ended']['predictions']
            oe_targets = all_data['open_ended']['targets']
            
            results['open_ended'] = {
                'accuracy': accuracy_score(oe_targets, oe_preds),
                'samples': len(oe_targets)
            }
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                oe_targets, oe_preds, average='macro', zero_division=0
            )
            results['open_ended']['macro_precision'] = precision
            results['open_ended']['macro_recall'] = recall
            results['open_ended']['macro_f1'] = f1
            
            # Per question type
            results['open_ended']['by_type'] = {}
            for qtype in set(all_data['open_ended']['question_types']):
                mask = [qt == qtype for qt in all_data['open_ended']['question_types']]
                if sum(mask) > 0:
                    type_preds = [p for p, m in zip(oe_preds, mask) if m]
                    type_targets = [t for t, m in zip(oe_targets, mask) if m]
                    results['open_ended']['by_type'][qtype] = {
                        'accuracy': accuracy_score(type_targets, type_preds),
                        'samples': sum(mask)
                    }
        else:
            results['open_ended'] = {'accuracy': 0, 'samples': 0}
        
        # Overall metrics
        total_correct = 0
        total_samples = 0
        
        if len(all_data['binary']['targets']) > 0:
            binary_correct = sum(
                p == t for p, t in zip(all_data['binary']['predictions'], 
                                      all_data['binary']['targets'])
            )
            total_correct += binary_correct
            total_samples += len(all_data['binary']['targets'])
        
        if len(all_data['open_ended']['targets']) > 0:
            oe_correct = sum(
                p == t for p, t in zip(all_data['open_ended']['predictions'],
                                       all_data['open_ended']['targets'])
            )
            total_correct += oe_correct
            total_samples += len(all_data['open_ended']['targets'])
        
        results['overall'] = {
            'accuracy': total_correct / max(total_samples, 1),
            'total_samples': total_samples
        }
        
        return results
    
    def _save_detailed_results(self, all_data, split_name: str):
        """Save detailed predictions to CSV."""
        # Binary results
        if len(all_data['binary']['targets']) > 0:
            binary_df = pd.DataFrame({
                'question': all_data['binary']['questions'],
                'true_answer': all_data['binary']['true_answers'],
                'predicted_answer': all_data['binary']['pred_answers'],
                'correct': [p == t for p, t in zip(
                    all_data['binary']['predictions'],
                    all_data['binary']['targets']
                )],
                'question_type': 'binary'
            })
            
            binary_path = self.output_dir / f'{split_name}_binary_detailed_{self.timestamp}.csv'
            binary_df.to_csv(binary_path, index=False)
            print(f"\n✓ Binary results saved to: {binary_path}")
        
        # Open-ended results
        if len(all_data['open_ended']['targets']) > 0:
            oe_df = pd.DataFrame({
                'question': all_data['open_ended']['questions'],
                'true_answer': all_data['open_ended']['true_answers'],
                'predicted_answer': all_data['open_ended']['pred_answers'],
                'correct': [p == t for p, t in zip(
                    all_data['open_ended']['predictions'],
                    all_data['open_ended']['targets']
                )],
                'question_type': all_data['open_ended']['question_types']
            })
            
            oe_path = self.output_dir / f'{split_name}_open_ended_detailed_{self.timestamp}.csv'
            oe_df.to_csv(oe_path, index=False)
            print(f"✓ Open-ended results saved to: {oe_path}")
        
        # Combined results
        if len(all_data['binary']['targets']) > 0 or len(all_data['open_ended']['targets']) > 0:
            combined_data = []
            
            # Add binary samples
            for i in range(len(all_data['binary']['questions'])):
                combined_data.append({
                    'question': all_data['binary']['questions'][i],
                    'true_answer': all_data['binary']['true_answers'][i],
                    'predicted_answer': all_data['binary']['pred_answers'][i],
                    'correct': all_data['binary']['predictions'][i] == all_data['binary']['targets'][i],
                    'question_type': 'binary',
                    'head_used': 'binary'
                })
            
            # Add open-ended samples
            for i in range(len(all_data['open_ended']['questions'])):
                combined_data.append({
                    'question': all_data['open_ended']['questions'][i],
                    'true_answer': all_data['open_ended']['true_answers'][i],
                    'predicted_answer': all_data['open_ended']['pred_answers'][i],
                    'correct': all_data['open_ended']['predictions'][i] == all_data['open_ended']['targets'][i],
                    'question_type': all_data['open_ended']['question_types'][i],
                    'head_used': 'open_ended'
                })
            
            combined_df = pd.DataFrame(combined_data)
            combined_path = self.output_dir / f'{split_name}_combined_detailed_{self.timestamp}.csv'
            combined_df.to_csv(combined_path, index=False)
            print(f"✓ Combined results saved to: {combined_path}")
    
    def _plot_confusion_matrices(self, all_data, split_name: str):
        """Plot confusion matrices for binary and open-ended."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Binary confusion matrix
        if len(all_data['binary']['targets']) > 0:
            binary_cm = confusion_matrix(
                all_data['binary']['targets'],
                all_data['binary']['predictions'],
                labels=[0, 1]
            )
            sns.heatmap(binary_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                       xticklabels=['no', 'yes'], yticklabels=['no', 'yes'])
            axes[0].set_xlabel('Predicted')
            axes[0].set_ylabel('True')
            axes[0].set_title('Binary Questions (Yes/No)')
        
        # Open-ended confusion matrix (top 15 classes)
        if len(all_data['open_ended']['targets']) > 0:
            from collections import Counter
            target_counts = Counter(all_data['open_ended']['targets'])
            top_classes = [cls for cls, _ in target_counts.most_common(15)]
            
            mask = [t in top_classes for t in all_data['open_ended']['targets']]
            if sum(mask) > 10:
                filtered_preds = [p for p, m in zip(all_data['open_ended']['predictions'], mask) if m]
                filtered_targets = [t for t, m in zip(all_data['open_ended']['targets'], mask) if m]
                
                oe_cm = confusion_matrix(filtered_targets, filtered_preds, labels=top_classes)
                
                class_labels = [self.idx_to_answer.get(cls, f'cls_{cls}')[:10] 
                               for cls in top_classes]
                sns.heatmap(oe_cm, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                           xticklabels=class_labels, yticklabels=class_labels)
                axes[1].set_xlabel('Predicted')
                axes[1].set_ylabel('True')
                axes[1].set_title('Open-Ended Questions (Top 15 Classes)')
                plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        cm_path = self.output_dir / f'{split_name}_confusion_matrices_{self.timestamp}.png'
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Confusion matrices saved to: {cm_path}")
    
    def _plot_comparison_charts(self, results, split_name: str):
        """Plot comparison charts between binary and open-ended performance."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy comparison
        categories = []
        accuracies = []
        
        if 'binary' in results and results['binary']['samples'] > 0:
            categories.append('Binary')
            accuracies.append(results['binary']['accuracy'] * 100)
        
        if 'open_ended' in results and results['open_ended']['samples'] > 0:
            categories.append('Open-Ended')
            accuracies.append(results['open_ended']['accuracy'] * 100)
        
        if 'overall' in results:
            categories.append('Overall')
            accuracies.append(results['overall']['accuracy'] * 100)
        
        axes[0].bar(categories, accuracies, color=['steelblue', 'seagreen', 'coral'][:len(categories)])
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_title('Accuracy Comparison')
        axes[0].set_ylim(0, 100)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        for i, (cat, acc) in enumerate(zip(categories, accuracies)):
            axes[0].text(i, acc + 2, f'{acc:.1f}%', ha='center', fontweight='bold')
        
        # Sample distribution
        if 'binary' in results and 'open_ended' in results:
            labels = ['Binary', 'Open-Ended']
            sizes = [results['binary']['samples'], results['open_ended']['samples']]
            colors = ['steelblue', 'seagreen']
            
            axes[1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                       startangle=90)
            axes[1].set_title('Question Type Distribution')
        
        plt.tight_layout()
        comp_path = self.output_dir / f'{split_name}_comparison_{self.timestamp}.png'
        plt.savefig(comp_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Comparison charts saved to: {comp_path}")
    
    def _print_summary(self, results):
        """Print evaluation summary."""
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        
        # Overall
        if 'overall' in results:
            print(f"\n📊 Overall Performance:")
            print(f"  Accuracy: {results['overall']['accuracy']:.4f} ({results['overall']['accuracy']*100:.2f}%)")
            print(f"  Total samples: {results['overall']['total_samples']}")
        
        # Binary
        if 'binary' in results and results['binary']['samples'] > 0:
            print(f"\n🔵 Binary Questions (Yes/No):")
            print(f"  Accuracy:  {results['binary']['accuracy']:.4f} ({results['binary']['accuracy']*100:.2f}%)")
            if 'precision' in results['binary']:
                print(f"  Precision: {results['binary']['precision']:.4f}")
                print(f"  Recall:    {results['binary']['recall']:.4f}")
                print(f"  F1-Score:  {results['binary']['f1']:.4f}")
            print(f"  Samples:   {results['binary']['samples']}")
        
        # Open-ended
        if 'open_ended' in results and results['open_ended']['samples'] > 0:
            print(f"\n🟢 Open-Ended Questions:")
            print(f"  Accuracy:       {results['open_ended']['accuracy']:.4f} ({results['open_ended']['accuracy']*100:.2f}%)")
            if 'macro_precision' in results['open_ended']:
                print(f"  Macro Precision: {results['open_ended']['macro_precision']:.4f}")
                print(f"  Macro Recall:    {results['open_ended']['macro_recall']:.4f}")
                print(f"  Macro F1-Score:  {results['open_ended']['macro_f1']:.4f}")
            print(f"  Samples:        {results['open_ended']['samples']}")
            
            if 'by_type' in results['open_ended']:
                print(f"\n  By Question Type:")
                for qtype, metrics in results['open_ended']['by_type'].items():
                    print(f"    {qtype}:")
                    print(f"      Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
                    print(f"      Samples:  {metrics['samples']}")
        
        print("=" * 80)
    
    def full_evaluation(self, val_loader=None, test_loader=None):
        """Perform full evaluation on validation and test sets."""
        metrics_summary = {
            'checkpoint_path': str(self.checkpoint_path),
            'architecture': 'dual_head',
            'model_type': self.model_type,
            'device': str(self.device),
            'timestamp': self.timestamp,
            'config': self.config
        }
        
        # Evaluate validation set
        if val_loader is not None:
            print("\n" + "="*80)
            print("VALIDATION SET EVALUATION")
            print("="*80)
            val_results = self.evaluate(val_loader, 'validation')
            metrics_summary['validation'] = val_results
        
        # Evaluate test set
        if test_loader is not None:
            print("\n" + "="*80)
            print("TEST SET EVALUATION")
            print("="*80)
            test_results = self.evaluate(test_loader, 'test')
            metrics_summary['test'] = test_results
        
        # Save metrics summary
        metrics_path = self.output_dir / f'metrics_summary_{self.timestamp}.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"\n✓ Metrics summary saved to: {metrics_path}")
        
        # Generate report
        self._generate_report(metrics_summary)
        
        return metrics_summary
    
    def _generate_report(self, metrics_summary: dict):
        """Generate comprehensive evaluation report."""
        report_path = self.output_dir / f'evaluation_report_{self.timestamp}.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DUAL-HEAD VQA MODEL EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Checkpoint: {metrics_summary['checkpoint_path']}\n")
            f.write(f"Architecture: {metrics_summary['architecture']}\n")
            f.write(f"Model Type: {self.model_type.upper()}\n")
            f.write(f"Device: {metrics_summary['device']}\n\n")
            
            # Config
            f.write("=" * 80 + "\n")
            f.write("MODEL CONFIGURATION\n")
            f.write("=" * 80 + "\n")
            if self.config:
                for key, value in sorted(self.config.items()):
                    f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Results for each split
            for split in ['validation', 'test']:
                if split in metrics_summary:
                    f.write("=" * 80 + "\n")
                    f.write(f"{split.upper()} SET RESULTS\n")
                    f.write("=" * 80 + "\n\n")
                    
                    results = metrics_summary[split]
                    
                    # Overall
                    if 'overall' in results:
                        f.write(f"Overall Accuracy: {results['overall']['accuracy']:.4f} "
                               f"({results['overall']['accuracy']*100:.2f}%)\n\n")
                    
                    # Binary
                    if 'binary' in results and results['binary']['samples'] > 0:
                        f.write("Binary Questions (Yes/No):\n")
                        f.write(f"  Accuracy:  {results['binary']['accuracy']:.4f}\n")
                        if 'precision' in results['binary']:
                            f.write(f"  Precision: {results['binary']['precision']:.4f}\n")
                            f.write(f"  Recall:    {results['binary']['recall']:.4f}\n")
                            f.write(f"  F1-Score:  {results['binary']['f1']:.4f}\n")
                        f.write(f"  Samples:   {results['binary']['samples']}\n\n")
                    
                    # Open-ended
                    if 'open_ended' in results and results['open_ended']['samples'] > 0:
                        f.write("Open-Ended Questions:\n")
                        f.write(f"  Accuracy:        {results['open_ended']['accuracy']:.4f}\n")
                        if 'macro_precision' in results['open_ended']:
                            f.write(f"  Macro Precision: {results['open_ended']['macro_precision']:.4f}\n")
                            f.write(f"  Macro Recall:    {results['open_ended']['macro_recall']:.4f}\n")
                            f.write(f"  Macro F1-Score:  {results['open_ended']['macro_f1']:.4f}\n")
                        f.write(f"  Samples:         {results['open_ended']['samples']}\n\n")
            
            # Training history
            if self.history:
                f.write("=" * 80 + "\n")
                f.write("TRAINING HISTORY\n")
                f.write("=" * 80 + "\n\n")
                
                if self.history.get('val_overall_acc'):
                    f.write(f"Best Overall Val Acc: {max(self.history['val_overall_acc']):.2f}%\n")
                if self.history.get('val_binary_acc'):
                    f.write(f"Best Binary Val Acc:  {max(self.history['val_binary_acc']):.2f}%\n")
                if self.history.get('val_open_ended_acc'):
                    f.write(f"Best Open-Ended Val Acc: {max(self.history['val_open_ended_acc']):.2f}%\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"✓ Evaluation report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Dual-Head VQA Model (Lightweight or Baseline)')
    
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/dual_head_fixed/best_model.pt',
                       help='Path to checkpoint')
    parser.add_argument('--model_type', type=str, default='auto',
                       choices=['auto', 'lightweight', 'baseline'],
                       help='Model type: auto (detect from path), lightweight, or baseline')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--max_answer_vocab_size', type=int, default=120)
    parser.add_argument('--output_dir', type=str, default='evaluation_results/dual_head')
    parser.add_argument('--device', type=str, default=None)
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    
    print("=" * 80)
    print("DUAL-HEAD VQA MODEL EVALUATION")
    print("Supports: Lightweight (MobileNetV3 + DistilBERT) and Baseline (ResNet-34 + BERT)")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, answer_vocab, _ = create_combined_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        max_answer_vocab_size=args.max_answer_vocab_size,
        encode_answers=False
    )
    
    # Check checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return
    
    # Create evaluator
    evaluator = DualHeadEvaluator(
        checkpoint_path,
        test_loader,
        answer_vocab,
        args.device,
        args.output_dir,
        model_type=args.model_type
    )
    
    # Run evaluation
    metrics_summary = evaluator.full_evaluation(
        val_loader=val_loader,
        test_loader=test_loader
    )
    
    print("\n" + "=" * 80)
    print("🎉 EVALUATION COMPLETED!")
    print("=" * 80)
    print(f"\nGenerated files in {evaluator.output_dir}/:")
    print("  📊 Detailed results (CSV)")
    print("  📈 Confusion matrices (PNG)")
    print("  📉 Comparison charts (PNG)")
    print("  📄 Evaluation report (TXT)")
    print("  📋 Metrics summary (JSON)")
    print("=" * 80)


if __name__ == "__main__":
    main()

