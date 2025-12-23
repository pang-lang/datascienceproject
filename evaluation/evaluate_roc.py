#!/usr/bin/env python3
"""
Enhanced evaluation script with AUC-ROC metrics and ROC curves.
Can be used for any VQA model checkpoint.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.combined_preprocessing import create_combined_data_loaders
from models.lightweight_model import DualHeadVQAModel
from models.baseline_model import DualHeadBaselineVQAModel


class EnhancedModelEvaluator:
    """Evaluator with AUC-ROC metrics and ROC curves."""
    
    def __init__(self,
                 checkpoint_path: str,
                 data_loader,
                 answer_vocab: dict,
                 device='cuda',
                 output_dir=None):
        self.checkpoint_path = Path(checkpoint_path)
        self.data_loader = data_loader
        self.answer_vocab = answer_vocab
        self.idx_to_answer = {idx: ans for ans, idx in answer_vocab.items()}
        self.num_classes = len(answer_vocab)
        self.device = device
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if output_dir is None:
            output_dir = f'evaluation_results_roc/{self.timestamp}'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = {}
        self.history = {}
        
        self.model = self._load_model()
        
        print(f"✓ Loaded model from: {self.checkpoint_path}")
        print(f"✓ Output directory: {self.output_dir}")
    
    def _load_model(self):
        """Load model from checkpoint."""
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        # Detect model type based on checkpoint path or config
        checkpoint_str = str(self.checkpoint_path).lower()
        
        if 'lightweight' in checkpoint_str or 'mobile' in checkpoint_str:
            # Lightweight dual-head model
            num_open_ended_classes = self.config.get('num_open_ended_classes', self.num_classes)
            model = DualHeadVQAModel(
                num_open_ended_classes=num_open_ended_classes,
                fusion_hidden_dim=self.config.get('fusion_hidden_dim', 256),
                num_attention_heads=self.config.get('num_attention_heads', 4),
                dropout=self.config.get('dropout', 0.3),
                freeze_vision_encoder=False,
                freeze_text_encoder=False
            )
        else:
            # Baseline dual-head model
            num_open_ended_classes = self.config.get('num_open_ended_classes', self.num_classes)
            model = DualHeadBaselineVQAModel(
                num_open_ended_classes=num_open_ended_classes,
                fusion_hidden_dim=self.config.get('fusion_hidden_dim', 512),
                num_attention_heads=self.config.get('num_attention_heads', 8),
                dropout=self.config.get('dropout', 0.35),
                freeze_vision_encoder=False,
                freeze_text_encoder=False
            )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    @torch.no_grad()
    def evaluate(self):
        """Run evaluation with AUC-ROC metrics (with proper dual-head routing)."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []  # For AUC-ROC
        all_logits = []
        all_question_types = []
        all_question_texts = []
        all_answer_texts = []
        all_head_used = []  # Track which head was used
        
        # Binary answer indices
        binary_answers = {'yes', 'no'}
        yes_idx = self.answer_vocab.get('yes', -1)
        no_idx = self.answer_vocab.get('no', -1)
        
        print("\n" + "="*80)
        print("RUNNING EVALUATION WITH AUC-ROC METRICS (DUAL-HEAD ROUTING)")
        print("="*80)
        print(f"Binary answer indices: yes={yes_idx}, no={no_idx}")
        
        for batch in tqdm(self.data_loader, desc='Evaluating'):
            images = batch['image'].to(self.device)
            input_ids = batch['question']['input_ids'].to(self.device)
            attention_mask = batch['question']['attention_mask'].to(self.device)
            answers = batch['answer_idx'].to(self.device)
            answer_texts_batch = batch['answer']['text']
            
            # Dual-head models return a dictionary with 'binary' and 'open_ended' logits
            outputs = self.model(images, input_ids, attention_mask)
            
            # Route each sample to appropriate head based on ground truth answer
            batch_size = answers.size(0)
            batch_predictions = []
            batch_probabilities = []
            
            for i in range(batch_size):
                answer_text = answer_texts_batch[i]
                is_binary = answer_text in binary_answers
                
                if is_binary and yes_idx != -1 and no_idx != -1:
                    # Use binary head
                    binary_logits = outputs['binary'][i:i+1]  # (1, 2)
                    probs_binary = F.softmax(binary_logits, dim=1)[0]  # (2,)
                    pred_binary = torch.argmax(binary_logits, dim=1).item()
                    
                    # Map binary head output (0=no, 1=yes) to full vocabulary indices
                    if pred_binary == 0:
                        final_pred = no_idx
                    else:
                        final_pred = yes_idx
                    
                    # Create full probability distribution
                    probs_full = torch.zeros(self.num_classes, device=self.device)
                    probs_full[no_idx] = probs_binary[0]
                    probs_full[yes_idx] = probs_binary[1]
                    
                    batch_predictions.append(final_pred)
                    batch_probabilities.append(probs_full.cpu().numpy())
                    all_head_used.append('binary')
                else:
                    # Use open-ended head
                    open_logits = outputs['open_ended'][i:i+1]  # (1, num_classes)
                    probs = F.softmax(open_logits, dim=1)[0]  # (num_classes,)
                    pred = torch.argmax(open_logits, dim=1).item()
                    
                    batch_predictions.append(pred)
                    batch_probabilities.append(probs.cpu().numpy())
                    all_head_used.append('open_ended')
            
            predictions = torch.tensor(batch_predictions, device=self.device)
            probabilities_batch = np.array(batch_probabilities)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(answers.cpu().numpy())
            all_probabilities.extend(probabilities_batch)
            all_question_types.extend(batch.get('question_type', ['unknown'] * len(answers)))
            all_question_texts.extend(batch['question'].get('text', [''] * len(answers)))
            all_answer_texts.extend(answer_texts_batch)
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        all_head_used = np.array(all_head_used)
        
        pred_answers = [self.idx_to_answer.get(p, 'unknown') for p in all_predictions]
        
        # Print head usage statistics
        binary_count = (all_head_used == 'binary').sum()
        open_ended_count = (all_head_used == 'open_ended').sum()
        print(f"\n📊 Head Usage Statistics:")
        print(f"   Binary head: {binary_count} samples ({binary_count/len(all_head_used)*100:.1f}%)")
        print(f"   Open-ended head: {open_ended_count} samples ({open_ended_count/len(all_head_used)*100:.1f}%)")
        
        # Calculate metrics
        results = self._calculate_metrics_with_roc(
            all_predictions, all_targets, all_probabilities,
            all_question_types, pred_answers, all_answer_texts
        )
        
        # Add head usage stats to results
        results['head_usage'] = {
            'binary': int(binary_count),
            'open_ended': int(open_ended_count),
            'binary_pct': float(binary_count/len(all_head_used)*100),
            'open_ended_pct': float(open_ended_count/len(all_head_used)*100)
        }
        
        # Save detailed results
        self._save_detailed_results(
            all_predictions, all_targets, all_probabilities,
            all_question_types, all_question_texts, 
            all_answer_texts, pred_answers, all_head_used
        )
        
        # Generate visualizations
        self._plot_confusion_matrix(all_predictions, all_targets)
        self._plot_roc_curves(all_targets, all_probabilities)
        self._plot_per_class_auc(all_targets, all_probabilities)
        
        # Print and save summary
        self._print_summary(results)
        self._save_summary(results)
        
        return results
    
    def _calculate_metrics_with_roc(self, predictions, targets, probabilities, 
                                     question_types, pred_answers, true_answers):
        """Calculate comprehensive metrics including AUC-ROC."""
        results = {}
        
        # Overall metrics
        results['accuracy'] = accuracy_score(targets, predictions)
        results['total_samples'] = len(targets)
        
        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average='macro', zero_division=0
        )
        results['macro_precision'] = precision
        results['macro_recall'] = recall
        results['macro_f1'] = f1
        
        # AUC-ROC Metrics
        try:
            # Multi-class AUC (One-vs-Rest)
            # Only calculate for classes present in test set
            unique_classes = np.unique(targets)
            
            if len(unique_classes) > 1:
                # Select only probabilities for classes present in test set
                present_probs = probabilities[:, unique_classes]
                # Renormalize so probabilities sum to 1.0
                present_probs = present_probs / present_probs.sum(axis=1, keepdims=True)
                
                results['auc_ovr_macro'] = roc_auc_score(
                    targets, present_probs, 
                    multi_class='ovr', 
                    average='macro',
                    labels=unique_classes
                )
                results['auc_ovr_weighted'] = roc_auc_score(
                    targets, present_probs,
                    multi_class='ovr',
                    average='weighted',
                    labels=unique_classes
                )
            else:
                print(f"⚠️  Only one class in test set, cannot calculate multi-class AUC")
                results['auc_ovr_macro'] = 0.0
                results['auc_ovr_weighted'] = 0.0
        except Exception as e:
            print(f"⚠️  Could not calculate multi-class AUC: {e}")
            results['auc_ovr_macro'] = 0.0
            results['auc_ovr_weighted'] = 0.0
        
        # Binary metrics (yes/no questions)
        binary_answers = {'yes', 'no'}
        binary_mask = [ans in binary_answers for ans in pred_answers]
        
        if sum(binary_mask) > 0:
            binary_preds = predictions[binary_mask]
            binary_targets = targets[binary_mask]
            binary_probs = probabilities[binary_mask]
            
            results['binary_accuracy'] = accuracy_score(binary_targets, binary_preds)
            results['binary_samples'] = sum(binary_mask)
            
            if len(set(binary_targets)) > 1:
                b_precision, b_recall, b_f1, _ = precision_recall_fscore_support(
                    binary_targets, binary_preds, average='macro', zero_division=0
                )
                results['binary_precision'] = b_precision
                results['binary_recall'] = b_recall
                results['binary_f1'] = b_f1
                
                # Binary AUC-ROC
                try:
                    # Get yes/no class indices
                    yes_idx = self.answer_vocab.get('yes', -1)
                    no_idx = self.answer_vocab.get('no', -1)
                    
                    if yes_idx != -1 and no_idx != -1:
                        # Binary classification: use probability of positive class
                        binary_labels = (binary_targets == yes_idx).astype(int)
                        binary_prob_positive = binary_probs[:, yes_idx]
                        
                        results['binary_auc'] = roc_auc_score(
                            binary_labels, binary_prob_positive
                        )
                except Exception as e:
                    print(f"⚠️  Could not calculate binary AUC: {e}")
                    results['binary_auc'] = 0.0
        else:
            results['binary_accuracy'] = 0
            results['binary_samples'] = 0
            results['binary_precision'] = 0
            results['binary_recall'] = 0
            results['binary_f1'] = 0
            results['binary_auc'] = 0
        
        # Per-class AUC
        results['per_class_auc'] = {}
        target_binarized = label_binarize(targets, classes=np.arange(self.num_classes))
        
        for class_idx in range(self.num_classes):
            class_name = self.idx_to_answer.get(class_idx, f'class_{class_idx}')
            
            # Only calculate if this class appears in targets
            if target_binarized[:, class_idx].sum() > 0:
                try:
                    class_auc = roc_auc_score(
                        target_binarized[:, class_idx],
                        probabilities[:, class_idx]
                    )
                    results['per_class_auc'][class_name] = class_auc
                except:
                    results['per_class_auc'][class_name] = 0.0
        
        # Per question type
        results['by_question_type'] = {}
        for qtype in set(question_types):
            mask = np.array([qt == qtype for qt in question_types])
            if sum(mask) > 0:
                type_preds = predictions[mask]
                type_targets = targets[mask]
                
                results['by_question_type'][qtype] = {
                    'accuracy': accuracy_score(type_targets, type_preds),
                    'samples': sum(mask)
                }
        
        # Top-K accuracy
        for k in [3, 5, 10]:
            top_k_preds = np.argsort(probabilities, axis=1)[:, -k:]
            top_k_correct = [t in top_k_preds[i] for i, t in enumerate(targets)]
            results[f'top_{k}_accuracy'] = np.mean(top_k_correct)
        
        return results
    
    def _plot_roc_curves(self, targets, probabilities):
        """Plot ROC curves for top classes (excluding yes/no) and macro-average."""
        from collections import Counter
        
        # Get all class counts
        target_counts = Counter(targets)
        
        # Exclude yes/no from open-ended class analysis (they use binary head)
        binary_answers = {'yes', 'no'}
        yes_idx = self.answer_vocab.get('yes', -1)
        no_idx = self.answer_vocab.get('no', -1)
        binary_indices = {yes_idx, no_idx} if yes_idx != -1 and no_idx != -1 else set()
        
        # Get top 10 open-ended classes (excluding yes/no)
        open_ended_classes = [
            (cls, count) for cls, count in target_counts.most_common()
            if cls not in binary_indices
        ][:10]
        top_classes = [cls for cls, _ in open_ended_classes]
        
        print(f"\n📊 Top 10 Open-Ended Classes (excluding yes/no):")
        for i, (cls, count) in enumerate(open_ended_classes, 1):
            class_name = self.idx_to_answer.get(cls, f'class_{cls}')
            print(f"  {i}. {class_name}: {count} samples")
        
        # Binarize targets
        target_binarized = label_binarize(targets, classes=np.arange(self.num_classes))
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Plot 1: Top 10 Open-Ended Classes (combined, excluding yes/no)
        ax1 = axes[0]
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for i, class_idx in enumerate(top_classes):
            if target_binarized[:, class_idx].sum() > 0:
                fpr, tpr, _ = roc_curve(
                    target_binarized[:, class_idx],
                    probabilities[:, class_idx]
                )
                roc_auc = auc(fpr, tpr)
                class_name = self.idx_to_answer.get(class_idx, f'Class {class_idx}')
                ax1.plot(fpr, tpr, lw=2.5, color=colors[i],
                        label=f'{class_name} (AUC={roc_auc:.3f})')
        
        ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Random', alpha=0.5)
        ax1.set_xlabel('False Positive Rate', fontsize=12)
        ax1.set_ylabel('True Positive Rate', fontsize=12)
        ax1.set_title('ROC Curves - Top 10 Open-Ended Classes\n(Binary questions excluded)', 
                     fontsize=13, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Binary (Yes/No) - Dedicated to binary head
        ax2 = axes[1]
        
        if yes_idx != -1 and target_binarized[:, yes_idx].sum() > 0:
            fpr, tpr, _ = roc_curve(
                target_binarized[:, yes_idx],
                probabilities[:, yes_idx]
            )
            roc_auc = auc(fpr, tpr)
            ax2.plot(fpr, tpr, lw=3, label=f'Yes (AUC={roc_auc:.3f})', color='green')
        
        if no_idx != -1 and target_binarized[:, no_idx].sum() > 0:
            fpr, tpr, _ = roc_curve(
                target_binarized[:, no_idx],
                probabilities[:, no_idx]
            )
            roc_auc = auc(fpr, tpr)
            ax2.plot(fpr, tpr, lw=3, label=f'No (AUC={roc_auc:.3f})', color='red')
        
        ax2.plot([0, 1], [0, 1], 'k--', lw=2, label='Random', alpha=0.5)
        ax2.set_xlabel('False Positive Rate', fontsize=12)
        ax2.set_ylabel('True Positive Rate', fontsize=12)
        ax2.set_title('ROC Curves - Binary Head (Yes/No)\nDual-head routing', 
                     fontsize=13, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Macro-average ROC (all classes)
        ax3 = axes[2]
        
        # Compute macro-average ROC across all classes
        all_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(all_fpr)
        n_classes_with_samples = 0
        
        for class_idx in range(self.num_classes):
            if target_binarized[:, class_idx].sum() > 0:
                fpr, tpr, _ = roc_curve(
                    target_binarized[:, class_idx],
                    probabilities[:, class_idx]
                )
                mean_tpr += np.interp(all_fpr, fpr, tpr)
                n_classes_with_samples += 1
        
        if n_classes_with_samples > 0:
            mean_tpr /= n_classes_with_samples
        macro_auc = auc(all_fpr, mean_tpr)
        
        ax3.plot(all_fpr, mean_tpr, lw=3, 
                label=f'Macro-average (AUC={macro_auc:.3f})',
                color='navy')
        ax3.plot([0, 1], [0, 1], 'k--', lw=2, label='Random', alpha=0.5)
        ax3.set_xlabel('False Positive Rate', fontsize=12)
        ax3.set_ylabel('True Positive Rate', fontsize=12)
        ax3.set_title('Macro-Average ROC Curve\n(All classes)', 
                     fontsize=13, fontweight='bold')
        ax3.legend(loc='lower right', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / f'roc_curves_{self.timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ ROC curves saved to: {plot_path}")
    
    def _plot_per_class_auc(self, targets, probabilities):
        """Plot per-class AUC scores."""
        from collections import Counter
        
        target_counts = Counter(targets)
        top_classes = [cls for cls, _ in target_counts.most_common(20)]
        
        target_binarized = label_binarize(targets, classes=np.arange(self.num_classes))
        
        class_aucs = []
        class_names = []
        
        for class_idx in top_classes:
            if target_binarized[:, class_idx].sum() > 0:
                try:
                    class_auc = roc_auc_score(
                        target_binarized[:, class_idx],
                        probabilities[:, class_idx]
                    )
                    class_aucs.append(class_auc)
                    class_names.append(self.idx_to_answer.get(class_idx, f'Class {class_idx}'))
                except:
                    pass
        
        # Sort by AUC
        sorted_pairs = sorted(zip(class_names, class_aucs), key=lambda x: x[1], reverse=True)
        class_names, class_aucs = zip(*sorted_pairs) if sorted_pairs else ([], [])
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = plt.cm.RdYlGn(np.array(class_aucs))
        bars = ax.barh(range(len(class_names)), class_aucs, color=colors)
        
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names)
        ax.set_xlabel('AUC-ROC Score', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class AUC-ROC Scores (Top 20)', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random (0.5)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, class_aucs)):
            ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plot_path = self.output_dir / f'per_class_auc_{self.timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Per-class AUC plot saved to: {plot_path}")
    
    def _plot_confusion_matrix(self, predictions, targets):
        """Plot confusion matrix for top classes."""
        from collections import Counter
        
        target_counts = Counter(targets)
        top_classes = [cls for cls, _ in target_counts.most_common(15)]
        
        mask = np.isin(targets, top_classes)
        if sum(mask) > 10:
            filtered_preds = predictions[mask]
            filtered_targets = targets[mask]
            
            cm = confusion_matrix(filtered_targets, filtered_preds, labels=top_classes)
            
            fig, ax = plt.subplots(figsize=(14, 12))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=[self.idx_to_answer.get(c, f'{c}') for c in top_classes],
                       yticklabels=[self.idx_to_answer.get(c, f'{c}') for c in top_classes])
            
            ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
            ax.set_ylabel('True', fontsize=12, fontweight='bold')
            ax.set_title('Confusion Matrix (Top 15 Classes)', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plot_path = self.output_dir / f'confusion_matrix_{self.timestamp}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Confusion matrix saved to: {plot_path}")
    
    def _save_detailed_results(self, predictions, targets, probabilities,
                               question_types, question_texts, 
                               true_answers, pred_answers, head_used):
        """Save detailed results to CSV."""
        import pandas as pd
        
        # Get confidence scores (max probability)
        confidences = np.max(probabilities, axis=1)
        
        df = pd.DataFrame({
            'question': question_texts,
            'true_answer': true_answers,
            'predicted_answer': pred_answers,
            'confidence': confidences,
            'correct': predictions == targets,
            'question_type': question_types,
            'head_used': head_used,
            'true_label': targets,
            'pred_label': predictions
        })
        
        csv_path = self.output_dir / f'detailed_results_{self.timestamp}.csv'
        df.to_csv(csv_path, index=False)
        print(f"✓ Detailed results saved to: {csv_path}")
    
    def _print_summary(self, results):
        """Print evaluation summary."""
        print("\n" + "="*80)
        print("EVALUATION RESULTS WITH AUC-ROC (DUAL-HEAD ROUTING)")
        print("="*80)
        
        print(f"\nOverall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"Total Samples: {results['total_samples']}")
        
        if 'head_usage' in results:
            print(f"\nDual-Head Usage:")
            print(f"  Binary head:      {results['head_usage']['binary']:>4} samples ({results['head_usage']['binary_pct']:.1f}%)")
            print(f"  Open-ended head:  {results['head_usage']['open_ended']:>4} samples ({results['head_usage']['open_ended_pct']:.1f}%)")
        
        print(f"\nMulti-class Metrics:")
        print(f"  Macro Precision: {results['macro_precision']:.4f}")
        print(f"  Macro Recall:    {results['macro_recall']:.4f}")
        print(f"  Macro F1-Score:  {results['macro_f1']:.4f}")
        
        print(f"\nAUC-ROC Scores:")
        print(f"  Macro-average:    {results['auc_ovr_macro']:.4f}")
        print(f"  Weighted-average: {results['auc_ovr_weighted']:.4f}")
        
        if results['binary_samples'] > 0:
            print(f"\nBinary Questions (Yes/No):")
            print(f"  Accuracy:  {results['binary_accuracy']:.4f}")
            print(f"  Precision: {results['binary_precision']:.4f}")
            print(f"  Recall:    {results['binary_recall']:.4f}")
            print(f"  F1-Score:  {results['binary_f1']:.4f}")
            print(f"  AUC-ROC:   {results.get('binary_auc', 0):.4f}")
        
        print(f"\nTop-K Accuracy:")
        print(f"  Top-3:  {results['top_3_accuracy']:.4f}")
        print(f"  Top-5:  {results['top_5_accuracy']:.4f}")
        print(f"  Top-10: {results['top_10_accuracy']:.4f}")
        
        print("\n" + "="*80)
    
    def _save_summary(self, results):
        """Save evaluation summary to file."""
        summary_path = self.output_dir / f'evaluation_summary_{self.timestamp}.json'
        
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✓ Summary saved to: {summary_path}")
        
        # Also save as text
        text_path = self.output_dir / f'evaluation_report_{self.timestamp}.txt'
        with open(text_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("VQA MODEL EVALUATION REPORT (WITH AUC-ROC & DUAL-HEAD ROUTING)\n")
            f.write("="*80 + "\n\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Checkpoint: {self.checkpoint_path}\n")
            f.write(f"Device: {self.device}\n\n")
            
            f.write("="*80 + "\n")
            f.write("RESULTS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Overall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n\n")
            
            if 'head_usage' in results:
                f.write("Dual-Head Usage:\n")
                f.write(f"  Binary head:      {results['head_usage']['binary']:>4} samples ({results['head_usage']['binary_pct']:.1f}%)\n")
                f.write(f"  Open-ended head:  {results['head_usage']['open_ended']:>4} samples ({results['head_usage']['open_ended_pct']:.1f}%)\n\n")
            
            f.write("Multi-class Metrics:\n")
            f.write(f"  Macro Precision: {results['macro_precision']:.4f}\n")
            f.write(f"  Macro Recall:    {results['macro_recall']:.4f}\n")
            f.write(f"  Macro F1-Score:  {results['macro_f1']:.4f}\n\n")
            
            f.write("AUC-ROC Scores:\n")
            f.write(f"  Macro-average:    {results['auc_ovr_macro']:.4f}\n")
            f.write(f"  Weighted-average: {results['auc_ovr_weighted']:.4f}\n\n")
            
            if results['binary_samples'] > 0:
                f.write("Binary Questions (Yes/No):\n")
                f.write(f"  Accuracy:  {results['binary_accuracy']:.4f}\n")
                f.write(f"  Precision: {results['binary_precision']:.4f}\n")
                f.write(f"  Recall:    {results['binary_recall']:.4f}\n")
                f.write(f"  F1-Score:  {results['binary_f1']:.4f}\n")
                f.write(f"  AUC-ROC:   {results.get('binary_auc', 0):.4f}\n\n")
            
            f.write("="*80 + "\n")
        
        print(f"✓ Text report saved to: {text_path}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced VQA model evaluation with AUC-ROC')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--batch_size', type=int, default=24,
                       help='Batch size')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/mps/cpu)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, answer_vocab, _ = create_combined_data_loaders(
        batch_size=args.batch_size,
        num_workers=0,
        max_answer_vocab_size=120
    )
    
    # Select split
    if args.split == 'train':
        data_loader = train_loader
    elif args.split == 'val':
        data_loader = val_loader
    else:
        data_loader = test_loader
    
    print(f"✓ Loaded {args.split} split with {len(data_loader)} batches")
    print(f"✓ Answer vocabulary size: {len(answer_vocab)}")
    
    # Run evaluation
    evaluator = EnhancedModelEvaluator(
        checkpoint_path=args.checkpoint,
        data_loader=data_loader,
        answer_vocab=answer_vocab,
        device=device,
        output_dir=args.output_dir
    )
    
    results = evaluator.evaluate()
    
    print("\n✅ Evaluation complete!")
    print(f"📁 Results saved to: {evaluator.output_dir}")


if __name__ == '__main__':
    main()

