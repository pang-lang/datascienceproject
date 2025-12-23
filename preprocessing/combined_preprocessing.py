#!/usr/bin/env python3
"""
Complete Fixed Combined Preprocessing with Answer Normalization Validation
"""

import json
import os
import string
import re
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Tuple, Optional
from collections import Counter
import matplotlib.pyplot as plt

from .image_preprocessing import (
    MedicalImagePreprocessor,
    get_or_compute_dataset_stats
)
from .text_preprocessing import MedicalTextPreprocessor

# Constants
UNK_ANSWER_TOKEN = "<unk>"
DEFAULT_SPLIT_SEED = 42
DEFAULT_SPLIT_PATH = os.path.join('data_splits', f'vqa_rad_seed{DEFAULT_SPLIT_SEED}.json')

# Binary answers
BINARY_ANSWERS = {"yes", "no"}

# Enhanced answer synonyms
ANSWER_SYNONYMS = {
    # ============================================================================
    # BINARY ANSWERS - Canonical: "no"
    # ============================================================================
    "no abnormality": "no",
    "no abnormalities": "no",
    "no abnormality detected": "no",
    "no evidence of disease": "no",
    "no evidence of acute disease": "no",
    "no acute abnormality": "no",
    "no acute abnormalities": "no",
    "no acute findings": "no",
    "no acute intracranial abnormality": "no", 
    "no acute disease": "no",
    "negative": "no",
    "unremarkable": "no",  
    "normal": "no",       
    "normal study": "no",
    "within normal limits": "no",
    "none": "no",
    "not seen": "no",
    "absent": "no",
    "without": "no",
    "not visible": "no",
    "not present": "no",
    
    # BINARY ANSWERS - Canonical: "yes"
    "positive": "yes",
    "present": "yes",
    "abnormal": "yes",
    "detected": "yes",
    "visible": "yes",
    "seen": "yes",

    # ============================================================================
    # DEMOGRAPHIC NORMALIZATION (Open-ended)
    # ============================================================================
    "female patient": "female",
    "woman": "female",
    "male patient": "male",
    "man": "male",
    
    # ============================================================================
    # MEDICAL CONDITION NORMALIZATION
    # ============================================================================
    "cardiac enlargement": "cardiomegaly",
    "enlarged cardiac silhouette": "cardiomegaly",
    "enlarged heart": "cardiomegaly",
    "cardiac silhouette enlargement": "cardiomegaly",
    "heart enlargement": "cardiomegaly",
    
    "pulmonary edema": "edema",
    "pulmonary congestion": "edema",
    "fluid in the lungs": "edema",
    "lung fluid": "edema",
    
    "pleural effusion": "effusion",
    "pericardial effusion": "effusion",
    "fluid in pleural space": "effusion",
    
    "brain bleed": "hemorrhage",
    "intracranial hemorrhage": "hemorrhage",
    "ich": "hemorrhage",
    
    "fractured": "fracture",
    "broken bone": "fracture",
    
    "collapsed lung": "pneumothorax",
    "lung collapse": "pneumothorax",
    "air in pleural space": "pneumothorax",
    
    "lung infection": "pneumonia",
    "pulmonary infection": "pneumonia",
}


def normalize_answer_text(answer: str) -> str:
    """
    Normalize answer text with improved handling.
    
    Args:
        answer: Raw answer string
        
    Returns:
        Normalized answer string or UNK_ANSWER_TOKEN
    """
    if not isinstance(answer, str):
        return UNK_ANSWER_TOKEN
    
    text = answer.lower().strip()
    if not text:
        return UNK_ANSWER_TOKEN
    
    # Normalize whitespace BEFORE removing punctuation
    text = " ".join(text.split())
    
    # Remove punctuation (keep hyphens for medical terms)
    punctuation_to_remove = string.punctuation.replace("-", "")
    text = text.translate(str.maketrans("", "", punctuation_to_remove))
    
    # Clean up hyphens
    text = re.sub(r"-+", "-", text)
    text = text.strip("-")
    
    # Final whitespace cleanup
    text = " ".join(text.split())
    
    if not text:
        return UNK_ANSWER_TOKEN
    
    # Apply synonym mapping
    if text in ANSWER_SYNONYMS:
        text = ANSWER_SYNONYMS[text]
    
    return text or UNK_ANSWER_TOKEN


def derive_question_type(answer: str) -> str:
    """Determine if question is binary or open-ended based on answer."""
    normalized = normalize_answer_text(answer)
    return 'binary' if normalized in BINARY_ANSWERS else 'open-ended'


def check_unk_rate(dataset_name: str = "flaviagiammarino/vqa-rad", split: str = "train"):
    """
    Check the <unk> rate and analyze answer distribution.
    
    This function helps validate your normalization is working correctly.
    """
    print("\n" + "="*80)
    print(f"CHECKING <UNK> RATE FOR {split.upper()} SPLIT")
    print("="*80)
    
    # Load dataset
    ds = load_dataset(dataset_name)
    data = ds[split]
    
    # Collect normalized answers
    normalized_answers = []
    raw_answers = []
    unk_samples = []
    
    for i, sample in enumerate(data):
        raw = sample['answer']
        normalized = normalize_answer_text(raw)
        
        raw_answers.append(raw)
        normalized_answers.append(normalized)
        
        if normalized == UNK_ANSWER_TOKEN:
            unk_samples.append({
                'idx': i,
                'raw': raw,
                'question': sample['question']
            })
    
    # Calculate statistics
    total = len(normalized_answers)
    unk_count = normalized_answers.count(UNK_ANSWER_TOKEN)
    unk_rate = 100 * unk_count / total
    
    # Count by type
    counts = Counter(normalized_answers)
    binary_count = sum(counts[ans] for ans in BINARY_ANSWERS if ans in counts)
    open_ended_count = total - binary_count - unk_count
    
    print(f"\nTotal samples: {total}")
    print(f"<unk> count: {unk_count} ({unk_rate:.2f}%)")
    print(f"Binary answers: {binary_count} ({100*binary_count/total:.1f}%)")
    print(f"Open-ended answers: {open_ended_count} ({100*open_ended_count/total:.1f}%)")
    
    # Show binary distribution
    print("\nBinary distribution:")
    for ans in BINARY_ANSWERS:
        count = counts.get(ans, 0)
        pct = 100 * count / binary_count if binary_count > 0 else 0
        print(f"  {ans}: {count} ({pct:.1f}%)")
    
    # Show top 10 answers
    print("\nTop 10 most common answers:")
    for i, (ans, count) in enumerate(counts.most_common(10), 1):
        pct = 100 * count / total
        ans_type = "binary" if ans in BINARY_ANSWERS else "open-ended" if ans != UNK_ANSWER_TOKEN else "unknown"
        print(f"  {i}. '{ans}' ({ans_type}): {count} ({pct:.1f}%)")
    
    # Show UNK samples (if any)
    if unk_samples:
        print(f"\n⚠️ WARNING: {len(unk_samples)} samples mapped to <unk>")
        print("\nFirst 5 <unk> samples:")
        for sample in unk_samples[:5]:
            print(f"  [{sample['idx']}] Q: {sample['question'][:60]}...")
            print(f"       A: '{sample['raw']}'")
    else:
        print("\n✅ No <unk> samples found!")
    
    # Show rare answers (count = 1)
    rare_answers = [ans for ans, count in counts.items() 
                    if count == 1 and ans not in BINARY_ANSWERS and ans != UNK_ANSWER_TOKEN]
    if rare_answers:
        print(f"\n⚠️ {len(rare_answers)} answers appear only once:")
        print(f"  {rare_answers[:10]}..." if len(rare_answers) > 10 else f"  {rare_answers}")
    
    print("="*80)
    
    return {
        'total': total,
        'unk_count': unk_count,
        'unk_rate': unk_rate,
        'binary_count': binary_count,
        'open_ended_count': open_ended_count,
        'unk_samples': unk_samples,
        'rare_answers': rare_answers,
        'distribution': counts
    }


def get_or_create_split_indices(dataset_name: str,
                                base_dataset=None,
                                split_path: str = DEFAULT_SPLIT_PATH,
                                seed: int = DEFAULT_SPLIT_SEED,
                                use_official_test_split: bool = True) -> Dict[str, List[int]]:
    """Create or load dataset split indices."""
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    
    if os.path.exists(split_path):
        with open(split_path, 'r') as fp:
            stored = json.load(fp)
        result = {}
        for k, v in stored.items():
            if isinstance(v, list):
                result[k] = list(v)
            else:
                result[k] = v
        return result
    
    if use_official_test_split:
        full_dataset = load_dataset(dataset_name)
        train_dataset = full_dataset['train']
        test_dataset = full_dataset['test']
        
        labels = [derive_question_type(train_dataset[i]['answer']) for i in range(len(train_dataset))]
        indices = np.arange(len(train_dataset))
        
        train_idx, val_idx = train_test_split(
            indices,
            test_size=0.2,
            random_state=seed,
            stratify=labels
        )
        
        test_idx = np.arange(len(train_dataset), len(train_dataset) + len(test_dataset))
        
        splits = {
            'train': train_idx.tolist(),
            'val': val_idx.tolist(),
            'test': test_idx.tolist(),
            'use_official_test': True,
            'train_size': len(train_dataset),
            'test_size': len(test_dataset)
        }
    else:
        if base_dataset is None:
            base_dataset = load_dataset(dataset_name)['train']
        
        labels = [derive_question_type(base_dataset[i]['answer']) for i in range(len(base_dataset))]
        indices = np.arange(len(base_dataset))
        
        train_idx, temp_idx, train_labels, temp_labels = train_test_split(
            indices,
            labels,
            train_size=0.7,
            random_state=seed,
            stratify=labels
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=0.5,
            random_state=seed,
            stratify=[labels[i] for i in temp_idx]
        )
        
        splits = {
            'train': train_idx.tolist(),
            'val': val_idx.tolist(),
            'test': test_idx.tolist(),
            'use_official_test': False
        }
    
    with open(split_path, 'w') as fp:
        json.dump(splits, fp)
    print(f"✓ Created new splits: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")
    return splits


class CombinedVQADataset(Dataset):
    """Combined dataset for image and text preprocessing."""
    
    def __init__(self, 
                 dataset_name: str = "flaviagiammarino/vqa-rad",
                 split: str = "train",
                 image_preprocessor: Optional[MedicalImagePreprocessor] = None,
                 text_preprocessor: Optional[MedicalTextPreprocessor] = None,
                 max_samples: Optional[int] = None,
                 question_types: Optional[List[str]] = None,
                 answer_vocab: Optional[Dict[str, int]] = None,
                 allow_vocab_updates: bool = False,
                 max_answer_vocab_size: Optional[int] = None,
                 preprocess_steps: Optional[List[str]] = None,
                 encode_answers: bool = False,
                 coverage_threshold: float = 0.95,
                 base_dataset=None,
                 indices: Optional[List[int]] = None):
        
        self.dataset_name = dataset_name
        self.split = split
        self.image_preprocessor = image_preprocessor or MedicalImagePreprocessor()
        self.text_preprocessor = text_preprocessor or MedicalTextPreprocessor()
        self.max_samples = max_samples
        self.question_types = question_types
        self.max_answer_vocab_size = max_answer_vocab_size
        self.preprocess_steps = preprocess_steps or []
        self.encode_answers = encode_answers
        self.coverage_threshold = coverage_threshold
        self.coverage_curve: List[Dict[str, float]] = []
        self.allow_vocab_updates = allow_vocab_updates
        
        # Load dataset
        if base_dataset is not None:
            self.dataset = base_dataset
        else:
            self.dataset = load_dataset(dataset_name)['train']
        
        if indices is not None:
            self.indices = list(indices)
        else:
            self.indices = list(range(len(self.dataset)))
        
        if question_types:
            target_types = set(question_types)
            self.indices = [
                idx for idx in self.indices
                if self._derive_question_type(self.dataset[idx].get('answer', '')) in target_types
            ]
        
        if max_samples:
            self.indices = self.indices[:min(max_samples, len(self.indices))]
        
        print(f"Loaded {len(self.indices)} indexed samples for {split} split")
        
        # Initialize answer vocabulary
        self.answer_vocab = answer_vocab if answer_vocab is not None else {}
        if self.allow_vocab_updates and not self.answer_vocab:
            self.answer_vocab[UNK_ANSWER_TOKEN] = 0
        elif UNK_ANSWER_TOKEN not in self.answer_vocab:
            self.answer_vocab[UNK_ANSWER_TOKEN] = 0
        
        if allow_vocab_updates:
            self._populate_answer_vocab()
            self.allow_vocab_updates = False
       
        self.idx_to_answer = {idx: ans for ans, idx in self.answer_vocab.items()}
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer text for stable vocabulary lookups."""
        cleaned = self.text_preprocessor.clean_text(answer) if self.text_preprocessor else answer
        normalized = normalize_answer_text(cleaned)
        return normalized if normalized else UNK_ANSWER_TOKEN
    
    def _populate_answer_vocab(self):
        """Populate vocabulary with answers present in this split."""
        counts = Counter()
        for idx in self.indices:
            sample = self.dataset[idx]
            normalized = self._normalize_answer(sample.get('answer', ''))
            # FIXED: Don't exclude binary answers from counting
            # They still need to be in the vocab for proper indexing
            counts[normalized] += 1
        
        total = sum(counts.values()) or 1
        sorted_answers = counts.most_common()
        
        kept_answers = []
        cumulative = 0
        coverage_curve = []
        limit = self.max_answer_vocab_size or len(sorted_answers)
        
        for rank, (answer, count) in enumerate(sorted_answers, start=1):
            if answer == UNK_ANSWER_TOKEN:
                continue
            cumulative += count
            coverage = cumulative / total
            coverage_curve.append({
                'k': rank,
                'answer': answer,
                'count': count,
                'coverage': coverage
            })
            kept_answers.append(answer)
            if coverage >= self.coverage_threshold or len(kept_answers) >= limit:
                break
        
        self.coverage_curve = coverage_curve
        
        # Ensure UNK token exists
        if UNK_ANSWER_TOKEN not in self.answer_vocab:
            self.answer_vocab[UNK_ANSWER_TOKEN] = 0
        
        next_index = max(self.answer_vocab.values()) + 1 if self.answer_vocab else 1
        for answer in kept_answers:
            if answer not in self.answer_vocab:
                self.answer_vocab[answer] = next_index
                next_index += 1
    
    def _derive_question_type(self, answer: str) -> str:
        """Return 'binary' if answer is yes/no else 'open-ended'."""
        normalized = self._normalize_answer(answer)
        return 'binary' if normalized in BINARY_ANSWERS else 'open-ended'
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Get a single sample with combined preprocessing."""
        dataset_idx = self.indices[idx]
        sample = self.dataset[dataset_idx]
        
        # Get image
        image = sample['image']
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        image_tensor = self.image_preprocessor.prepare_image(
            image_np,
            is_training=(self.split == 'train'),
            preprocess_steps=self.preprocess_steps
        )
        
        # Get text data
        question = sample['question']
        answer = sample['answer']
        
        # Preprocess text
        if self.text_preprocessor:
            processed_question = self.text_preprocessor.preprocess_question(question)
        else:
            processed_question = {
                'input_ids': torch.tensor([0]),
                'attention_mask': torch.tensor([0]),
                'tokens': [],
                'medical_terms': {}
            }
        
        if self.encode_answers and self.text_preprocessor:
            processed_answer = self.text_preprocessor.preprocess_answer(answer)
            answer_cleaned = processed_answer['cleaned']
            answer_medical_terms = processed_answer.get('medical_terms', {})
        else:
            answer_cleaned = self.text_preprocessor.clean_text(answer) if self.text_preprocessor else (answer or "")
            answer_medical_terms = self.text_preprocessor.extract_medical_terms(answer_cleaned) if self.text_preprocessor else {}
            processed_answer = {
                'cleaned': answer_cleaned,
                'medical_terms': answer_medical_terms
            }
        
        # FIXED: Use consistent normalization
        normalized_answer = self._normalize_answer(answer)
        question_type = 'binary' if normalized_answer in BINARY_ANSWERS else 'open-ended'
        
        # Handle unknown answers
        if normalized_answer not in self.answer_vocab:
            if self.allow_vocab_updates:
                new_index = max(self.answer_vocab.values()) + 1
                self.answer_vocab[normalized_answer] = new_index
                self.idx_to_answer[new_index] = normalized_answer
            else:
                normalized_answer = UNK_ANSWER_TOKEN
        
        answer_index = self.answer_vocab.get(normalized_answer, self.answer_vocab.get(UNK_ANSWER_TOKEN, 0))
        
        answer_payload = {
            'text': answer,
            'cleaned': answer_cleaned,
            'medical_terms': answer_medical_terms,
            'tokens': processed_answer.get('tokens', [])
        }
        if self.encode_answers:
            answer_payload.update({
                'input_ids': processed_answer['input_ids'],
                'attention_mask': processed_answer['attention_mask'],
                'tokens': processed_answer.get('tokens', []),
                'pad_token_id': self.text_preprocessor.pad_token_id if self.text_preprocessor else 0
            })
        
        return {
            'image': image_tensor,
            'question': {
                'text': question,
                'input_ids': processed_question['input_ids'],
                'attention_mask': processed_question['attention_mask'],
                'tokens': processed_question.get('tokens', []),
                'medical_terms': processed_question.get('medical_terms', {}),
                'pad_token_id': self.text_preprocessor.pad_token_id if self.text_preprocessor else 0
            },
            'answer': answer_payload,
            'image_id': sample.get('image_id', idx),
            'question_id': sample.get('question_id', idx),
            'answer_idx': torch.tensor(answer_index, dtype=torch.long),
            'answer_normalized': normalized_answer,
            'question_type': question_type
        }


class CombinedVQADataModule:
    """Data module for combined image and text preprocessing."""
    
    def __init__(self, 
                 dataset_name: str = "flaviagiammarino/vqa-rad",
                 batch_size: int = 32,
                 num_workers: int = 4,
                 max_samples: Optional[int] = None,
                 question_types: Optional[List[str]] = None,
                 image_target_size: Tuple[int, int] = (224, 224),
                 text_model_name: str = "distilbert-base-uncased",
                 text_max_length: int = 40,
                 max_answer_vocab_size: Optional[int] = 500,
                 coverage_threshold: float = 0.95,
                 image_preprocess_steps: Optional[List[str]] = None,
                 encode_answers: bool = False,
                 split_seed: int = DEFAULT_SPLIT_SEED,
                 split_path: str = DEFAULT_SPLIT_PATH,
                 stats_max_samples: Optional[int] = 2000):
        
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_samples = max_samples
        self.question_types = question_types
        self.max_answer_vocab_size = max_answer_vocab_size
        self.coverage_threshold = coverage_threshold
        self.image_preprocess_steps = image_preprocess_steps or []
        self.encode_answers = encode_answers
        self.split_seed = split_seed
        self.split_path = split_path
        self.stats_max_samples = stats_max_samples
        
        # Initialize preprocessors
        self.image_preprocessor = MedicalImagePreprocessor(target_size=image_target_size)
        stats = get_or_compute_dataset_stats(
            dataset_name=self.dataset_name,
            split='train',
            target_size=image_target_size,
            preprocess_steps=self.image_preprocess_steps,
            max_samples=self.stats_max_samples
        )
        if stats:
            self.image_preprocessor.set_normalization_stats(stats['mean'], stats['std'])
        self.text_preprocessor = MedicalTextPreprocessor(
            model_name=text_model_name,
            max_length=text_max_length
        )
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.answer_vocab: Dict[str, int] = {UNK_ANSWER_TOKEN: 0}
        self.base_dataset = None
        self.test_base_dataset = None
        self.split_indices: Optional[Dict[str, List[int]]] = None
    
    def setup(self, stage: str = None, use_official_test_split: bool = True):
        """Setup datasets for different stages."""
        if not hasattr(self, 'test_base_dataset'):
            self.test_base_dataset = None
        
        if self.split_indices is None:
            self.split_indices = get_or_create_split_indices(
                self.dataset_name,
                base_dataset=self.base_dataset,
                split_path=self.split_path,
                seed=self.split_seed,
                use_official_test_split=use_official_test_split
            )
        
        if self.base_dataset is None:
            full_dataset = load_dataset(self.dataset_name)
            self.base_dataset = full_dataset['train']
            
            if self.split_indices.get('use_official_test', False):
                self.test_base_dataset = full_dataset['test']
        
        if stage == 'fit' or stage is None:
            self.train_dataset = CombinedVQADataset(
                dataset_name=self.dataset_name,
                split='train',
                image_preprocessor=self.image_preprocessor,
                text_preprocessor=self.text_preprocessor,
                max_samples=self.max_samples,
                question_types=self.question_types,
                answer_vocab=self.answer_vocab,
                allow_vocab_updates=True,
                max_answer_vocab_size=self.max_answer_vocab_size,
                preprocess_steps=self.image_preprocess_steps,
                encode_answers=self.encode_answers,
                coverage_threshold=self.coverage_threshold,
                base_dataset=self.base_dataset,
                indices=self.split_indices['train']
            )
            self.answer_vocab = self.train_dataset.answer_vocab
            
            self.val_dataset = CombinedVQADataset(
                dataset_name=self.dataset_name,
                split='val',
                image_preprocessor=self.image_preprocessor,
                text_preprocessor=self.text_preprocessor,
                max_samples=self.max_samples,
                question_types=self.question_types,
                answer_vocab=self.answer_vocab,
                allow_vocab_updates=False,
                max_answer_vocab_size=self.max_answer_vocab_size,
                preprocess_steps=self.image_preprocess_steps,
                encode_answers=self.encode_answers,
                coverage_threshold=self.coverage_threshold,
                base_dataset=self.base_dataset,
                indices=self.split_indices['val']
            )
        
        if stage == 'test' or stage is None:
            use_official = self.split_indices.get('use_official_test', False)
            
            if use_official and self.test_base_dataset is not None:
                test_base = self.test_base_dataset
                test_indices = list(range(len(self.test_base_dataset)))
            else:
                test_base = self.base_dataset
                test_indices = self.split_indices['test']
            
            self.test_dataset = CombinedVQADataset(
                dataset_name=self.dataset_name,
                split='test',
                image_preprocessor=self.image_preprocessor,
                text_preprocessor=self.text_preprocessor,
                max_samples=self.max_samples,
                question_types=self.question_types,
                answer_vocab=self.answer_vocab,
                allow_vocab_updates=False,
                max_answer_vocab_size=self.max_answer_vocab_size,
                preprocess_steps=self.image_preprocess_steps,
                encode_answers=self.encode_answers,
                coverage_threshold=self.coverage_threshold,
                base_dataset=test_base,
                indices=test_indices
            )
    
    def train_dataloader(self):
        """Create training data loader."""
        if self.train_dataset is None:
            self.setup('fit')
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn
        )
    
    def val_dataloader(self):
        """Create validation data loader."""
        if self.val_dataset is None:
            self.setup('fit')
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
    
    def test_dataloader(self):
        """Create test data loader."""
        if self.test_dataset is None:
            self.setup('test')
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )


def create_combined_data_loaders(dataset_name: str = "flaviagiammarino/vqa-rad",
                                 batch_size: int = 32,
                                 num_workers: int = 4,
                                 max_samples: Optional[int] = None,
                                 image_target_size: Tuple[int, int] = (224, 224),
                                 text_model_name: str = "distilbert-base-uncased",
                                 text_max_length: int = 40,
                                 max_answer_vocab_size: Optional[int] = 500,
                                 image_preprocess_steps: Optional[List[str]] = None,
                                 coverage_threshold: float = 0.95,
                                 encode_answers: bool = False,
                                 split_seed: int = DEFAULT_SPLIT_SEED,
                                 split_path: str = DEFAULT_SPLIT_PATH,
                                 stats_max_samples: Optional[int] = 2000,
                                 base_dataset=None,
                                 use_official_test_split: bool = True) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Dict[str, int], Dict[str, List[int]]]:
    """
    Create combined data loaders for training, validation, and testing.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, answer_vocab, split_indices)
    """
    
    data_module = CombinedVQADataModule(
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=max_samples,
        image_target_size=image_target_size,
        text_model_name=text_model_name,
        text_max_length=text_max_length,
        max_answer_vocab_size=max_answer_vocab_size,
        coverage_threshold=coverage_threshold,
        image_preprocess_steps=image_preprocess_steps,
        encode_answers=encode_answers,
        split_seed=split_seed,
        split_path=split_path,
        stats_max_samples=stats_max_samples
    )
    
    if base_dataset is not None:
        data_module.base_dataset = base_dataset
    
    data_module.setup(use_official_test_split=use_official_test_split)
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader() if data_module.test_dataset else None
    
    return train_loader, val_loader, test_loader, data_module.answer_vocab, data_module.split_indices


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences."""
    question_pad_token_id = batch[0]['question'].get('pad_token_id', 0)
    answer_pad_token_id = batch[0]['answer'].get('pad_token_id', question_pad_token_id)
    
    images = []
    question_texts = []
    answer_texts = []
    question_input_ids = []
    question_attention_masks = []
    question_tokens = []
    question_medical_terms = []
    answer_tokens = []
    answer_medical_terms = []
    image_ids = []
    question_ids = []
    answer_indices = []
    answer_normalized = []
    answer_cleaned = []
    question_types = []
    answer_input_ids = []
    answer_attention_masks = []
    
    for item in batch:
        images.append(item['image'])
        question_texts.append(item['question']['text'])
        answer_texts.append(item['answer']['text'])
        question_input_ids.append(item['question']['input_ids'])
        question_attention_masks.append(item['question']['attention_mask'])
        question_tokens.append(item['question']['tokens'])
        question_medical_terms.append(item['question']['medical_terms'])
        answer_cleaned.append(item['answer'].get('cleaned', item['answer']['text']))
        answer_medical_terms.append(item['answer'].get('medical_terms', {}))
        if 'tokens' in item['answer']:
            answer_tokens.append(item['answer']['tokens'])
        else:
            answer_tokens.append([])
        if 'input_ids' in item['answer']:
            answer_input_ids.append(item['answer']['input_ids'])
            answer_attention_masks.append(item['answer']['attention_mask'])
        image_ids.append(item['image_id'])
        question_ids.append(item['question_id'])
        answer_indices.append(item['answer_idx'])
        answer_normalized.append(item['answer_normalized'])
        question_types.append(item.get('question_type', 'unknown'))
    
    images = torch.stack(images)
    question_input_ids = pad_sequence(question_input_ids, batch_first=True, padding_value=question_pad_token_id)
    question_attention_masks = pad_sequence(question_attention_masks, batch_first=True, padding_value=0)
    answer_indices = torch.stack(answer_indices)
    
    answer_bundle = {
        'text': answer_texts,
        'cleaned': answer_cleaned,
        'medical_terms': answer_medical_terms,
        'tokens': answer_tokens
    }
    
    if answer_input_ids:
        answer_bundle['input_ids'] = pad_sequence(
            answer_input_ids, batch_first=True, padding_value=answer_pad_token_id
        )
        answer_bundle['attention_mask'] = pad_sequence(
            answer_attention_masks, batch_first=True, padding_value=0
        )
        answer_bundle['pad_token_id'] = answer_pad_token_id
    
    return {
        'image': images,
        'question': {
            'text': question_texts,
            'input_ids': question_input_ids,
            'attention_mask': question_attention_masks,
            'tokens': question_tokens,
            'medical_terms': question_medical_terms,
            'pad_token_id': question_pad_token_id
        },
        'answer': answer_bundle,
        'image_id': torch.tensor(image_ids),
        'question_id': torch.tensor(question_ids),
        'answer_idx': answer_indices,
        'answer_normalized': answer_normalized,
        'question_type': question_types
    }


if __name__ == "__main__":
    print("="*80)
    print("ANSWER NORMALIZATION VALIDATION")
    print("="*80)
    
    # Check <unk> rate for train split
    train_stats = check_unk_rate(split='train')
    
    # Check test split if available
    try:
        test_stats = check_unk_rate(split='test')
    except:
        print("\nTest split not available or error occurred")
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print("\n✅ If <unk> rate is < 1%, normalization is working well!")
    print("⚠️  If <unk> rate is > 1%, review the samples and add synonyms")