#!/usr/bin/env python3
"""
Benchmark inference times for Lightweight and Baseline VQA models.
This script measures inference speed with warm-up runs and provides statistics.
"""

import torch
import time
import json
import numpy as np
from pathlib import Path
import sys
from PIL import Image
from torchvision import transforms

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.lightweight_model import DualHeadVQAModel
from models.baseline_model import DualHeadBaselineVQAModel
from transformers import DistilBertTokenizer, BertTokenizer

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
LIGHTWEIGHT_CHECKPOINT = PROJECT_ROOT / "checkpoints/lightweight/best_model.pt"
BASELINE_CHECKPOINT = PROJECT_ROOT / "checkpoints/baseline/best_model.pt"

# Benchmark settings
NUM_WARMUP_RUNS = 10      # Warm-up runs (not counted)
NUM_BENCHMARK_RUNS = 100  # Actual benchmark runs

print("="*80)
print("VQA MODEL INFERENCE BENCHMARK")
print("="*80)
print(f"Device: {DEVICE}")
print(f"Warm-up runs: {NUM_WARMUP_RUNS}")
print(f"Benchmark runs: {NUM_BENCHMARK_RUNS}")
print("="*80)


def load_lightweight_model():
    """Load lightweight dual-head model from checkpoint."""
    checkpoint = torch.load(LIGHTWEIGHT_CHECKPOINT, map_location=DEVICE, weights_only=False)
    config = checkpoint.get('config', {})
    num_open_ended_classes = config.get('num_open_ended_classes', 121)
    
    model = DualHeadVQAModel(
        num_open_ended_classes=num_open_ended_classes,
        fusion_hidden_dim=config.get('fusion_hidden_dim', 256),
        num_attention_heads=config.get('num_attention_heads', 4),
        dropout=config.get('dropout', 0.3),
        freeze_vision_encoder=False,
        freeze_text_encoder=False
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    return model, tokenizer, num_open_ended_classes


def load_baseline_model():
    """Load baseline dual-head model from checkpoint."""
    checkpoint = torch.load(BASELINE_CHECKPOINT, map_location=DEVICE, weights_only=False)
    config = checkpoint.get('config', {})
    num_open_ended_classes = config.get('num_open_ended_classes', 121)
    
    model = DualHeadBaselineVQAModel(
        num_open_ended_classes=num_open_ended_classes,
        fusion_hidden_dim=config.get('fusion_hidden_dim', 512),
        num_attention_heads=config.get('num_attention_heads', 8),
        dropout=config.get('dropout', 0.35),
        freeze_vision_encoder=False,
        freeze_text_encoder=False
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    return model, tokenizer, num_open_ended_classes


def create_dummy_inputs(tokenizer, batch_size=1):
    """Create dummy inputs for benchmarking."""
    # Create dummy image
    images = torch.randn(batch_size, 3, 224, 224).to(DEVICE)
    
    # Create dummy question
    question = "What abnormalities are visible in this image?"
    encoded = tokenizer(
        question,
        padding='max_length',
        truncation=True,
        max_length=64,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(DEVICE)
    attention_mask = encoded['attention_mask'].to(DEVICE)
    
    return images, input_ids, attention_mask


def benchmark_model(model, tokenizer, model_name, warmup_runs=10, benchmark_runs=100):
    """
    Benchmark inference time for a model.
    
    Returns:
        dict with timing statistics
    """
    print(f"\n{'='*80}")
    print(f"BENCHMARKING: {model_name}")
    print(f"{'='*80}")
    
    # Create dummy inputs
    images, input_ids, attention_mask = create_dummy_inputs(tokenizer)
    
    # Warm-up runs (to stabilize GPU/CPU performance)
    print(f"Running {warmup_runs} warm-up iterations...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            # Dual-head models return dict, we just need to run forward pass
            _ = model(images, input_ids, attention_mask)
    
    # Synchronize (important for GPU timing)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    elif DEVICE.type == 'mps':
        torch.mps.synchronize()
    
    # Benchmark runs
    print(f"Running {benchmark_runs} benchmark iterations...")
    inference_times = []
    
    with torch.no_grad():
        for i in range(benchmark_runs):
            # Start timer
            start_time = time.perf_counter()
            
            # Forward pass
            _ = model(images, input_ids, attention_mask)
            
            # Synchronize before stopping timer
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            elif DEVICE.type == 'mps':
                torch.mps.synchronize()
            
            # Stop timer
            end_time = time.perf_counter()
            
            inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
            inference_times.append(inference_time)
            
            # Progress indicator
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{benchmark_runs} iterations")
    
    # Calculate statistics
    inference_times = np.array(inference_times)
    stats = {
        'model_name': model_name,
        'device': str(DEVICE),
        'num_runs': benchmark_runs,
        'mean_ms': float(np.mean(inference_times)),
        'std_ms': float(np.std(inference_times)),
        'median_ms': float(np.median(inference_times)),
        'min_ms': float(np.min(inference_times)),
        'max_ms': float(np.max(inference_times)),
        'p95_ms': float(np.percentile(inference_times, 95)),
        'p99_ms': float(np.percentile(inference_times, 99)),
    }
    
    # Print results
    print(f"\n{'─'*80}")
    print(f"RESULTS: {model_name}")
    print(f"{'─'*80}")
    print(f"Mean inference time:   {stats['mean_ms']:>8.3f} ms ± {stats['std_ms']:.3f} ms")
    print(f"Median inference time: {stats['median_ms']:>8.3f} ms")
    print(f"Min inference time:    {stats['min_ms']:>8.3f} ms")
    print(f"Max inference time:    {stats['max_ms']:>8.3f} ms")
    print(f"95th percentile:       {stats['p95_ms']:>8.3f} ms")
    print(f"99th percentile:       {stats['p99_ms']:>8.3f} ms")
    print(f"Throughput:            {1000/stats['mean_ms']:>8.2f} images/second")
    
    return stats


def main():
    """Run benchmarks for both models."""
    results = {}
    
    # Benchmark Lightweight Model
    try:
        print("\n" + "="*80)
        print("LOADING LIGHTWEIGHT MODEL")
        print("="*80)
        model_light, tokenizer_light, num_open_ended_classes_light = load_lightweight_model()
        print(f"✅ Loaded successfully (num_open_ended_classes={num_open_ended_classes_light})")
        
        params_light = sum(p.numel() for p in model_light.parameters())
        print(f"   Total parameters: {params_light:,}")
        
        results['lightweight'] = benchmark_model(
            model_light, 
            tokenizer_light, 
            "Lightweight Model (MobileNetV3-Small + DistilBERT)",
            warmup_runs=NUM_WARMUP_RUNS,
            benchmark_runs=NUM_BENCHMARK_RUNS
        )
        results['lightweight']['num_parameters'] = params_light
        
    except Exception as e:
        print(f"❌ Error benchmarking lightweight model: {e}")
        import traceback
        traceback.print_exc()
    
    # Benchmark Baseline Model
    try:
        print("\n" + "="*80)
        print("LOADING BASELINE MODEL")
        print("="*80)
        model_base, tokenizer_base, num_open_ended_classes_base = load_baseline_model()
        print(f"✅ Loaded successfully (num_open_ended_classes={num_open_ended_classes_base})")
        
        params_base = sum(p.numel() for p in model_base.parameters())
        print(f"   Total parameters: {params_base:,}")
        
        results['baseline'] = benchmark_model(
            model_base,
            tokenizer_base,
            "Baseline Model (ResNet-34 + BERT-base)",
            warmup_runs=NUM_WARMUP_RUNS,
            benchmark_runs=NUM_BENCHMARK_RUNS
        )
        results['baseline']['num_parameters'] = params_base
        
    except Exception as e:
        print(f"❌ Error benchmarking baseline model: {e}")
        import traceback
        traceback.print_exc()
    
    # Comparison Summary
    if 'lightweight' in results and 'baseline' in results:
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        
        speedup = results['baseline']['mean_ms'] / results['lightweight']['mean_ms']
        param_ratio = results['baseline']['num_parameters'] / results['lightweight']['num_parameters']
        
        print(f"\nInference Time:")
        print(f"  Lightweight: {results['lightweight']['mean_ms']:.3f} ms")
        print(f"  Baseline:    {results['baseline']['mean_ms']:.3f} ms")
        print(f"  Speedup:     {speedup:.2f}x faster")
        
        print(f"\nModel Size:")
        print(f"  Lightweight: {results['lightweight']['num_parameters']:,} parameters")
        print(f"  Baseline:    {results['baseline']['num_parameters']:,} parameters")
        print(f"  Reduction:   {param_ratio:.2f}x smaller")
        
        print(f"\nThroughput:")
        print(f"  Lightweight: {1000/results['lightweight']['mean_ms']:.2f} images/second")
        print(f"  Baseline:    {1000/results['baseline']['mean_ms']:.2f} images/second")
    
    # Save results to JSON
    output_file = PROJECT_ROOT / "benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ Results saved to: {output_file}")
    print(f"{'='*80}\n")
    
    # Generate LaTeX table for report
    generate_latex_table(results)


def generate_latex_table(results):
    """Generate LaTeX table for report."""
    latex_file = PROJECT_ROOT / "benchmark_table.tex"
    
    with open(latex_file, 'w') as f:
        f.write("% Inference Time Comparison Table\n")
        f.write("% Copy this into your LaTeX report\n\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Inference Time Comparison}\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Metric} & \\textbf{Lightweight} & \\textbf{Baseline} \\\\\n")
        f.write("\\hline\n")
        
        if 'lightweight' in results and 'baseline' in results:
            f.write(f"Mean Inference Time (ms) & {results['lightweight']['mean_ms']:.2f} $\\pm$ {results['lightweight']['std_ms']:.2f} & {results['baseline']['mean_ms']:.2f} $\\pm$ {results['baseline']['std_ms']:.2f} \\\\\n")
            f.write(f"Median Inference Time (ms) & {results['lightweight']['median_ms']:.2f} & {results['baseline']['median_ms']:.2f} \\\\\n")
            f.write(f"Parameters & {results['lightweight']['num_parameters']:,} & {results['baseline']['num_parameters']:,} \\\\\n")
            f.write(f"Throughput (images/s) & {1000/results['lightweight']['mean_ms']:.2f} & {1000/results['baseline']['mean_ms']:.2f} \\\\\n")
            
            speedup = results['baseline']['mean_ms'] / results['lightweight']['mean_ms']
            f.write(f"Speedup & {speedup:.2f}x & 1.00x \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write(f"\\label{{tab:inference_benchmark}}\n")
        f.write("\\end{table}\n")
    
    print(f"✅ LaTeX table saved to: {latex_file}")


if __name__ == "__main__":
    main()

