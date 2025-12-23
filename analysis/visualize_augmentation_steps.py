#!/usr/bin/env python3
"""
Visualize image preprocessing and augmentation steps stage-by-stage.
Perfect for creating presentation slides showing the data pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageEnhance
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from pathlib import Path
import sys
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
DATA_SPLITS_PATH = PROJECT_ROOT / "data_splits/vqa_rad_seed42.json"
OUTPUT_DIR = PROJECT_ROOT / "augmentation_visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

# Normalization stats (ImageNet defaults)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def denormalize_tensor(tensor, mean=MEAN, std=STD):
    """Convert normalized tensor back to displayable image."""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    tensor = torch.clamp(tensor, 0, 1)
    return F.to_pil_image(tensor)


def visualize_all_steps(image_path, output_prefix="augmentation_steps", seed=42):
    """
    Visualize all preprocessing and augmentation steps.
    
    Args:
        image_path: Path to the input image
        output_prefix: Prefix for output files
        seed: Random seed for reproducibility
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load original image
    original_img = Image.open(image_path).convert('RGB')
    
    print(f"Original image size: {original_img.size}")
    
    # Create a list to store (image, title, description) tuples
    steps = []
    
    # ========== Step 1: Original Image ==========
    steps.append((
        original_img,
        "1. Original Image",
        f"Raw medical image\nSize: {original_img.size[0]}×{original_img.size[1]} pixels"
    ))
    
    # ========== Step 2: Resize ==========
    resized_img = F.resize(original_img, (224, 224))
    steps.append((
        resized_img,
        "2. Resize",
        "Resized to 224×224 pixels\nStandard input size for CNNs"
    ))
    
    # ========== Step 3: Horizontal Flip ==========
    flipped_img = F.hflip(resized_img)
    steps.append((
        flipped_img,
        "3. Horizontal Flip",
        "Random horizontal flip (p=0.5)\nIncreases left-right invariance"
    ))
    
    # ========== Step 4: Rotation ==========
    rotated_img = F.rotate(resized_img, angle=10)  # Show with 10 degrees for visibility
    steps.append((
        rotated_img,
        "4. Random Rotation",
        "Random rotation (±3°)\nHandles slight angle variations"
    ))
    
    # ========== Step 5: Affine Transform ==========
    # Apply translate, scale, shear
    affine_img = F.affine(
        resized_img,
        angle=0,
        translate=(10, 5),  # 5% translation
        scale=1.05,
        shear=5
    )
    steps.append((
        affine_img,
        "5. Affine Transform",
        "Translation, scaling, shearing\nHandles positioning variations"
    ))
    
    # ========== Step 6: Color Jitter - Brightness ==========
    brightness_img = F.adjust_brightness(resized_img, brightness_factor=1.3)
    steps.append((
        brightness_img,
        "6a. Brightness Adjustment",
        "Random brightness (±20%)\nHandles lighting variations"
    ))
    
    # ========== Step 7: Color Jitter - Contrast ==========
    contrast_img = F.adjust_contrast(resized_img, contrast_factor=1.3)
    steps.append((
        contrast_img,
        "6b. Contrast Adjustment",
        "Random contrast (±20%)\nHandles exposure differences"
    ))
    
    # ========== Step 8: Color Jitter - Saturation ==========
    saturation_img = F.adjust_saturation(resized_img, saturation_factor=1.3)
    steps.append((
        saturation_img,
        "6c. Saturation Adjustment",
        "Random saturation (±20%)\nHandles color intensity"
    ))
    
    # ========== Step 9: All Augmentations Combined ==========
    # Apply all augmentations together (as in training)
    combined_img = resized_img
    combined_img = F.hflip(combined_img)  # Flip
    combined_img = F.rotate(combined_img, angle=2)  # Small rotation
    combined_img = F.affine(combined_img, angle=0, translate=(5, 3), scale=1.02, shear=3)
    combined_img = F.adjust_brightness(combined_img, 1.15)
    combined_img = F.adjust_contrast(combined_img, 1.1)
    combined_img = F.adjust_saturation(combined_img, 1.05)
    
    steps.append((
        combined_img,
        "7. Combined Augmentations",
        "All augmentations applied together\n(as used during training)"
    ))
    
    # ========== Step 10: Normalized Tensor ==========
    # Convert to tensor and normalize
    tensor_img = F.to_tensor(resized_img)
    normalized_tensor = F.normalize(tensor_img, mean=MEAN, std=STD)
    denormalized_img = denormalize_tensor(normalized_tensor)
    
    steps.append((
        denormalized_img,
        "8. Normalized (Denormalized for Display)",
        "Normalized using ImageNet stats\nμ=[0.485, 0.456, 0.406]\nσ=[0.229, 0.224, 0.225]"
    ))
    
    # ========== Create Visualizations ==========
    
    # Visualization 1: All steps in a grid
    create_grid_visualization(steps, output_prefix)
    
    # Visualization 2: Sequential comparison (original vs augmented)
    create_comparison_visualization(steps, output_prefix)
    
    # Visualization 3: Individual step cards for slides
    create_individual_slides(steps, output_prefix)
    
    # Visualization 4: Pipeline flowchart
    create_pipeline_flowchart(steps, output_prefix)
    
    print(f"\n✅ All visualizations saved to: {OUTPUT_DIR}/")
    print(f"   - {output_prefix}_grid.png (overview)")
    print(f"   - {output_prefix}_comparison.png (before/after)")
    print(f"   - {output_prefix}_pipeline.png (flowchart)")
    print(f"   - Individual slides in: {output_prefix}_slides/")


def create_grid_visualization(steps, output_prefix):
    """Create a grid showing all steps."""
    n_steps = len(steps)
    n_cols = 4
    n_rows = (n_steps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, (img, title, desc) in enumerate(steps):
        axes[idx].imshow(img)
        axes[idx].set_title(title, fontsize=11, fontweight='bold', pad=10)
        axes[idx].axis('off')
        
        # Add description as text below image
        axes[idx].text(0.5, -0.1, desc, transform=axes[idx].transAxes,
                      ha='center', va='top', fontsize=8, style='italic',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Hide unused subplots
    for idx in range(len(steps), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Image Preprocessing & Augmentation Pipeline', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    
    output_path = OUTPUT_DIR / f"{output_prefix}_grid.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved grid visualization: {output_path}")
    plt.close()


def create_comparison_visualization(steps, output_prefix):
    """Create before/after comparisons."""
    # Show original vs key augmented versions
    comparisons = [
        (0, 2),  # Original vs Flip
        (0, 3),  # Original vs Rotation
        (0, 5),  # Original vs Brightness
        (0, 8),  # Original vs Combined
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for col, (before_idx, after_idx) in enumerate(comparisons):
        # Before
        axes[0, col].imshow(steps[before_idx][0])
        axes[0, col].set_title(steps[before_idx][1], fontsize=11, fontweight='bold')
        axes[0, col].axis('off')
        
        # After
        axes[1, col].imshow(steps[after_idx][0])
        axes[1, col].set_title(steps[after_idx][1], fontsize=11, fontweight='bold')
        axes[1, col].axis('off')
    
    # Add row labels
    fig.text(0.02, 0.75, 'Before', rotation=90, fontsize=14, 
             fontweight='bold', va='center')
    fig.text(0.02, 0.25, 'After', rotation=90, fontsize=14, 
             fontweight='bold', va='center')
    
    plt.suptitle('Before & After: Key Augmentation Steps', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0.03, 0, 1, 0.96])
    
    output_path = OUTPUT_DIR / f"{output_prefix}_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved comparison visualization: {output_path}")
    plt.close()


def create_individual_slides(steps, output_prefix):
    """Create individual images for each step (for PowerPoint slides)."""
    slides_dir = OUTPUT_DIR / f"{output_prefix}_slides"
    slides_dir.mkdir(exist_ok=True)
    
    for idx, (img, title, desc) in enumerate(steps):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(img)
        ax.axis('off')
        
        # Add title above
        fig.text(0.5, 0.95, title, ha='center', fontsize=16, 
                fontweight='bold', transform=fig.transFigure)
        
        # Add description below
        fig.text(0.5, 0.05, desc, ha='center', fontsize=11, 
                style='italic', transform=fig.transFigure,
                bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.92])
        
        output_path = slides_dir / f"step_{idx:02d}_{title.split('.')[1].strip().replace(' ', '_').lower()}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    print(f"✅ Saved {len(steps)} individual slide images to: {slides_dir}/")


def create_pipeline_flowchart(steps, output_prefix):
    """Create a flowchart showing the pipeline."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.axis('off')
    
    # Define positions for flowchart boxes
    y_start = 0.95
    y_step = 0.85 / len(steps)
    box_width = 0.7
    box_height = y_step * 0.7
    
    for idx, (img, title, desc) in enumerate(steps):
        y_pos = y_start - idx * y_step
        
        # Draw box
        rect = mpatches.FancyBboxPatch(
            (0.15, y_pos - box_height/2), box_width, box_height,
            boxstyle="round,pad=0.01", 
            edgecolor='steelblue', facecolor='lightblue', 
            alpha=0.3, linewidth=2,
            transform=ax.transAxes
        )
        ax.add_patch(rect)
        
        # Add title text
        ax.text(0.25, y_pos, title, transform=ax.transAxes,
               fontsize=12, fontweight='bold', va='center')
        
        # Add description text
        ax.text(0.25, y_pos - 0.02, desc, transform=ax.transAxes,
               fontsize=9, style='italic', va='top', color='gray')
        
        # Add small thumbnail
        thumbnail_ax = fig.add_axes([0.88, y_pos - box_height/2 + 0.005, 0.1, box_height - 0.01])
        thumbnail_ax.imshow(img)
        thumbnail_ax.axis('off')
        
        # Add arrow to next step
        if idx < len(steps) - 1:
            arrow = mpatches.FancyArrowPatch(
                (0.5, y_pos - box_height/2 - 0.01), 
                (0.5, y_pos - y_step + box_height/2 + 0.01),
                transform=ax.transAxes,
                arrowstyle='->', mutation_scale=30, linewidth=2,
                color='steelblue'
            )
            ax.add_patch(arrow)
    
    # Title
    fig.text(0.5, 0.98, 'Data Augmentation Pipeline', 
            ha='center', fontsize=18, fontweight='bold')
    
    output_path = OUTPUT_DIR / f"{output_prefix}_pipeline.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved pipeline flowchart: {output_path}")
    plt.close()


def main():
    """Main function to generate visualizations."""
    print("="*80)
    print("IMAGE AUGMENTATION VISUALIZATION FOR SLIDES")
    print("="*80)
    
    # Load the actual dataset
    from datasets import load_dataset
    
    print("\nLoading VQA-RAD dataset...")
    dataset = load_dataset("flaviagiammarino/vqa-rad", cache_dir=str(PROJECT_ROOT / "hf_cache"))
    
    # Get training sample
    sample = dataset['train'][2]
    image = sample['image']
    question = sample['question']
    answer = sample['answer']
    
    # Save the image temporarily
    temp_image_path = OUTPUT_DIR / "temp_sample.jpg"
    image.save(temp_image_path)
    
    print(f"\nUsing sample from VQA-RAD training set:")
    print(f"  Question: {question}")
    print(f"  Answer: {answer}")
    print(f"  Image size: {image.size}")
    print()
    
    # Generate all visualizations
    visualize_all_steps(str(temp_image_path), output_prefix="augmentation_steps", seed=42)
    
    # Clean up temp file
    temp_image_path.unlink()
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\n📁 All files saved to: {OUTPUT_DIR}/")
    print("\n💡 Usage for your slides:")
    print("   1. Use 'augmentation_steps_grid.png' for overview")
    print("   2. Use 'augmentation_steps_comparison.png' for before/after")
    print("   3. Use 'augmentation_steps_pipeline.png' for pipeline flowchart")
    print("   4. Use individual images from 'augmentation_steps_slides/' folder")
    print("      for step-by-step explanation")
    print()


if __name__ == "__main__":
    main()

