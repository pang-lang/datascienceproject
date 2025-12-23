#!/bin/bash
# Quick start script for Radiology VQA Streamlit App

echo "🏥 Starting Radiology VQA Streamlit App..."
echo ""
echo "Make sure you have:"
echo "  ✓ Trained model checkpoints in checkpoints/lightweight/ and checkpoints/baseline/"
echo "  ✓ Required packages: streamlit, torch, transformers, PIL, datasets"
echo ""

cd "/Users/panglang/Desktop/data science project/dua head copy"

# Check if checkpoints exist
if [ ! -f "checkpoints/lightweight/best_model.pt" ]; then
    echo "❌ Error: Lightweight model checkpoint not found!"
    echo "   Please train the model first using: python training/train_lightweight.py"
    exit 1
fi

if [ ! -f "checkpoints/baseline/best_model.pt" ]; then
    echo "⚠️  Warning: Baseline model checkpoint not found!"
    echo "   Only lightweight model will be available."
fi

echo "✅ Launching Streamlit app..."
echo ""
streamlit run radvqa_streamlit.py

