#!/usr/bin/env python3
"""
Save answer vocabulary from training to JSON file for deployment.
This ensures the Streamlit app uses the exact same vocabulary as training.
"""

import json
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.combined_preprocessing import create_combined_data_loaders

def save_answer_vocab():
    """Load and save answer vocabulary to JSON file."""
    print("Loading answer vocabulary from training data...")
    
    try:
        # Load data to get the answer vocabulary
        _, _, _, answer_vocab, _ = create_combined_data_loaders(
            dataset_name="flaviagiammarino/vqa-rad",
            batch_size=1,
            num_workers=0,
            max_samples=None,
            split_seed=42,
            use_official_test_split=True
        )
        
        print(f"✅ Loaded vocabulary with {len(answer_vocab)} answers")
        
        # Save to JSON file
        output_file = PROJECT_ROOT / "answer_vocab.json"
        with open(output_file, 'w') as f:
            json.dump(answer_vocab, f, indent=2)
        
        print(f"✅ Saved answer vocabulary to: {output_file}")
        print(f"\nTop 10 most common answers:")
        # Sort by index to see the order
        sorted_answers = sorted(answer_vocab.items(), key=lambda x: x[1])
        for answer, idx in sorted_answers[:10]:
            print(f"  {idx}: {answer}")
        
        print(f"\n📤 NEXT STEP:")
        print(f"Upload '{output_file.name}' to your HuggingFace repository:")
        print(f"  https://huggingface.co/daphne04/radvqa-lightweight")
        print(f"\nClick 'Add file' → 'Upload files' → Select 'answer_vocab.json'")
        
        return answer_vocab
        
    except Exception as e:
        print(f"❌ Error loading vocabulary: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    save_answer_vocab()

