import streamlit as st
from PIL import Image
import io
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import json
import sys
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.lightweight_model import DualHeadVQAModel
from models.baseline_model import DualHeadBaselineVQAModel
from transformers import DistilBertTokenizer, BertTokenizer

# Set page configuration
st.set_page_config(
    page_title="Radiology VQA",
    page_icon="🏥",
    layout="wide"
)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# Paths
LIGHTWEIGHT_CHECKPOINT = PROJECT_ROOT / "checkpoints/lightweight/best_model.pt"
BASELINE_CHECKPOINT = PROJECT_ROOT / "checkpoints/baseline/best_model.pt"
# Note: Answer vocab is now loaded from preprocessing data directly


@st.cache_resource
def load_answer_vocab():
    """
    Load and build answer vocabulary from preprocessing data.
    This uses the same preprocessing logic as training.
    """
    from preprocessing.combined_preprocessing import (
        create_combined_data_loaders, 
        normalize_answer_text,
        BINARY_ANSWERS
    )
    
    # Load a small portion of data just to get the answer vocab
    # Use the same split configuration as training
    try:
        _, _, _, answer_vocab, _ = create_combined_data_loaders(
            dataset_name="flaviagiammarino/vqa-rad",
            batch_size=1,
            num_workers=0,
            max_samples=None,  # Load all to get complete vocab
            split_seed=42,
            use_official_test_split=True
        )
        
        idx_to_answer = {idx: ans for ans, idx in answer_vocab.items()}
        
        return answer_vocab, idx_to_answer, BINARY_ANSWERS
    except Exception as e:
        st.error(f"Failed to load answer vocabulary: {e}")
        raise


@st.cache_resource
def load_lightweight_model():
    """Load the lightweight dual-head VQA model."""
    # Load checkpoint first to get the config
    checkpoint = torch.load(LIGHTWEIGHT_CHECKPOINT, map_location=DEVICE, weights_only=False)
    
    # Extract config from checkpoint
    config = checkpoint.get('config', {})
    num_open_ended_classes = config.get('num_open_ended_classes', 121)
    fusion_hidden_dim = config.get('fusion_hidden_dim', 256)
    num_attention_heads = config.get('num_attention_heads', 4)
    dropout = config.get('dropout', 0.3)
    
    # Initialize dual-head model with checkpoint config
    model = DualHeadVQAModel(
        num_open_ended_classes=num_open_ended_classes,
        fusion_hidden_dim=fusion_hidden_dim,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
        freeze_vision_encoder=False,
        freeze_text_encoder=False
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    return model, tokenizer, num_open_ended_classes


@st.cache_resource
def load_baseline_model():
    """Load the baseline dual-head VQA model."""
    # Load checkpoint first to get the config
    checkpoint = torch.load(BASELINE_CHECKPOINT, map_location=DEVICE, weights_only=False)
    
    # Extract config from checkpoint
    config = checkpoint.get('config', {})
    num_open_ended_classes = config.get('num_open_ended_classes', 121)
    fusion_hidden_dim = config.get('fusion_hidden_dim', 512)
    num_attention_heads = config.get('num_attention_heads', 8)
    dropout = config.get('dropout', 0.35)
    
    # Initialize dual-head model with checkpoint config
    model = DualHeadBaselineVQAModel(
        num_open_ended_classes=num_open_ended_classes,
        fusion_hidden_dim=fusion_hidden_dim,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
        freeze_vision_encoder=False,
        freeze_text_encoder=False
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    return model, tokenizer, num_open_ended_classes


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model input."""
    # Define the same transforms used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    img_tensor = transform(image)
    return img_tensor.unsqueeze(0)  # Add batch dimension


def preprocess_question(question: str, tokenizer, max_length: int = 64) -> dict:
    """Tokenize and preprocess question."""
    encoded = tokenizer(
        question,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return encoded


def is_binary_question(question: str) -> bool:
    """
    Heuristically determine if a question is likely binary (yes/no).
    
    This is a simple heuristic - for best results, you could train a classifier
    or use more sophisticated NLP, but this works reasonably well.
    """
    question_lower = question.lower().strip()
    
    # Common patterns for binary questions
    binary_indicators = [
        'is there', 'are there', 'is this', 'are these',
        'does this', 'do these', 'can you see',
        'is it', 'are they', 'was there', 'were there',
        'has the', 'have the', 'did the',
        'is the', 'are the'
    ]
    
    # Check if starts with binary indicator
    for indicator in binary_indicators:
        if question_lower.startswith(indicator):
            return True
    
    # Check for question words that typically lead to open-ended questions
    open_ended_words = ['what', 'which', 'where', 'when', 'who', 'how', 'why', 'describe']
    for word in open_ended_words:
        if question_lower.startswith(word):
            return False
    
    # Default to binary if it ends with a question mark and is relatively short
    if question_lower.endswith('?') and len(question.split()) < 10:
        return True
    
    return False


def predict(model, image_tensor, question_encoding, question_text, idx_to_answer, binary_answers, top_k=5):
    """
    Run inference on dual-head model and return predictions with confidence scores.
    
    Args:
        model: Dual-head VQA model
        image_tensor: Preprocessed image tensor
        question_encoding: Tokenized question
        question_text: Original question text (for binary detection)
        idx_to_answer: Mapping from indices to answers
        binary_answers: Set of binary answers (yes, no)
        top_k: Number of top predictions to return
    
    Returns:
        answer: predicted answer string (or "Not confident" for <unk>)
        confidence: confidence score (0-1)
        top_k_predictions: list of (answer, confidence) tuples
        attention_weights: cross-modal attention weights (if available)
        is_binary: whether the question was classified as binary
    """
    with torch.no_grad():
        # Move inputs to device
        image_tensor = image_tensor.to(DEVICE)
        input_ids = question_encoding['input_ids'].to(DEVICE)
        attention_mask = question_encoding['attention_mask'].to(DEVICE)
        
        # Determine if question is binary
        is_binary = is_binary_question(question_text)
        
        # Forward pass with attention weights (dual-head models return dict)
        # Different models use different parameter names:
        # - Lightweight: return_features=True
        # - Baseline: return_attention=True
        try:
            # Try baseline model signature first
            outputs = model(image_tensor, input_ids, attention_mask, return_attention=True)
        except TypeError:
            # Fall back to lightweight model signature
            outputs = model(image_tensor, input_ids, attention_mask, return_features=True)
        
        # Extract the appropriate head's outputs
        if is_binary:
            logits = outputs['binary']  # Shape: (batch, 2)
            # Build binary-specific idx_to_answer mapping
            current_idx_to_answer = {0: 'no', 1: 'yes'}
        else:
            logits = outputs['open_ended']  # Shape: (batch, num_open_ended_classes)
            # Use the full answer vocabulary
            current_idx_to_answer = idx_to_answer
        
        # Extract attention weights (key depends on model version)
        attention_weights = outputs.get('attention', outputs.get('attention_weights'))
        
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get top prediction
        top_prob, top_idx = torch.max(probs, dim=1)
        predicted_idx = top_idx.item()
        predicted_answer = current_idx_to_answer.get(predicted_idx, '<unk>')
        confidence = top_prob.item()
        
        # Replace <unk> with "Not confident" for UI display
        if predicted_answer == '<unk>' or predicted_answer.startswith('<unknown'):
            display_answer = "Not confident"
        else:
            display_answer = predicted_answer
        
        # Get top-k predictions
        top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, probs.size(1)), dim=1)
        top_k_predictions = []
        for idx, prob in zip(top_k_indices[0], top_k_probs[0]):
            ans = current_idx_to_answer.get(idx.item(), '<unk>')
            # Replace <unk> for display
            if ans == '<unk>' or ans.startswith('<unknown'):
                ans = "Not confident"
            top_k_predictions.append((ans, prob.item()))
        
    return display_answer, confidence, top_k_predictions, attention_weights, is_binary


def visualize_cross_modal_attention(attention_weights, question_text):
    """
    Visualize cross-modal attention weights showing how the model
    weighs visual vs. textual information.
    
    Args:
        attention_weights: Attention weights from the fusion module
                          Shape: (batch, seq_len, seq_len) where seq_len=2 for [visual, text]
                          Note: PyTorch's MultiheadAttention returns weights averaged across heads
        question_text: The question text for display
    
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    # Extract attention weights for first sample and convert to numpy
    attn = attention_weights[0].cpu().numpy()  # (2, 2) where 2 = [visual, text]
    
    # Check if we need to average across heads (shape would be (num_heads, 2, 2))
    if attn.ndim == 3:
        # Average across heads
        attn_avg = attn.mean(axis=0)  # (2, 2)
    elif attn.ndim == 2:
        # Already averaged across heads
        attn_avg = attn  # (2, 2)
    else:
        raise ValueError(f"Unexpected attention weights shape: {attn.shape}. Expected (2, 2) or (num_heads, 2, 2)")
    
    # Verify final shape
    if attn_avg.shape != (2, 2):
        raise ValueError(f"Invalid attention matrix shape: {attn_avg.shape}. Expected (2, 2)")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # ===== Subplot 1: Attention Matrix Heatmap =====
    im = ax1.imshow(attn_avg, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Visual', 'Text'], fontsize=11)
    ax1.set_yticklabels(['Visual', 'Text'], fontsize=11)
    ax1.set_xlabel('Key', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Query', fontsize=12, fontweight='bold')
    ax1.set_title('Cross-Modal Attention Matrix\n(Averaged across heads)', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax1.text(j, i, f'{attn_avg[i, j]:.3f}',
                           ha="center", va="center", color="black", fontsize=11, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', fontsize=10)
    
    # ===== Subplot 2: Modality Importance Bar Chart =====
    # Calculate modality importance as the average attention TO each modality
    visual_importance = attn_avg[:, 0].mean()  # How much attention to visual
    text_importance = attn_avg[:, 1].mean()    # How much attention to text
    
    modalities = ['Visual\nFeatures', 'Text\nFeatures']
    importances = [visual_importance, text_importance]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = ax2.bar(modalities, importances, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Average Attention Weight', fontsize=12, fontweight='bold')
    ax2.set_title('Modality Importance', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, val in zip(bars, importances):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1%}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add interpretation text
    dominant = "Visual" if visual_importance > text_importance else "Text"
    ratio = max(visual_importance, text_importance) / min(visual_importance, text_importance)
    
    interpretation = f"The model relies more on {dominant.lower()} information ({ratio:.1f}x)"
    fig.text(0.5, 0.02, interpretation, ha='center', fontsize=10, 
             style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    return fig

# Title
st.title("🏥 Radiology Visual Question Answering")

# About section
st.markdown("""
This application allows you to upload radiology images (X-ray, CT, MRI) and ask questions about them 
to get AI-powered answers.

""")

st.markdown("---")

# Load answer vocabulary
try:
    answer_vocab, idx_to_answer, binary_answers = load_answer_vocab()
    # st.success(f"✅ Loaded vocabulary with {len(answer_vocab)} answers")
except Exception as e:
    st.error(f"❌ Error loading answer vocabulary: {e}")
    st.stop()

# Model Selection
st.subheader("⚙️ Model Selection")

col1, col2 = st.columns([4,1])  # left wider, right narrow

with col2:
    show_attention = st.checkbox(
        "Show Attention Visualization",
        value=True,
        help="Display cross-modal attention showing how the model weighs visual vs. text information"
    )


model_type = st.selectbox(
    "Choose Model:",
    ["Lightweight Model", "Baseline Model"],
    help="Select between lightweight (faster, smaller) or baseline (more accurate) model"
)

# Load selected model
try:
    with st.spinner(f"Loading {model_type}..."):
        if model_type == "Lightweight Model":
            model, tokenizer, num_open_ended_classes = load_lightweight_model()
            st.info(f"🚀 **Lightweight Model** - MobileNetV3-Small + DistilBERT")
        else:
            model, tokenizer, num_open_ended_classes = load_baseline_model()
            st.info(f"🔬 **Baseline Model** - ResNet-34 + BERT-base")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    import traceback
    st.code(traceback.format_exc())
    st.stop()

st.markdown("---")

# Image Upload Section
st.subheader("📤 Upload Medical Image")

uploaded_file = st.file_uploader(
    "Choose a radiology image...",
    type=["png", "jpg", "jpeg", "dcm"],
    help="Upload X-ray, CT, MRI or other medical images",
    key="file_uploader"
)

# Display uploaded image
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        
        # Display original image
        st.markdown("**Original Image**")
        st.image(image, use_container_width=True)
        
        # Placeholder for Attention Visualization
        if show_attention:
            st.markdown("**Cross-Modal Attention Visualization**")
            attention_placeholder = st.empty()
            attention_placeholder.info("Attention visualization will appear here after answering a question")
        
        # Show image details
        st.info(f"📊 Image Size: {image.size[0]} x {image.size[1]} pixels")
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")

st.markdown("---")

# Question Section
st.subheader("❓ Ask a Question")

question = st.text_input(
    "Enter your question:",
    placeholder="e.g., What abnormalities are visible in this image?",
    help="Ask questions about the uploaded radiology image"
)

# Submit button
if st.button("🔍 Get Answer", type="primary", use_container_width=True):
    if uploaded_file is None:
        st.warning("⚠️ Please upload an image first.")
    elif not question:
        st.warning("⚠️ Please enter a question.")
    else:
        try:
            st.success("✅ Processing your question...")
            
            with st.spinner("Analyzing image..."):
                start_time = time.time()
                
                # Preprocess image
                img_tensor = preprocess_image(image)
                
                # Preprocess question
                question_encoding = preprocess_question(question, tokenizer)
                
                # Run inference with dual-head model
                predicted_answer, confidence_score, top_k_predictions, attention_weights, is_binary = predict(
                    model, img_tensor, question_encoding, question, idx_to_answer, binary_answers, top_k=5
                )
                
                end_time = time.time()
                inference_time = end_time - start_time
                
                # Display results in columns
                result_col1, result_col2 = st.columns([2, 1])
                
                with result_col1:
                    st.markdown("### 💬 Answer:")
                    st.markdown(f"**{predicted_answer}**")
                    
                    # Show which head was used
                    head_type = "Binary (Yes/No)" if is_binary else "Open-Ended"
                    st.caption(f"🎯 Question Type: {head_type}")
                    
                    # Show top-k predictions
                    with st.expander("🔍 See Top 5 Predictions"):
                        for i, (ans, conf) in enumerate(top_k_predictions, 1):
                            st.write(f"{i}. **{ans}** - {conf:.2%}")
                
                with result_col2:
                    st.markdown("### 📊 Metrics:")
                    st.metric("Confidence", f"{confidence_score:.2%}")
                    st.metric("Inference Time", f"{inference_time:.3f}s")
                    st.metric("Device", DEVICE.type.upper())
                
                st.markdown("**ℹ️ Note**: If the model displays 'Not confident', it means the answer falls outside its trained vocabulary.")

                # Display cross-modal attention visualization if enabled
                if show_attention and uploaded_file is not None and attention_weights is not None:
                    st.markdown("---")
                    st.markdown("### 🧠 Cross-Modal Attention Analysis")
                    
                    try:
                        
                        # Generate attention visualization
                        fig = visualize_cross_modal_attention(attention_weights, question)
                        
                        # Clear placeholder and display
                        if attention_placeholder:
                            attention_placeholder.empty()
                        
                        st.pyplot(fig)
                        plt.close(fig)  # Clean up
                        
                        # Add explanation
                        st.info("""
                        **💡 Understanding the Visualization:**
                        - **Left plot**: Shows how visual and text features attend to each other
                        - **Right plot**: Shows the overall importance of each modality for the answer
                        - Higher attention weight = more influence on the final prediction
                        """)
                        
                        # Debug: Show attention weights shape
                        with st.expander("🔍 Debug Info (Attention Shape)", expanded=False):
                            st.write(f"Attention weights shape: {attention_weights.shape}")
                            st.write(f"Attention weights type: {type(attention_weights)}")

                    except Exception as e:
                        st.error(f"⚠️ Could not visualize attention: {e}")
                        import traceback
                        with st.expander("🐛 Full Error Trace"):
                            st.code(traceback.format_exc())
        
        except Exception as e:
            st.error(f"❌ Error during inference: {e}")
            import traceback
            st.code(traceback.format_exc())

st.markdown("---")

# Sample Questions Section
st.header("📝 Sample Questions")

col_q1, col_q2 = st.columns(2)

with col_q1:
    st.markdown("""
    **Detection Questions:**
    - Is there any fracture visible?
    - Are there any abnormalities?
    - What pathological findings can you identify?
    """)

with col_q2:
    st.markdown("""
    **Description Questions:**
    - What type of imaging modality is this?
    - Describe the findings in this scan.
    - What is the patient's condition?
    """)

# Footer
st.markdown("---")
st.caption("⚕️ This tool is for educational purposes only and should not replace professional medical advice.")