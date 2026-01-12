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
import os
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
import pandas as pd

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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Device configuration
DEVICE = torch.device('cpu')

# Hugging Face configuration
HF_REPO_ID = "daphne04/radvqa-lightweight"
LIGHTWEIGHT_MODEL_FILE = "lightweight_best_model.pt"
BASELINE_MODEL_FILE = "baseline_best_model.pt"
ANSWER_VOCAB_FILE = "answer_vocab.json"

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("🏥 Radiology VQA")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", ["📊 Project Overview", "🎯 Demo"], label_visibility="collapsed")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def download_from_hf(filename: str, repo_id: str = HF_REPO_ID):
    try:
        with st.spinner(f"📥 Downloading {filename} from Hugging Face..."):
            file_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=None)
        return file_path
    except Exception as e:
        st.error(f"❌ Error downloading {filename} from Hugging Face: {e}")
        st.info(f"Please ensure the file exists in repository '{repo_id}'")
        raise

@st.cache_resource
def load_answer_vocab():
    BINARY_ANSWERS = {'yes', 'no'}
    try:
        vocab_path = download_from_hf(ANSWER_VOCAB_FILE)
        with open(vocab_path, 'r') as f:
            answer_vocab = json.load(f)
        idx_to_answer = {idx: ans for ans, idx in answer_vocab.items()}
        st.success(f"✅ Loaded vocabulary with {len(answer_vocab)} answers from training")
        return answer_vocab, idx_to_answer, BINARY_ANSWERS
    except Exception as e:
        st.warning(f"⚠️ Could not load answer_vocab.json from HuggingFace. Using fallback vocabulary.")
        st.info("To use the exact training vocabulary, upload 'answer_vocab.json' to your HuggingFace repo.")
        fallback_answers = [
            '<unk>', 'yes', 'no', 'normal', 'abnormal', 'lung', 'chest', 'left', 'right',
            'pneumonia', 'ct scan', 'mri', 'x-ray', 'brain', 'abdomen', 'heart', 'kidney',
            'liver', 'spine', 'bone', 'fracture', 'tumor', 'cancer', 'mass', 'lesion',
            'effusion', 'edema', 'enlarged', 'calcification', 'atelectasis', 'consolidation',
            'nodule', 'opacity', 'pleural', 'cardiomegaly', 'male', 'female', 'both'
        ]
        answer_vocab = {answer: idx for idx, answer in enumerate(fallback_answers)}
        idx_to_answer = {idx: answer for answer, idx in answer_vocab.items()}
        return answer_vocab, idx_to_answer, BINARY_ANSWERS

@st.cache_resource
def load_lightweight_model():
    checkpoint_path = download_from_hf(LIGHTWEIGHT_MODEL_FILE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    config = checkpoint.get('config', {})
    model = DualHeadVQAModel(
        num_open_ended_classes=config.get('num_open_ended_classes', 121),
        fusion_hidden_dim=config.get('fusion_hidden_dim', 256),
        num_attention_heads=config.get('num_attention_heads', 4),
        dropout=config.get('dropout', 0.3),
        freeze_vision_encoder=False,
        freeze_text_encoder=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    return model, tokenizer, config.get('num_open_ended_classes', 121)

@st.cache_resource
def load_baseline_model():
    checkpoint_path = download_from_hf(BASELINE_MODEL_FILE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    config = checkpoint.get('config', {})
    model = DualHeadBaselineVQAModel(
        num_open_ended_classes=config.get('num_open_ended_classes', 121),
        fusion_hidden_dim=config.get('fusion_hidden_dim', 512),
        num_attention_heads=config.get('num_attention_heads', 8),
        dropout=config.get('dropout', 0.35),
        freeze_vision_encoder=False,
        freeze_text_encoder=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer, config.get('num_open_ended_classes', 121)

def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return transform(image).unsqueeze(0)

def preprocess_question(question: str, tokenizer, max_length: int = 64) -> dict:
    return tokenizer(question, padding='max_length', truncation=True, 
                    max_length=max_length, return_tensors='pt')

def is_binary_question(question: str) -> bool:
    question_lower = question.lower().strip()
    binary_indicators = [
        'is there', 'are there', 'is this', 'are these',
        'does this', 'do these', 'can you see',
        'is it', 'are they', 'was there', 'were there',
        'has the', 'have the', 'did the'
    ]
    
    for indicator in binary_indicators:
        if question_lower.startswith(indicator):
            return True
    
    open_ended_words = ['what', 'which', 'where', 'when', 'who', 'how', 'why', 'describe']
    for word in open_ended_words:
        if question_lower.startswith(word):
            return False
    
    if question_lower.endswith('?') and len(question.split()) < 10:
        return True
    
    return False

def predict(model, image_tensor, question_encoding, question_text, idx_to_answer, binary_answers, top_k=5):
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        input_ids = question_encoding['input_ids'].to(DEVICE)
        attention_mask = question_encoding['attention_mask'].to(DEVICE)
        
        is_binary = is_binary_question(question_text)
        
        try:
            outputs = model(image_tensor, input_ids, attention_mask, return_attention=True)
        except TypeError:
            outputs = model(image_tensor, input_ids, attention_mask, return_features=True)
        
        if is_binary:
            logits = outputs['binary']
            current_idx_to_answer = {0: 'no', 1: 'yes'}
        else:
            logits = outputs['open_ended']
            current_idx_to_answer = idx_to_answer
        
        attention_weights = outputs.get('attention', outputs.get('attention_weights'))
        
        probs = F.softmax(logits, dim=-1)
        
        top_prob, top_idx = torch.max(probs, dim=1)
        predicted_idx = top_idx.item()
        predicted_answer = current_idx_to_answer.get(predicted_idx, '<unk>')
        confidence = top_prob.item()
        
        if predicted_answer == '<unk>' or predicted_answer.startswith('<unknown'):
            display_answer = "Not confident"
        else:
            display_answer = predicted_answer
        
        top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, probs.size(1)), dim=1)
        top_k_predictions = []
        for idx, prob in zip(top_k_indices[0], top_k_probs[0]):
            ans = current_idx_to_answer.get(idx.item(), '<unk>')
            if ans == '<unk>' or ans.startswith('<unknown'):
                ans = "Not confident"
            top_k_predictions.append((ans, prob.item()))
        
    return display_answer, confidence, top_k_predictions, attention_weights, is_binary

def visualize_cross_modal_attention(attention_weights, question_text):
    attn = attention_weights[0].cpu().numpy()
    
    if attn.ndim == 3:
        attn_avg = attn.mean(axis=0)
    elif attn.ndim == 2:
        attn_avg = attn
    else:
        raise ValueError(f"Unexpected attention weights shape: {attn.shape}")
    
    if attn_avg.shape != (2, 2):
        raise ValueError(f"Invalid attention matrix shape: {attn_avg.shape}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Attention Matrix Heatmap
    im = ax1.imshow(attn_avg, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Visual', 'Text'], fontsize=11)
    ax1.set_yticklabels(['Visual', 'Text'], fontsize=11)
    ax1.set_xlabel('Key', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Query', fontsize=12, fontweight='bold')
    ax1.set_title('Cross-Modal Attention Matrix\n(Averaged across heads)', fontsize=12, fontweight='bold')
    
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, f'{attn_avg[i, j]:.3f}',
                    ha="center", va="center", color="black", fontsize=11, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', fontsize=10)
    
    # Modality Importance Bar Chart
    visual_importance = attn_avg[:, 0].mean()
    text_importance = attn_avg[:, 1].mean()
    
    modalities = ['Visual\nFeatures', 'Text\nFeatures']
    importances = [visual_importance, text_importance]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = ax2.bar(modalities, importances, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Average Attention Weight', fontsize=12, fontweight='bold')
    ax2.set_title('Modality Importance', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars, importances):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1%}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    dominant = "Visual" if visual_importance > text_importance else "Text"
    ratio = max(visual_importance, text_importance) / min(visual_importance, text_importance)
    
    interpretation = f"The model relies more on {dominant.lower()} information ({ratio:.1f}x)"
    fig.text(0.5, 0.02, interpretation, ha='center', fontsize=10, 
             style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    return fig

# ============================================================================
# PAGE 1: DEMO PAGE (ORIGINAL LAYOUT PRESERVED)
# ============================================================================

if page == "🎯 Demo":

    st.sidebar.markdown("### 📝 Sample Questions")

    sample_questions = [
        "Is there any abnormality?",
        "Is there a fracture?",
        "Is this image normal?",
        "Is this a CT image?",
        "Is there free fluid?"
    ]

    for q in sample_questions:
        cols = st.sidebar.columns([1])   # full width
        if cols[0].button(q, key=q, use_container_width=True):
            st.session_state.user_question = q
    
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
    except Exception as e:
        st.error(f"❌ Error loading answer vocabulary: {e}")
        st.stop()

    # Model Selection
    st.subheader("⚙️ Model Selection")

    col1, col2 = st.columns([3,1])

    with col2:
        show_attention = st.checkbox(
            "   Show Attention Visualization",
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

            # Center image using columns
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                st.image(image, width=500)

            # Placeholder for Attention Visualization
            if show_attention:
                attention_placeholder = st.empty()
               
            st.info(f"📊 Image Size: {image.size[0]} x {image.size[1]} pixels")

        except Exception as e:
            st.error(f"Error loading image: {str(e)}")

    st.markdown("---")

    # Question Section
    st.subheader("❓ Ask a Question")

    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What abnormalities are visible in this image?",
        help="Ask questions about the uploaded radiology image",
        key="user_question"
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
                        with st.expander("🔍 See Top Predictions"):
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
                            plt.close(fig)
                            
                            # Add explanation
                            st.info("""
                            **💡 Understanding the Visualization:**
                            - **Left plot**: Shows how visual and text features attend to each other
                            - **Right plot**: Shows the overall importance of each modality for the answer
                            - Higher attention weight = more influence on the final prediction
                            """)

                        except Exception as e:
                            st.error(f"⚠️ Could not visualize attention: {e}")
                            import traceback
                            with st.expander("🛠 Full Error Trace"):
                                st.code(traceback.format_exc())
            
            except Exception as e:
                st.error(f"❌ Error during inference: {e}")
                import traceback
                st.code(traceback.format_exc())

    st.markdown("---")
    st.caption("⚕️ This tool is for educational purposes only and should not replace professional medical advice.")

# ============================================================================
# PAGE 2: PROJECT OVERVIEW (KEEP YOUR ORIGINAL CODE)
# ============================================================================

elif page == "📊 Project Overview":
    st.title("📊 Lightweight VQA in Radiology Domain")
    st.markdown("### WIH3001 Data Science Project")
    
    st.info("""
    👨‍🎓 Prepared by: Kueh Pang Lang (23005227)  
    👨‍🏫 Supervised by: Dr Hoo Wai Lam  
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📖 Background", "📊 Dataset", "📈 EDA", "🎯 Performance"])
    
    # TAB 1: BACKGROUND
    with tab1:
        st.subheader("❓ What is Radiology VQA?")
        st.markdown("""
        - 🖼️ Combines **Computer Vision** and **Natural Language Processing**
        - 💬 Answers questions about medical images
        - 🧠 Uses **multimodal fusion**
        - 🏥 Supports clinical decision making
        """)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Objectives")
            st.markdown("""
            1. To build a deep learning model that can answer natural language questions about radiological images.
            2. To evaluate the performance of different VQA architectures.
            3. To develop a web-based application for interactive Radiology VQA.
            """)

        with col2:
            st.subheader("⚠️ Problem Statement")
            st.markdown("""
            - LLMs lack visual understanding
            - Existing VQA models are large (300–500M params)
            - High inference time and computationally heavy
            """)

        st.markdown("---")
        st.subheader("�️ Architecture Comparison")

        comparison_data = {
            'Component': ['Vision Encoder', 'Text Encoder', 'Total Parameter Counts', 'Model Size', 'Inference Time'],
            'Baseline': ['ResNet-34', 'BERT-base', '134M', '512MB', '16.4ms'],
            'Lightweight': ['MobileNetV3', 'DistilBERT', '70M', '270MB', '15.8ms'],
            'Reduction': ['8.5×', '1.67×', '1.9×', '1.9×', '1.04×']
        }

        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("🔑 Methodology Flow")
        st.image('analysis/eda_reports/methodology.png')
        
    # TAB 2: DATASET
    with tab2:
        st.header("Dataset - VQA-RAD")
        
        st.markdown("""**Lau et al. (2018)**  [VQA-RAD Dataset — Scientific Data](https://www.nature.com/articles/sdata2018251)""")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Q-A Pairs", "2,248")
        col2.metric("Images", "315")
        col3.metric("Modalities", "3")
        col4.metric("Vocabulary", "120")
        
        st.image('analysis/eda_reports/question type.png')
        st.markdown("<p style='text-align: center;'> Question type distribution for open-ended questions", unsafe_allow_html=True)

        st.subheader("\n")

        st.image('analysis/eda_reports/Workflow.webp')
        st.markdown("<p style='text-align: center;'> Workflow of the dataset creation", unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📋 Dataset Splits")
        
        split_data = {
            'Split': ['Train', 'Validation', 'Test'],
            'Samples': [1434, 359, 451],
            'Percentage': ['64%', '16%', '20%']
        }
        split_data['Samples'] = list(map(str, split_data['Samples']))
        df = pd.DataFrame(split_data)

        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.info("✅ Stratified split: 60/40 for binary and open-ended questions maintained across splits")

        st.markdown("---")
        st.subheader("🔤 Vocabulary Strategy")
        
        vocab_col1, vocab_col2 = st.columns(2)
        
        with vocab_col1:
            st.markdown("""
            **Challenge:**
            - 450+ unique answers
            - Long-tail distribution
            - Risk of overfitting
            """)
        
        with vocab_col2:
            st.markdown("""
            **Solution:**
            - Top 120 answers (95% coverage)
            - Rare terms mapped to `<unk>` token
            - Better generalization
            """)
        
        st.markdown("**📊 Answer Coverage Curve**")
        st.image('analysis/eda_reports/answer_coverage_curve.png')
        st.markdown("<p style='text-align: center;'> Coverage threshold: 0.95</p>", unsafe_allow_html=True)
    
    # TAB 3: EDA
    with tab3:
        st.subheader("🖼 Image Size Distribution")
        st.image('analysis/eda_reports/image_size_distribution.png')
        
        st.markdown("---")
        st.subheader("📊 Question Type Distribution")
        st.markdown("""
        - **Binary (60%):** Yes/no answers
        - **Open-Ended (40%):** Medical terms
        """)
        st.image('analysis/eda_reports/question_type_distribution.png')
        
        st.markdown("---")
        st.subheader("📏 Question Length")
        st.image('analysis/eda_reports/question_length_hist.png')
        
        st.markdown("---")
        st.subheader("🔤 Top Answers")
        top_answers = {
            'Rank': list(range(1, 11)),
            'Answer': ['no', 'yes', 'axial', 'right', 'left', 'pa', 'ct', 'brain', 'fat','mri'],
            'Count': [606, 585, 43, 26, 19, 15, 13, 13, 9, 8],
            'Percentage': ['27.0%', '26.1%', '1.9%', '1.2%', '0.8%', '0.7%', '0.6%', '0.6%', '0.4%','0.4%']
        }
        st.dataframe(pd.DataFrame(top_answers), use_container_width=True, hide_index=True)
    
    # TAB 4: PERFORMANCE
    with tab4:
        st.header("Model Performance")
        
        st.subheader("📊 Overall Accuracy")
        col1, col2, col3 = st.columns(3)
        col1.metric("Baseline", "66.3%")
        col2.metric("Lightweight", "69.4%", delta="+3.1%")
        col3.metric("Improvement", "4.7%")
        
        st.markdown("---")
        st.subheader("🎯 Binary Questions")
        binary_metrics = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
            'Baseline': ['62.55%', '59.23%', '65.25%', '62.10%', '67.12%'],
            'Lightweight': ['65.74%', '60.39%', '78.81%', '68.38%', '64.69%'],
            'Change': ['+3.19%', '+1.16%', '+13.56%', '+6.28%', '-2.43%']
        }
        st.dataframe(pd.DataFrame(binary_metrics), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.subheader("📝 Multi-Class Questions")
        multiclass_metrics = {
            'Metric': ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1', 'Top-3', 'Top-5','Top 10'],
            'Baseline': ['71.00%', '11.49%', '10.37%', '9.58%', '89.80%', '90.47%','90.69%'],
            'Lightweight': ['74.00%', '26.86%', '20.24%', '21.26%', '93.35%', '94.90%','96.01%'],
            'Change': ['+3.0%', '+15.4%', '+9.8%', '+11.7%', '+3.6%', '+4.4%','+5.3%']
        }
        st.dataframe(pd.DataFrame(multiclass_metrics), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.subheader("⚡ Efficiency")
        
        efficiency_data = {
            'Metric': ['Size', 'Parameters', 'Time', 'Speedup'],
            'Baseline': ['512 MB', '134M', '16.4 ms', '-'],
            'Lightweight': ['270 MB', '70M', '15.8 ms', '1.04×'],
            'Improvement': ['1.9× smaller', '1.9× fewer', '3.7% faster','-']
        }
        st.dataframe(pd.DataFrame(efficiency_data), use_container_width=True, hide_index=True)
        
        st.success("""
        ✅ **Key Finding:** Lightweight achieves 93% of baseline accuracy 
        with 1.9× parameter reduction and faster inference.
        """)
    
    st.markdown("---")
    st.caption("⚕️ Educational purposes only. Not for medical diagnosis.")