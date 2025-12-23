# Streamlit Cloud Deployment Guide

This guide will help you deploy your Radiology VQA application to Streamlit Cloud.

## ✅ Pre-Deployment Checklist

### 1. Hugging Face Setup
- [x] Models uploaded to Hugging Face repository
- [x] Repository name: `radvqa-lightweight`
- [x] Model files:
  - `lightweight_best_model.pt`
  - `baseline_best_model.pt`
- [x] Repository is PUBLIC (required for free deployment)

### 2. Code Updates
- [x] Updated `requirements.txt` with `huggingface_hub`
- [x] Updated `radvqa_streamlit.py` to download from HuggingFace
- [x] Simplified answer vocabulary loading (no full dataset download)
- [x] Set device to CPU (Streamlit Cloud doesn't have GPU)
- [x] Created `.streamlit/config.toml`

## 📋 Required Files

Make sure these files are in your repository:

```
your-repo/
├── radvqa_streamlit.py          # Main Streamlit app
├── requirements.txt              # Python dependencies
├── .streamlit/
│   └── config.toml              # Streamlit configuration
├── models/
│   ├── lightweight_model.py     # Lightweight model definition
│   └── baseline_model.py        # Baseline model definition
├── preprocessing/
│   └── (preprocessing files)    # Keep for imports
└── README.md                    # Optional but recommended
```

## 🚀 Deployment Steps

### Step 1: Verify Your Hugging Face Repository

1. Go to your HuggingFace repo: `https://huggingface.co/YOUR_USERNAME/radvqa-lightweight`
2. Verify these files exist:
   - `lightweight_best_model.pt`
   - `baseline_best_model.pt`
3. Make sure the repository is **PUBLIC**
4. Note your HuggingFace username for the next step

### Step 2: Update Repository ID (if needed)

If your HuggingFace username is different from the repo name, update `radvqa_streamlit.py`:

```python
# Update this line with your full HuggingFace repo ID
HF_REPO_ID = "YOUR_USERNAME/radvqa-lightweight"
```

Current setting: `HF_REPO_ID = "radvqa-lightweight"`

### Step 3: Push to GitHub

1. Create a new GitHub repository (if not already done)

2. Push your code to GitHub:
```bash
git init
git add .
git commit -m "Initial commit for Streamlit deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

**Important:** Make sure `.gitignore` excludes local checkpoints (already configured)

### Step 4: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)

2. Sign in with your GitHub account

3. Click **"New app"**

4. Configure deployment:
   - **Repository:** Select your GitHub repository
   - **Branch:** `main`
   - **Main file path:** `radvqa_streamlit.py`
   - **App URL:** Choose your subdomain (e.g., `your-app.streamlit.app`)

5. Click **"Deploy!"**

### Step 5: Monitor Deployment

The deployment will take 5-10 minutes:
1. Streamlit will install dependencies from `requirements.txt`
2. The app will start and download models from HuggingFace (first time only)
3. Models are cached for subsequent runs

**Watch the logs** for any errors during deployment.

## 🔧 Configuration Details

### Device Configuration
```python
DEVICE = torch.device('cpu')  # Streamlit Cloud uses CPU only
```
- GPU is not available on free tier
- CPU inference is sufficient for demo purposes
- Typical inference time: 2-5 seconds per question

### Model Loading
Models are downloaded from HuggingFace on first run and cached:
```python
@st.cache_resource
def download_model_from_hf(model_file: str):
    model_path = hf_hub_download(
        repo_id="radvqa-lightweight",
        filename=model_file
    )
    return model_path
```

### Answer Vocabulary
Simplified vocabulary is used to avoid downloading full VQA-RAD dataset:
- Pre-defined list of 100+ common medical terms
- Includes binary answers (yes/no) for dual-head routing
- Faster startup time

## 🐛 Troubleshooting

### Error: "Model file not found"
**Solution:** 
1. Check your HuggingFace repo is public
2. Verify file names match exactly:
   - `lightweight_best_model.pt`
   - `baseline_best_model.pt`
3. Update `HF_REPO_ID` with your username if different

### Error: "Out of memory"
**Solution:**
- Streamlit Cloud free tier has 1GB RAM limit
- Models are ~20-130MB each
- Should work fine, but if issues arise:
  - Deploy only one model (comment out the other)
  - Or upgrade to Streamlit Cloud paid tier

### Error: "Module not found"
**Solution:**
- Ensure all imports are in `requirements.txt`
- Check that `models/` and `preprocessing/` folders are in your repo
- Verify folder structure matches expected layout

### Slow First Load
**Normal behavior:**
- First deployment takes 5-10 minutes
- Models download from HuggingFace (200MB total)
- Subsequent loads are much faster (cached models)

### Model Performance Issues
**Expected on CPU:**
- Inference time: 2-5 seconds per question
- This is normal for CPU inference
- GPU would be 10-100x faster but requires paid tier

## 📊 Resource Usage

| Resource | Usage | Limit (Free Tier) |
|----------|-------|-------------------|
| RAM | ~800MB | 1GB |
| Storage | ~300MB | 200MB* |
| CPU | Light | Shared |
| Bandwidth | ~5MB per session | Unlimited |

*Storage limit is per app, models are cached efficiently

## 🎯 Post-Deployment

### Testing Your Deployed App

1. Visit your app URL: `https://your-app.streamlit.app`
2. Wait for models to download (first time only)
3. Upload a medical image
4. Ask a question
5. Verify:
   - Model loads correctly
   - Inference works (2-5 seconds)
   - Attention visualization displays
   - Both models (lightweight/baseline) work

### Sharing Your App

Your app is now live! Share the URL:
- `https://your-app.streamlit.app`
- No authentication required (public)
- Can handle multiple concurrent users

### Monitoring

Streamlit Cloud dashboard shows:
- App status (running/stopped/error)
- Resource usage
- Logs (for debugging)
- Analytics (views, usage)

## 🔄 Updating Your App

To update the app after deployment:

1. Make changes locally
2. Test locally: `streamlit run radvqa_streamlit.py`
3. Push to GitHub:
```bash
git add .
git commit -m "Update: description of changes"
git push
```
4. Streamlit Cloud automatically redeploys on push!

### Updating Models

To update models on HuggingFace:
1. Upload new `*_best_model.pt` to HuggingFace
2. Clear cache in Streamlit Cloud (Settings → "Clear cache")
3. App will download new models on next run

## 📝 Environment Variables (Optional)

If your HuggingFace repo becomes private later:

1. In Streamlit Cloud dashboard, go to **Settings**
2. Add secrets in **Secrets** section:
```toml
HF_TOKEN = "your_huggingface_token"
```
3. Update code to use token:
```python
model_path = hf_hub_download(
    repo_id="radvqa-lightweight",
    filename=model_file,
    token=st.secrets.get("HF_TOKEN", None)
)
```

## ✅ Deployment Complete!

Your Radiology VQA app is now live! 🎉

**App URL:** `https://your-app.streamlit.app`

### Quick Links
- [Streamlit Cloud Dashboard](https://share.streamlit.io)
- [Your HuggingFace Repo](https://huggingface.co/YOUR_USERNAME/radvqa-lightweight)
- [Streamlit Documentation](https://docs.streamlit.io)

---

## 🆘 Need Help?

- **Streamlit Community:** [discuss.streamlit.io](https://discuss.streamlit.io)
- **Docs:** [docs.streamlit.io/streamlit-cloud](https://docs.streamlit.io/streamlit-cloud)
- **Status:** [status.streamlit.io](https://status.streamlit.io)

Good luck with your deployment! 🚀

