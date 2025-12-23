# 🚀 Quick Deployment Guide

## Step-by-Step Deployment Process

### ✅ Step 1: Save Your Answer Vocabulary

Run this script to extract the answer vocabulary from your training data:

```bash
python save_answer_vocab.py
```

**Output:** This creates `answer_vocab.json` with your exact training vocabulary.

**What it does:**
- Loads the VQA-RAD dataset
- Extracts the answer vocabulary used during training
- Saves it to `answer_vocab.json`
- Shows you the top 10 answers

---

### ✅ Step 2: Upload to HuggingFace

Go to your HuggingFace repository:
👉 [https://huggingface.co/daphne04/radvqa-lightweight](https://huggingface.co/daphne04/radvqa-lightweight)

Upload these 3 files:
1. ✅ `lightweight_best_model.pt` (already uploaded)
2. ✅ `baseline_best_model.pt` (already uploaded)
3. **📤 `answer_vocab.json`** ← Upload this now!

**How to upload:**
1. Click "Files and versions"
2. Click "Add file" → "Upload files"
3. Select `answer_vocab.json`
4. Click "Commit changes to main"

---

### ✅ Step 3: Verify Your Files

Your HuggingFace repo should now contain:

```
daphne04/radvqa-lightweight/
├── lightweight_best_model.pt  (~20MB)
├── baseline_best_model.pt      (~130MB)
└── answer_vocab.json           (~5KB)
```

---

### ✅ Step 4: Test Locally (Optional)

Before deploying, test locally:

```bash
streamlit run radvqa_streamlit.py
```

**What to check:**
- ✅ Models download from HuggingFace
- ✅ Answer vocabulary loads successfully
- ✅ Inference works correctly
- ✅ Shows "Loaded vocabulary with X answers from training"

---

### ✅ Step 5: Push to GitHub

```bash
# Initialize git (if not already done)
git init

# Add files
git add .

# Commit
git commit -m "Add Streamlit deployment with HuggingFace integration"

# Add your GitHub repo as remote
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# Push to GitHub
git branch -M main
git push -u origin main
```

**Important:** Make sure these files are included:
- ✅ `radvqa_streamlit.py`
- ✅ `requirements.txt`
- ✅ `.streamlit/config.toml`
- ✅ `models/` folder (model definitions)
- ✅ `preprocessing/` folder (imports needed)

**Don't upload:**
- ❌ Local checkpoints (`.pt` files) - These are on HuggingFace
- ❌ `__pycache__` folders - Git ignores these
- ❌ Large datasets - Not needed for deployment

---

### ✅ Step 6: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)

2. Click **"New app"**

3. Configure:
   - **Repository:** Your GitHub repo
   - **Branch:** `main`
   - **Main file:** `radvqa_streamlit.py`
   - **App URL:** Choose a name (e.g., `radiology-vqa`)

4. Click **"Deploy!"**

5. **Wait 5-10 minutes** for:
   - Dependencies to install
   - Models to download from HuggingFace
   - App to start

---

### ✅ Step 7: Verify Deployment

Once deployed, check:

1. **Answer Vocabulary:**
   - Should see: "✅ Loaded vocabulary with 121 answers from training"
   - If you see a warning about fallback vocabulary, `answer_vocab.json` is missing from HuggingFace

2. **Models:**
   - Both Lightweight and Baseline models should load
   - First load takes ~1-2 minutes (downloading from HuggingFace)
   - Subsequent loads are instant (cached)

3. **Inference:**
   - Upload a medical image
   - Ask a question
   - Should get answer in 2-5 seconds (CPU inference)
   - Attention visualization should display

---

## 🎯 Your Deployment URLs

**HuggingFace Repo:**
- https://huggingface.co/daphne04/radvqa-lightweight

**Streamlit App:** (after deployment)
- https://YOUR-APP-NAME.streamlit.app

---

## 📊 What Changed from Hardcoded Vocabulary?

### Before (Hardcoded):
```python
# Hardcoded list of ~40 common medical terms
common_answers = ['<unk>', 'yes', 'no', 'normal', ...]
```
**Problems:**
- ❌ Might not match training vocabulary exactly
- ❌ Missing some answers from training
- ❌ Indices might not align

### After (From Training):
```python
# Downloads actual vocabulary from HuggingFace
vocab_path = download_from_hf("answer_vocab.json")
answer_vocab = json.load(vocab_path)
```
**Benefits:**
- ✅ **Exact same vocabulary as training**
- ✅ **Correct indices for all answers**
- ✅ **All 121 answers available**
- ✅ **Better prediction accuracy**

---

## 🐛 Troubleshooting

### Issue: "Could not load answer_vocab.json"

**Symptoms:**
- Warning message in app
- Falls back to minimal vocabulary
- May have lower accuracy

**Solution:**
1. Check `answer_vocab.json` exists in your HuggingFace repo
2. File name must be exactly: `answer_vocab.json` (lowercase, with extension)
3. Repo must be public
4. Clear Streamlit cache and reload

---

### Issue: "Models not loading"

**Solution:**
1. Verify repo ID in code: `HF_REPO_ID = "daphne04/radvqa-lightweight"`
2. Check files exist in HuggingFace
3. Ensure repo is public
4. Check Streamlit logs for errors

---

### Issue: "Out of memory"

**Solution:**
- Streamlit Cloud free tier: 1GB RAM
- Your models: ~150MB total
- Should work fine, but if issues:
  - Comment out one model (deploy only lightweight or baseline)
  - Or upgrade to paid tier

---

## 📝 Files Checklist

Before deployment, ensure these files exist:

### In Your Local Repository:
- [x] `radvqa_streamlit.py` - Updated with HuggingFace download
- [x] `requirements.txt` - Includes `huggingface_hub`
- [x] `.streamlit/config.toml` - Streamlit configuration
- [x] `models/lightweight_model.py` - Model definition
- [x] `models/baseline_model.py` - Model definition
- [x] `preprocessing/` - Preprocessing modules
- [x] `save_answer_vocab.py` - Script to extract vocabulary
- [x] `.gitignore` - Excludes checkpoints and cache

### On HuggingFace (daphne04/radvqa-lightweight):
- [x] `lightweight_best_model.pt`
- [x] `baseline_best_model.pt`
- [x] `answer_vocab.json` ← **Make sure you upload this!**

### On GitHub:
- [x] All local repository files (except ignored ones)
- [x] No large `.pt` files (they're on HuggingFace)

---

## 🎉 Deployment Complete!

Once all steps are done:
1. Your app loads the **exact training vocabulary** from HuggingFace
2. Models download automatically on first run
3. Everything is cached for fast subsequent loads
4. Users can access your app at: `https://your-app.streamlit.app`

---

## 🔄 Updating After Deployment

### Update Answer Vocabulary:
1. Re-run `python save_answer_vocab.py`
2. Upload new `answer_vocab.json` to HuggingFace
3. In Streamlit Cloud: Settings → "Clear cache"
4. App will download new vocabulary on next load

### Update Models:
1. Upload new `.pt` files to HuggingFace
2. In Streamlit Cloud: Settings → "Clear cache"
3. App will download new models on next load

### Update Code:
1. Make changes locally
2. `git push` to GitHub
3. Streamlit Cloud auto-redeploys! ✨

---

**Need help?** Check `STREAMLIT_DEPLOYMENT.md` for detailed troubleshooting!

Good luck with your deployment! 🚀

