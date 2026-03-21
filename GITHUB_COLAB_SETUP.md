# 🚀 Quick Setup Guide: GitHub → Colab Pipeline

This guide walks you through uploading to GitHub and running on Google Colab.

## Step 1: Create GitHub Repository (5 minutes)

### 1.1 Sign up / Log in
- Go to https://github.com
- Create account or log in

### 1.2 Create New Repository
1. Click **"+"** (top right) → **"New repository"**
2. Name: `image-captioning`
3. Description: `Explainable Image Captioning with Vision Transformers`
4. Make it **Public** (easier to share) or **Private**
5. **Initialize with**: None (we'll push existing code)
6. Click **"Create repository"**

### 1.3 Push Local Code to GitHub

Open PowerShell in your project folder:

```powershell
# Navigate to project
cd C:\Users\91876\Desktop\SEM6\TDL\PROJECT

# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Explainable Image Captioning Model"

# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/image-captioning.git

# Rename branch to main (GitHub default)
git branch -M main

# Push to GitHub
git push -u origin main
```

**Done!** Your code is now on GitHub at:  
`https://github.com/YOUR_USERNAME/image-captioning`

---

## Step 2: Run on Google Colab (5 minutes)

### 2.1 Open New Colab Notebook
- Go to https://colab.research.google.com
- Click **"New notebook"**

### 2.2 Copy-Paste This Code

**Cell 1: Clone Repository**
```python
!git clone https://github.com/YOUR_USERNAME/image-captioning.git
%cd image-captioning
```

**Cell 2: Install Dependencies**
```python
!pip install -q -r requirements.txt
```

**Cell 3: Check GPU**
```python
import torch
print(f"✓ GPU Available: {torch.cuda.is_available()}")
print(f"✓ GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

**Cell 4: Enable GPU (if not already enabled)**
- Go to **Runtime** → **Change runtime type** → **GPU** → **Save**

**Cell 5: Create GPU Config**
```python
import yaml

# Read original config
config = yaml.safe_load(open('configs/config.yaml'))

# Optimize for GPU
config['device'] = 'cuda'
config['batch_size'] = 128  # Larger on GPU
config['learning_rate'] = 0.0001

# Save GPU config
with open('configs/gpu_config.yaml', 'w') as f:
    yaml.dump(config, f)

print("✓ GPU configuration created")
```

**Cell 6: Run Training (Debug Mode First)**
```python
!python main.py --config configs/gpu_config.yaml --debug
```

This runs 2 epochs in ~2 minutes on GPU to test everything.

**Cell 7: View TensorBoard**
```python
%load_ext tensorboard
%tensorboard --logdir outputs/logs/
```

A TensorBoard dashboard will appear showing loss curves!

**Cell 8: Run Full Training**
```python
# Run full 20 epochs
!python main.py --config configs/gpu_config.yaml
```

Expected time: 10-30 minutes on free Colab GPU

**Cell 9: Download Results**
```python
from google.colab import files
import shutil

# Zip checkpoints
shutil.make_archive('checkpoints', 'zip', 'checkpoints')

# Zip logs
shutil.make_archive('logs', 'zip', 'outputs/logs')

# Download to your computer
files.download('checkpoints.zip')
files.download('logs.zip')
```

**Cell 10: Push Results Back to GitHub**
```python
# Configure git
!git config user.email "your@email.com"
!git config user.name "Your Name"

# Add new files
!git add -A

# Commit
!git commit -m "Training complete: 20 epochs on Colab GPU"

# Push back
!git push origin main
```

---

## Step 3: Optional - Set Up GitHub Actions (10 minutes)

GitHub Actions will automatically test your code on every push:

The `.github/workflows/tests.yml` file is already configured. It will:
- ✅ Test Python 3.8, 3.9, 3.10
- ✅ Lint code for style issues
- ✅ Check all imports work
- ✅ Run debug mode test

Just push and watch it run! 🚀

---

## 📊 What You Get

### From GitHub:
- 🔐 Cloud backup of all code
- 📝 Version history (can revert changes)
- 🔗 Shareable link for professors/team
- 📊 Visual stats (commits, contributors)
- 🤖 Automatic testing on every push

### From Colab:
- 💻 **Free GPU access** (NVIDIA T4 or P100)
- ⚡ 10-50× faster training than CPU
- 📱 Run from any device (just need browser)
- 💾 1TB free storage
- ⏲️ 12-hour continuous sessions

---

## 🔄 Typical Workflow

```
1. Make changes locally
   ↓
2. Git commit & push to GitHub
   ↓
3. Clone in Colab
   ↓
4. Train on GPU
   ↓
5. Download results
   ↓
6. Analyze results locally
   ↓
7. Back to step 1
```

---

## ❓ FAQ

### Q: Do I need to pay for Colab?
**A**: No! Free tier gives you enough. Colab Pro ($10/month) for longer sessions.

### Q: How long can I train?
**A**: Up to 12 hours per session on free tier. GPU might disconnect if idle.

### Q: Can I save intermediate checkpoints?
**A**: Yes! Training auto-saves checkpoints every epoch to `checkpoints/`.  
Zip and download mid-training if needed.

### Q: What if session disconnects?
**A**: Colab will keep training if not actively used. Download results using `file.download()` before session ends.

### Q: Can I use my local GPU instead?
**A**: Yes! Change `device: "cuda"` in config.yaml and run locally (need NVIDIA GPU + CUDA).

### Q: How do I update code that's already in Colab?
**A**: 
```python
# In Colab
!cd image-captioning && git pull origin main
```

---

## 🎓 Example Training Session

### Time: ~30 minutes

```
0:00  → Start Colab
0:05  → Clone repo & install packages
0:10  → Enable GPU & create config
0:12  → Run debug mode (2 epochs) - PASS ✓
0:15  → Start full training (20 epochs)
0:40  → Training complete!
0:42  → Download checkpoints.zip
0:45  → Push results to GitHub
```

---

## 📚 Resources

- [GitHub Docs](https://docs.github.com/en)
- [Google Colab Docs](https://colab.research.google.com/)
- [Git Tutorial](https://git-scm.com/book/en/v2)

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| "fatal: repository not found" | Check GitHub username in git URL |
| "permission denied" | Generate GitHub token: Settings → Developer settings → Personal access tokens |
| GPU not available | Runtime → Change runtime type → GPU |
| Colab disconnected | Save checkpoints frequently, try Colab Pro |
| Out of memory | Reduce batch_size in config |

---

## ✅ Checklist

- [ ] GitHub account created
- [ ] Repository created and pushed
- [ ] Colab notebook created
- [ ] GPU enabled in Colab
- [ ] Dependencies installed
- [ ] Debug mode runs successfully
- [ ] Full training completes
- [ ] Results downloaded
- [ ] Results pushed back to GitHub

---

**You're all set!** 🎉  
Start training your model on free GPU now!

Questions? Check [README.md](README.md) or [TRAINING_SUMMARY.md](TRAINING_SUMMARY.md)
