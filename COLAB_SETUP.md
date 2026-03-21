# Google Colab Setup Guide

## Cell 1: Clone Repository and Fix Dependencies

```python
# Clone the repo
!git clone https://github.com/SwathiHRao28/tdlProject.git /content/tdlProject

# Navigate to project
%cd /content/tdlProject

# Fix TensorBoard/TensorFlow compatibility
!pip install --upgrade tensorboard tensorflow torch torchvision --quiet

# Install other dependencies
!pip install pyyaml pillow tensorboard torch torchvision --quiet
```

## Cell 2: Mount Google Drive (Optional - for data)

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Cell 3: Download COCO Data (if not already in Drive)

```python
import os

# Create data directory
os.makedirs('/content/tdlProject/data/coco/images', exist_ok=True)
os.makedirs('/content/tdlProject/data/coco/captions', exist_ok=True)

# If images/captions are in Drive, link them:
# !ln -s /content/drive/MyDrive/coco_data/* /content/tdlProject/data/coco/

# Or download from COCO:
# !wget http://images.cocodataset.org/zips/val2014.zip -O /tmp/val2014.zip
# !unzip -q /tmp/val2014.zip -d /content/tdlProject/data/coco/images/

# Copy annotations
# !cp captions_train2014.json captions_val2014.json /content/tdlProject/data/coco/captions/
```

## Cell 4: Verify Data Structure

```python
import os

data_dir = '/content/tdlProject/data/coco'
print("Checking data structure:")
print(f"Images exist: {os.path.exists(os.path.join(data_dir, 'images'))}")
print(f"Captions exist: {os.path.exists(os.path.join(data_dir, 'captions'))}")

# List caption files
caption_files = os.listdir(os.path.join(data_dir, 'captions'))
print(f"Caption files: {caption_files}")
```

## Cell 5: Run Training

```python
%cd /content/tdlProject

# Run with GPU
!python main.py --config configs/config.yaml
```

## Cell 6: View TensorBoard Logs

```python
%load_ext tensorboard
%tensorboard --logdir=/content/tdlProject/outputs/logs
```

---

## Troubleshooting

**If you get ImportError about tensorboard:**
```python
!pip uninstall tensorboard -y
!pip install tensorboard==2.11.0 --quiet
```

**If COCO data is not loading:**
Check the file names in notebook:
```python
import os
print(os.listdir('/content/tdlProject/data/coco/captions'))
```

Should show:
- `captions_train2014.json`
- `captions_val2014.json`

**If getting GPU memory errors:**
Edit `configs/config.yaml`:
```yaml
batch_size: 16  # Reduce from 32
```
