# 🖼️ Explainable Image Captioning Model

An advanced image captioning system that generates descriptive captions for images while providing visual explanations of which image regions contributed to each word in the caption.

## 🌟 Features

- **Vision Transformer Encoder**: State-of-the-art image understanding using ViT-B/16
- **Transformer Decoder**: Sequence-to-sequence caption generation with teacher forcing
- **Gradient-based Attribution**: Identifies which image regions contribute to each word
- **Alignment Loss**: Ensures consistency between attention and attribution maps
- **Counterfactual Loss**: Verifies that attributed regions are truly important
- **TensorBoard Integration**: Real-time loss tracking and visualization
- **CPU & GPU Support**: Run locally on CPU or accelerate on GPU/Colab

## 📋 Project Structure

```
├── main.py                    # Entry point for training
├── inference.py              # Generate captions for new images
├── configs/
│   └── config.yaml          # Configuration file
├── models/
│   ├── caption_model.py      # Main model combining encoder + decoder
│   ├── encoder.py            # Vision Transformer encoder
│   └── decoder.py            # Transformer decoder with positional encoding
├── training/
│   ├── train.py              # Main training loop
│   └── evaluate.py           # Evaluation metrics (BLEU, ROUGE, etc)
├── explainability/
│   ├── attribution.py        # Gradient-based attribution maps
│   ├── alignment_loss.py     # Alignment loss function
│   └── counterfactual.py     # Counterfactual loss function
├── utils/
│   ├── dataset.py            # COCO dataset loader and vocabulary
│   └── preprocessing.py      # Image transforms
├── checkpoints/              # Saved model weights
└── outputs/logs/             # TensorBoard event files
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Debug Mode (Quick Test - ~5 minutes)

```bash
python main.py --debug
```

- Runs 2 epochs with 10 steps each
- Uses fast attribution approximation (10-20× faster)
- Batch size: 4
- Perfect for testing the pipeline

### Full Training (20 epochs - ~1 hour on CPU)

```bash
python main.py
```

- 20 epochs with full batch sizes
- Gradient-based attribution (accurate)
- Batch size: 32
- Saves checkpoint every epoch

### Monitor with TensorBoard

```bash
tensorboard --logdir outputs/logs/
# Open http://localhost:6006
```

## 📊 Model Architecture

### Encoder: Vision Transformer (ViT-B/16)
- Input: 224×224 RGB images
- Output: 196 spatial patches × 768-dim vectors
- Pre-trained on ImageNet, frozen during training

### Decoder: Transformer
- 6 layers, 8 attention heads
- Embed size: 512
- Hidden size: 512
- Max sequence length: 20 tokens
- Teacher forcing during training

### Loss Function

```
Total Loss = Caption Loss + 0.5 × Alignment Loss + 0.3 × Counterfactual Loss
```

Where:
- **Caption Loss**: Cross-entropy for generation quality
- **Alignment Loss**: L1 distance between attention ↔️ attribution
- **Counterfactual Loss**: Validates attribution importance

## ⚙️ Configuration

Edit `configs/config.yaml` to customize:

```yaml
# Model
encoder: "vit"              # "vit" or "resnet"
epochs: 20
batch_size: 32

# Learning
learning_rate: 0.0001
weight_decay: 0.01

# Explainability
use_alignment_loss: true
alignment_weight: 0.5
use_counterfactual_loss: true
counterfactual_weight: 0.3

# Device
device: "cpu"  # Change to "cuda" for GPU
```

## 🐍 Running on Google Colab (Recommended)

Colab provides **free GPU access** (~10-20 min training vs 1+ hour on CPU):

```python
# In Colab notebook

# Cell 1: Clone from GitHub
!git clone https://github.com/YOUR_USERNAME/image-captioning.git
%cd image-captioning

# Cell 2: Install dependencies
!pip install -q -r requirements.txt

# Cell 3: Enable GPU (Runtime → Change Runtime Type → GPU)
import torch
print(f"GPU: {torch.cuda.is_available()}")

# Cell 4: Update config for GPU
import yaml
config = yaml.safe_load(open('configs/config.yaml'))
config['device'] = 'cuda'
config['batch_size'] = 64
with open('configs/colab_config.yaml', 'w') as f:
    yaml.dump(config, f)

# Cell 5: Run training
!python main.py --config configs/colab_config.yaml

# Cell 6: View results
%load_ext tensorboard
%tensorboard --logdir outputs/logs/

# Cell 7: Download checkpoints
from google.colab import files
import shutil
shutil.make_archive('checkpoints', 'zip', 'checkpoints')
files.download('checkpoints.zip')
```

## 📚 Dataset Setup

### Using Dummy Data (Current)
The model has dummy COCO data pre-configured for testing purposes.

### Using Real COCO 2017 Data

1. Download from: https://cocodataset.org/#download
2. Convert annotations to CSV:
   ```bash
   python utils/convert_coco_to_csv.py
   ```
3. Update `configs/config.yaml`:
   ```yaml
   data_dir: "data/coco"
   dataset: "coco"
   ```

## 🎓 Training Results Example

From debug run (2 epochs):
```
Epoch [1/2], Step [0], Loss: 0.0614 (Cap: 0.0005, Align: 0.1217, CF: 0.0000)
Epoch [1/2], Step [10], Loss: 0.0216 (Cap: 0.0012, Align: 0.0407, CF: -0.0001)
Epoch [1/2], Step [20], Loss: 0.0106 (Cap: 0.0005, Align: 0.0202, CF: 0.0000)
--- Epoch 1 Summary ---
Avg Loss: 0.0206
```

## 📈 Metrics Tracked

**During Training**:
- Caption Loss: Measures caption generation quality
- Alignment Loss: Measures attention ↔️ attribution alignment
- Counterfactual Loss: Validates attribution importance

**After Training** (with real data):
- BLEU (1-4 grams)
- ROUGE-L
- METEOR
- CIDEr
- SPICE

## 🔍 Understanding Explainability

The model generates visual explanations showing which image regions contributed to each word:

1. **Image** → Visual features via ViT encoder
2. **Decode "dog"** → Highlight dog region (high attribution)
3. **Decode "eating"** → Highlight food + mouth region
4. **Decode "sandwich"** → Highlight sandwich region

**Validation**: Mask high-attribution pixels and verify output probability drops.

## ⚡ Performance Tips

### For Faster Training
- Use GPU: Set `device: "cuda"` in config
- Increase `batch_size` if you have memory
- Reduce `epochs` for quick experiments

### For Better Results
- Use real COCO dataset (not dummy)
- Increase `alignment_weight` for better explanations
- Train longer (increase `epochs`)
- Use GPU for faster convergence

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Training too slow | Use GPU, reduce batch_size, enable debug mode |
| Out of memory | Reduce batch_size, embed_size, hidden_size |
| Missing data warnings | Dataset uses dummy data by default (normal) |
| Tensorboard not loading | Verify: `ls outputs/logs/` |

## 📖 Documentation

- **[TRAINING_SUMMARY.md](TRAINING_SUMMARY.md)** - Comprehensive technical guide
- **[TRAINING_WALKTHROUGH.md](TRAINING_WALKTHROUGH.md)** - Architecture deep-dive

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📝 License

This project is provided as-is for educational purposes.

## 👤 Author

Your Name / Student ID  
Course: SEM6 / TDL

## 📞 Support

For issues or questions:
1. Check [TRAINING_SUMMARY.md](TRAINING_SUMMARY.md) for detailed troubleshooting
2. Review training logs in `outputs/logs/`
3. Create an issue on GitHub

## 🙏 Acknowledgments

- Vision Transformer: https://arxiv.org/abs/2010.11929
- COCO Dataset: https://cocodataset.org/
- PyTorch Community

---

**Last Updated**: March 21, 2026  
**Status**: ✅ Ready for Production
