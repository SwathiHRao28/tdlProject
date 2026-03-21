# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-03-21

### Added
- Initial release with working image captioning model
- Vision Transformer (ViT-B/16) encoder
- Transformer decoder with teacher forcing
- Gradient-based attribution maps for explainability
- Alignment loss for consistency between attention and attribution
- Counterfactual loss for verification
- CPU & GPU support
- TensorBoard integration for training visualization
- Debug mode for quick testing
- Comprehensive documentation

### Features
- Image captioning with explainable attention
- Multiple loss components (caption, alignment, counterfactual)
- Fast attribution mode for CPU debugging
- Full gradient-based attribution for accuracy
- Checkpoint saving and resuming
- Configuration-based hyperparameter tuning

### Fixed
- CLI argument parsing (--debug flag)
- Attribution computation performance (10-20× speedup)

### Known Issues
- PyTorch deprecation warning for mask types (non-critical)
- Dummy COCO data by default (real data requires download)
- CPU training is slow (recommend GPU for production)

## [0.9.0] - 2026-03-20

### In Progress
- Full COCO dataset support with real captions
- Additional encoder architectures (ResNet, BLIP, CLIP)
- Mixed precision training
- Distributed training support
- Model quantization for deployment
- Interactive web UI for inference

## Unreleased

### Planned
- [ ] Real COCO dataset integration
- [ ] Additional metrics (SPICE, evaluation consistency)
- [ ] Model compression for mobile deployment
- [ ] API for inference
- [ ] Colab notebook template
- [ ] Docker container for reproducibility

### In consideration
- [ ] Multilingual caption generation
- [ ] Video captioning support
- [ ] Dense caption generation (per region)
- [ ] Zero-shot caption generation
- [ ] Prompt-based caption generation (with LLM)

---

## Version History

### v1.0.0 (Current)
- ✅ Working training pipeline
- ✅ Debug mode functional
- ✅ TensorBoard logging
- ✅ Checkpoint system
- ✅ Fast & accurate attribution modes

### v0.9.0
- 🔧 In development

---

## How to Update

### From v0.9 to v1.0
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

### Release Notes for v1.0.0

**Major Changes:**
- Stable training pipeline
- Bug fixes for argument parsing
- Performance optimization (attribution computation)
- Comprehensive documentation

**Backward Compatibility:**
- All configurations are compatible
- Model checkpoints from v0.9 can be loaded

**Migration Guide:**
- No breaking changes
- Simply pull latest and retrain

---

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute changes that will be reflected in the changelog.

---

Generated: March 21, 2026
