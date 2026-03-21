# Contributing to Explainable Image Captioning

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and constructive
- Provide detailed issue descriptions
- Follow Python PEP 8 style guide
- Write clear commit messages

## Getting Started

### 1. Fork the Repository
Click the "Fork" button on GitHub to create your copy.

### 2. Clone Your Fork
```bash
git clone https://github.com/YOUR_USERNAME/image-captioning.git
cd image-captioning
```

### 3. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-description
```

### 4. Install Development Dependencies
```bash
pip install -r requirements.txt
pip install flake8 black pytest
```

## Making Changes

### Code Style
- Follow PEP 8
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep lines under 100 characters

### Example Format
```python
def compute_batch_attribution(model, features, captions, pad_idx, fast_mode=False):
    """
    Compute attribution maps for all non-padded words in the sequence.
    
    Args:
        model: The CaptionModel
        features: (B, num_pixels, hidden_size) - visual features
        captions: (B, seq_len) - input captions
        pad_idx: Padding index
        fast_mode: If True, use L2-norm approximation
    
    Returns:
        attribution_maps: (B, seq_len, num_pixels) - attribution heatmaps
    """
    # Implementation
    pass
```

### Testing
```bash
# Lint your code
flake8 models/ training/ utils/

# Test imports
python -c "from models.caption_model import CaptionModel"

# Run debug mode
python main.py --debug
```

## Committing Changes

### Write Clear Commit Messages
```
feat: Add new feature
fix: Fix critical bug
docs: Update documentation
refactor: Improve code structure
test: Add unit tests
```

Examples:
```bash
git commit -m "feat: Add GPU support for training"
git commit -m "fix: Resolve memory leak in attribution computation"
git commit -m "docs: Add Colab setup instructions"
```

### Push to Your Fork
```bash
git push origin feature/your-feature-name
```

## Creating a Pull Request

1. Go to GitHub and click "Compare & pull request"
2. Write a clear description of changes
3. Link any related issues: `Closes #123`
4. Describe what the PR does and why

### PR Template
```
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Performance improvement

## Testing
How to test these changes?

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Debug mode passes
- [ ] No breaking changes
```

## Areas for Contribution

### Code Improvements
- Optimize attribution computation
- Add more encoder variants (BLIP, CLIP)
- Implement mixed precision training
- Add data augmentation

### Documentation
- Improve docstrings
- Add usage examples
- Create tutorials
- Document configuration options

### Testing
- Add unit tests
- Add integration tests
- Improve test coverage

### Features
- New explainability methods
- Additional evaluation metrics
- Interactive visualization tools
- Model deployment options

## Reporting Issues

### Bug Report Template
```
## Description
Clear description of the bug

## Environment
- OS: Windows/Mac/Linux
- Python: 3.8/3.9/3.10
- PyTorch version: 2.0.0

## Steps to Reproduce
1. Run command X
2. Observe error Y

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Screenshots
If applicable
```

### Feature Request Template
```
## Description
Clear description of feature

## Motivation
Why is this needed?

## Implementation
How would you implement it?

## Additional Context
Any other information
```

## Development Workflow

```
1. Create feature branch
   git checkout -b feature/my-feature

2. Make changes and test
   python main.py --debug

3. Commit changes
   git commit -m "feat: Add my feature"

4. Push to fork
   git push origin feature/my-feature

5. Create pull request on GitHub

6. Address review comments

7. Merge when approved
```

## Questions?

- Check [README.md](README.md) for general info
- Review [TRAINING_SUMMARY.md](TRAINING_SUMMARY.md) for technical details
- Open a GitHub discussion for questions

## License

By contributing, you agree your code will be under the same license as the project.

---

Thank you for contributing! 🚀
