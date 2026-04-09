import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def evaluate_model(model, val_loader, vocab, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    
    total_bleu1 = 0.0
    total_bleu4 = 0.0
    total_samples = 0
    
    # We use a smoothing function for BLEU
    chencherry = SmoothingFunction()
    
    with torch.no_grad():
        for batch_idx, (images, targets, captions) in enumerate(val_loader):
            images = images.to(device)
            # targets is not necessarily used if we use greedy generation
            
            for i in range(images.size(0)):
                image = images[i:i+1] # (1, 3, H, W)
                true_caption = captions[i]
                
                # Tokenize true caption
                reference = vocab.tokenize(true_caption)
                
                generated_words, _ = model.generate(image, vocab)
                # Remove special tokens from generated words
                hypothesis = []
                for w in generated_words:
                    if w in [vocab.start_token, vocab.pad_token, vocab.unk_token]:
                        continue
                    if w == vocab.end_token:
                        break
                    hypothesis.append(w)
                
                # Compute BLEU
                try:
                    b1 = sentence_bleu([reference], hypothesis, weights=(1, 0, 0, 0), smoothing_function=chencherry.method1)
                    b4 = sentence_bleu([reference], hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)
                except:
                    b1, b4 = 0.0, 0.0
                    
                total_bleu1 += b1
                total_bleu4 += b4
                total_samples += 1
                
            # Stop early to save hours of evaluation time!
            # We'll evaluate on 5000 images (Karpathy test split standard size)
            max_eval_samples = 5000
            if total_samples >= max_eval_samples:
                print(f"Stopping Evaluation early at {total_samples} samples for speed!")
                break
                
            # Limit evaluation for speed in debug mode
            if getattr(val_loader.dataset, 'debug', False) and batch_idx >= 1:
                break
                
    avg_bleu1 = total_bleu1 / max(total_samples, 1)
    avg_bleu4 = total_bleu4 / max(total_samples, 1)
    
    return {
        "BLEU-1": avg_bleu1,
        "BLEU-4": avg_bleu4
    }
