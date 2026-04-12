"""
evaluate_checkpoint.py

Run BLEU evaluation on a specific local checkpoint.

Usage:
    python evaluate_checkpoint.py --checkpoint baseline_epoch_10.pt
    python evaluate_checkpoint.py --checkpoint proposed_epoch_09.pt
    python evaluate_checkpoint.py --checkpoint proposed_epoch_09.pt --max-samples 1000
"""
import argparse, yaml, torch
from utils.dataset import get_loaders
from models.caption_model import CaptionModel
from training.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Max validation images to evaluate (default 5000)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Force CPU if no GPU available locally
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Checkpoint: {args.checkpoint}")

    # ── Load vocab + val loader ──────────────────────────────────────────────
    config["debug"] = False   # make sure we get the real val set
    _, val_loader, vocab = get_loaders(config)
    vocab_size = len(vocab)
    print(f"Vocab size : {vocab_size}")
    print(f"Val batches: {len(val_loader)}")

    # ── Build model ──────────────────────────────────────────────────────────
    model = CaptionModel(
        encoder_type=config["encoder"],
        vocab_size=vocab_size,
        embed_size=config["embed_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["decoder_layers"],
        nhead=config["decoder_heads"],
        max_length=config["max_length"],
    ).to(device)

    # ── Load weights ─────────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device)
    # Support both raw state-dict saves and wrapped saves
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded weights (epoch {epoch})\n")

    # ── Patch max_eval_samples so we can control it from CLI ─────────────────
    import training.evaluate as eval_mod
    _orig = eval_mod.evaluate_model

    def patched_evaluate(model, val_loader, vocab, device):
        # Temporarily monkey-patch the hard-coded 5000 limit
        import nltk
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        model.eval()
        total_b1, total_b4, total_n = 0.0, 0.0, 0
        chencherry = SmoothingFunction()

        with torch.no_grad():
            for batch_idx, (images, targets, captions) in enumerate(val_loader):
                images = images.to(device)
                for i in range(images.size(0)):
                    if total_n >= args.max_samples:
                        break
                    image = images[i:i+1]
                    reference = vocab.tokenize(captions[i])
                    generated_words, _ = model.generate(image, vocab)
                    hypothesis = [w for w in generated_words
                                  if w not in (vocab.start_token, vocab.pad_token,
                                               vocab.unk_token, vocab.end_token)]
                    try:
                        b1 = sentence_bleu([reference], hypothesis,
                                           weights=(1,0,0,0),
                                           smoothing_function=chencherry.method1)
                        b4 = sentence_bleu([reference], hypothesis,
                                           weights=(.25,.25,.25,.25),
                                           smoothing_function=chencherry.method1)
                    except Exception:
                        b1, b4 = 0.0, 0.0
                    total_b1 += b1
                    total_b4 += b4
                    total_n  += 1

                if total_n >= args.max_samples:
                    print(f"  ... stopped at {total_n} samples")
                    break
                if (batch_idx + 1) % 10 == 0:
                    print(f"  [{total_n:>5}/{args.max_samples}]  "
                          f"running BLEU-1={total_b1/total_n:.4f}  "
                          f"BLEU-4={total_b4/total_n:.4f}")

        return {"BLEU-1": total_b1 / max(total_n, 1),
                "BLEU-4": total_b4 / max(total_n, 1),
                "samples": total_n}

    # ── Run evaluation ───────────────────────────────────────────────────────
    print("Starting evaluation ...\n")
    metrics = patched_evaluate(model, val_loader, vocab, device)

    print("\n" + "="*45)
    print("  FINAL EVALUATION RESULTS")
    print("="*45)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Samples    : {metrics['samples']}")
    print(f"  BLEU-1     : {metrics['BLEU-1']:.4f}")
    print(f"  BLEU-4     : {metrics['BLEU-4']:.4f}")
    print("="*45)

if __name__ == "__main__":
    main()
