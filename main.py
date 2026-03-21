import os
import argparse
import yaml
import torch
from utils.dataset import get_loaders
from models.caption_model import CaptionModel
from training.train import train_model

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="Train Explainable Image Captioning Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--debug", action="store_true", default=False, help="Run in debug mode (subset of data)")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Override debug mode if passed via args
    if args.debug:
        config["debug"] = True
        print(f"--- Running in DEBUG MODE ---")
        
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Set seeds
    torch.manual_seed(config["seed"])

    # Load Data
    print("Loading data...")
    train_loader, val_loader, vocab = get_loaders(config)
    
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    # Initialize Model
    print("Initializing model...")
    model = CaptionModel(
        encoder_type=config["encoder"],
        vocab_size=vocab_size,
        embed_size=config["embed_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["decoder_layers"],
        nhead=config["decoder_heads"],
        max_length=config["max_length"]
    ).to(device)

    # Resume from checkpoint if it exists and we're not starting fresh
    start_epoch = 1
    checkpoint_dir = config["checkpoint_dir"]
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")])
        if checkpoints:
            latest_ckpt = os.path.join(checkpoint_dir, checkpoints[-1])
            print(f"Resuming from checkpoint: {latest_ckpt}")
            checkpoint = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1

    # Train
    print("Starting training...")
    train_model(model, train_loader, val_loader, vocab, config, device)
    
    # Final evaluation
    if not config["debug"] and val_loader is not None:
        print("Evaluating model...")
        from training.evaluate import evaluate_model
        metrics = evaluate_model(model, val_loader, vocab, device)
        print("Evaluation Metrics:", metrics)

if __name__ == "__main__":
    main()
