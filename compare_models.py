import os
import argparse
import random
import yaml
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob

from models.caption_model import CaptionModel
from explainability.attribution import compute_batch_attribution
from utils.preprocessing import get_transforms
from utils.dataset import ImageCaptionDataset

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_model(config, vocab_size, ckpt_path, device):
    model = CaptionModel(
        encoder_type=config["encoder"],
        vocab_size=vocab_size,
        embed_size=config["embed_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["decoder_layers"],
        nhead=config["decoder_heads"],
        max_length=config["max_length"]
    ).to(device)
    
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

def run_inference(model, image_tensor, vocab, device):
    words, attn_weights = model.generate(image_tensor, vocab)
    display_words = [w for w in words if w not in [vocab.start_token, vocab.end_token, vocab.pad_token, vocab.unk_token]]
    if not display_words: display_words = ["<empty>"]
    
    gen_idx = [vocab.stoi.get(w, vocab.stoi[vocab.unk_token]) for w in words]
    gen_tensor = torch.tensor([vocab.stoi[vocab.start_token]] + gen_idx).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = model.encoder(image_tensor)
        features = model.vis_project(features)
        
    attr_maps = compute_batch_attribution(model, features, gen_tensor[:, :-1], vocab.stoi[vocab.pad_token])
    
    seq_len = min(len(display_words), attr_maps.shape[1], attn_weights.shape[1] if attn_weights is not None else 999)
    if attn_weights is not None: attn_weights = attn_weights[:, :seq_len, :]
    attr_maps = attr_maps[:, :seq_len, :]
    
    return display_words[:seq_len], attr_maps

def save_comparison_plot(image_tensor, bas_words, bas_attr, prop_words, prop_attr, output_path):
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(image_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image_tensor.device)
    img_disp = image_tensor[0] * std + mean
    img_disp = img_disp.permute(1, 2, 0).cpu().numpy()
    img_disp = np.clip(img_disp, 0, 1)
    
    bas_caption = " ".join(bas_words)
    prop_caption = " ".join(prop_words)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Ablation Analysis: Explainability vs Baseline\n", fontsize=18, fontweight='bold')
    
    # Baseline Original
    axes[0, 0].imshow(img_disp)
    axes[0, 0].set_title(f"[Baseline Output]:\n{bas_caption}", fontsize=12, wrap=True)
    axes[0, 0].axis("off")
    
    # Baseline Combined Attribution (sum across all words to show general focus)
    if bas_attr is not None:
        agg_attr = bas_attr[0].sum(dim=0).view(14, 14).detach().cpu().numpy()
        axes[0, 1].imshow(img_disp)
        axes[0, 1].imshow(np.kron(agg_attr, np.ones((16, 16))), alpha=0.6, cmap='hot')
    axes[0, 1].set_title("Baseline: Internal Focus Heatmap", fontsize=12, fontweight='bold')
    axes[0, 1].axis("off")
    
    # Proposed Original
    axes[1, 0].imshow(img_disp)
    axes[1, 0].set_title(f"[Proposed Output]:\n{prop_caption}", fontsize=12, wrap=True)
    axes[1, 0].axis("off")
    
    # Proposed Combined Attribution
    if prop_attr is not None:
        agg_attr = prop_attr[0].sum(dim=0).view(14, 14).detach().cpu().numpy()
        axes[1, 1].imshow(img_disp)
        axes[1, 1].imshow(np.kron(agg_attr, np.ones((16, 16))), alpha=0.6, cmap='jet')
    axes[1, 1].set_title("Proposed: Internal Focus Heatmap", fontsize=12, fontweight='bold')
    axes[1, 1].axis("off")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True, help="Path to baseline model .pt file")
    parser.add_argument("--proposed", type=str, required=True, help="Path to proposed model .pt file")
    parser.add_argument("--images-dir", type=str, default="data/coco/images/val2014")
    parser.add_argument("--num", type=int, default=100, help="Number of random val images to scan")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Rebuilding vocabulary from full training dataset...")
    import sys
    sys.path.append(os.getcwd())
    dummy_dataset = ImageCaptionDataset(data_root=config["data_dir"], split="train", debug=False, freq_threshold=config.get("min_word_freq", 5))
    vocab = dummy_dataset.vocab

    print("\nLoading Baseline Model...")
    baseline_model = load_model(config, len(vocab), args.baseline, device)
    
    print("Loading Proposed Model...")
    proposed_model = load_model(config, len(vocab), args.proposed, device)

    transform = get_transforms(image_size=config["image_size"], is_train=False)
    
    images = glob.glob(f"{args.images_dir}/*.jpg")
    if not images:
        print(f"No images found in {args.images_dir}")
        return
        
    random.shuffle(images)
    subset = images[:args.num]
    
    os.makedirs("comparisons", exist_ok=True)
    
    diff_count = 0
    print(f"\nScanning {len(subset)} random validation images for caption differences...")
    print("-" * 50)
    
    for i, img_path in enumerate(subset):
        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        bas_words, bas_attr = run_inference(baseline_model, image_tensor, vocab, device)
        prop_words, prop_attr = run_inference(proposed_model, image_tensor, vocab, device)
        
        bas_caption = " ".join(bas_words)
        prop_caption = " ".join(prop_words)
        
        if bas_caption != prop_caption:
            diff_count += 1
            print(f"\n[{diff_count}] Image: {img_path}")
            print(f"  🔴 Baseline: {bas_caption}")
            print(f"  🟢 Proposed: {prop_caption}")
            
            output_name = f"comparisons/diff_{diff_count}_{os.path.basename(img_path)}"
            save_comparison_plot(image_tensor, bas_words, bas_attr, prop_words, prop_attr, output_name)
            
    print(f"\n✅ Finished! Discovered {diff_count} differing captions out of {args.num} scanned.")
    print("🖼️  Check the 'comparisons/' folder for gorgeous 4-panel side-by-side matrices ready for your paper!")

if __name__ == "__main__":
    main()
