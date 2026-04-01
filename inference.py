import os
import argparse
import yaml
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot import plt
import numpy as np

from models.caption_model import CaptionModel
from explainability.attribution import compute_batch_attribution
from utils.preprocessing import get_transforms
from utils.dataset import ImageCaptionDataset

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def visualize_and_save(image_tensor, caption_words, attn_weights, attr_maps, masked_image, output_prefix="output"):
    os.makedirs("outputs", exist_ok=True)
    
    # Denormalize image for display
    # Image tensor is (1, 3, H, W)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(image_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image_tensor.device)
    img_disp = image_tensor[0] * std + mean
    img_disp = img_disp.permute(1, 2, 0).cpu().numpy()
    img_disp = np.clip(img_disp, 0, 1)
    
    caption_str = " ".join(caption_words)
    
    # 1. Save original image with caption
    plt.figure(figsize=(6, 6))
    plt.imshow(img_disp)
    plt.title(f"Caption: {caption_str}")
    plt.axis("off")
    plt.savefig(f"outputs/{output_prefix}_caption.png", bbox_inches='tight')
    plt.close()
    
    seq_len = len(caption_words)
    
    # 2. Save Attention and Attribution Maps
    # Create a grid for each word
    fig, axes = plt.subplots(2, seq_len, figsize=(3 * seq_len, 6))
    
    if seq_len == 1:
        axes = np.expand_dims(axes, axis=1)
        
    for i, word in enumerate(caption_words):
        # Attention map for word i
        # attn_weights shape is (1, seq_len, num_pixels)
        if attn_weights is not None:
            attn = attn_weights[0, i].view(14, 14).detach().cpu().numpy()
            axes[0, i].imshow(img_disp)
            axes[0, i].imshow(np.kron(attn, np.ones((16, 16))), alpha=0.5, cmap='jet')
        axes[0, i].set_title(f"Attn: {word}")
        axes[0, i].axis("off")
        
        # Attribution map for word i
        # attr_maps shape is (1, seq_len, num_pixels)
        if attr_maps is not None:
            attr = attr_maps[0, i].view(14, 14).detach().cpu().numpy()
            axes[1, i].imshow(img_disp)
            axes[1, i].imshow(np.kron(attr, np.ones((16, 16))), alpha=0.5, cmap='hot')
        axes[1, i].set_title(f"Attr: {word}")
        axes[1, i].axis("off")
        
    plt.tight_layout()
    plt.savefig(f"outputs/{output_prefix}_attention.png", bbox_inches='tight')
    plt.close()
    
    # 3. Save Counterfactual Image
    # Just show the masked image for the first highly-attributed word (e.g. argmax over all words)
    if attr_maps is not None:
        total_attr = attr_maps[0].sum(dim=0) # (num_pixels)
        k = max(1, int(196 * 0.2))
        topk_vals, _ = torch.topk(total_attr, k)
        threshold = topk_vals[-1]
        mask = (total_attr < threshold).view(14, 14).float().detach().cpu().numpy()
        upscaled_mask = np.kron(mask, np.ones((16, 16)))
        upscaled_mask = np.expand_dims(upscaled_mask, axis=-1)
        
        masked_img_disp = img_disp * upscaled_mask
        
        plt.figure(figsize=(6, 6))
        plt.imshow(masked_img_disp)
        plt.title("Counterfactual Masked Image")
        plt.axis("off")
        plt.savefig(f"outputs/{output_prefix}_counterfactual.png", bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config.get("device", "cpu"))
    
    # Try to load vocabulary from dummy dataset for mapping
    # Since we didn't save vocab.pkl to disk during training, we MUST rebuild
    # the exact same vocabulary natively by reading the full training captions.
    print("Rebuilding vocabulary from full training dataset (takes a few seconds)...")
    dummy_dataset = ImageCaptionDataset(
        data_root=config["data_dir"], 
        split="train", 
        debug=False, 
        freq_threshold=config.get("min_word_freq", 5)
    )
    vocab = dummy_dataset.vocab
    
    print(f"Loading image from {args.image}")
    transform = get_transforms(image_size=config["image_size"], is_train=False)
    image = Image.open(args.image).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Initialize model
    model = CaptionModel(
        encoder_type=config["encoder"],
        vocab_size=len(vocab),
        embed_size=config["embed_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["decoder_layers"],
        nhead=config["decoder_heads"],
        max_length=config["max_length"]
    ).to(device)
    
    # Load checkpoint
    checkpoint_dir = config["checkpoint_dir"]
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")])
        if checkpoints:
            latest_ckpt = os.path.join(checkpoint_dir, checkpoints[-1])
            print(f"Loading checkpoint: {latest_ckpt}")
            checkpoint = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("No checkpoints found. Running with untrained random weights!")
    else:
        print("No checkpoints found. Running with untrained random weights!")

    model.eval()

    # Generate caption
    words, attn_weights = model.generate(image_tensor, vocab)
    
    # Filter out special tokens
    display_words = [w for w in words if w not in [vocab.start_token, vocab.end_token, vocab.pad_token, vocab.unk_token]]
    if not display_words:
        display_words = ["<empty>"]
        
    print(f"Generated Caption: {' '.join(display_words)}")

    # Compute attribution for the generated sequence
    # For this, we treat the generated sequence as the "true" target
    gen_idx = [vocab.stoi.get(w, vocab.stoi[vocab.unk_token]) for w in words]
    gen_tensor = torch.tensor([vocab.stoi[vocab.start_token]] + gen_idx).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = model.encoder(image_tensor)
        features = model.vis_project(features)
        
    attr_maps = compute_batch_attribution(model, features, gen_tensor[:, :-1], vocab.stoi[vocab.pad_token])
    
    # We truncate attention and attribution maps to match the display words
    # length of words is say N. display_words is N-2. 
    # attn_weights might have shape (1, N-1, 196) (from generate)
    seq_len = min(len(display_words), attr_maps.shape[1], attn_weights.shape[1] if attn_weights is not None else 999)
    if attn_weights is not None:
        attn_weights = attn_weights[:, :seq_len, :]
    attr_maps = attr_maps[:, :seq_len, :]
    
    # Generate Output Visualizations
    output_prefix = os.path.splitext(os.path.basename(args.image))[0]
    visualize_and_save(image_tensor, display_words[:seq_len], attn_weights, attr_maps, None, output_prefix=output_prefix)
    
    print(f"Visualizations saved in outputs/ directory.")

if __name__ == "__main__":
    main()
