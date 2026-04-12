import modal
import os
import sys

app = modal.App("java-eval-robust")

# ── Persistent volumes ──────────────────────────────────────────
data_volume = modal.Volume.from_name("coco-dataset-vol")
checkpoints_volume = modal.Volume.from_name("caption-checkpoints-vol")

# ── Repository ──────────────────────────────────────────────────
REPO_URL = "https://github.com/SwathiHRao28/tdlProject.git"

# ── Robust image setup using aac-metrics ────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "default-jre", "wget", "unzip")
    .pip_install(
        "torch", "torchvision", "pycocoevalcap", "pycocotools", 
        "tqdm", "pyyaml", "nltk", "pandas", "Pillow", "aac-metrics"
    )
    .run_commands(
        "python -m nltk.downloader punkt punkt_tab",
        "python -m aac_metrics.download"
    )
)

@app.function(
    image=image,
    gpu="A10G",
    memory=16384, 
    volumes={"/data": data_volume, "/checkpoints": checkpoints_volume},
    timeout=7200, 
)
def evaluate_epoch_robust(run_name: str, epoch: int, max_samples: int = 5000):
    import torch
    import yaml
    import json
    import shutil
    import subprocess
    from tqdm import tqdm
    from pycocotools.coco import COCO
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    from aac_metrics.classes.meteor import METEOR
    from aac_metrics.classes.spice import SPICE

    print(f"\n--- Starting Robust Evaluation: {run_name} Epoch {epoch} ---")

    # 1. Setup local environment
    repo_dir = f"/tmp/project_{epoch}"
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)
    subprocess.run(["git", "clone", "--depth=1", REPO_URL, repo_dir], check=True)
    sys.path.insert(0, repo_dir)
    os.chdir(repo_dir)

    # Symlink data
    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/coco"):
        os.symlink("/data/coco", "data/coco")

    # 2. Load dependencies from project
    from utils.dataset import get_loaders
    from models.caption_model import CaptionModel

    # Load config
    config_path = "configs/config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config.update({
        "debug": False, 
        "data_dir": "data/coco", 
        "device": "cuda", 
        "num_workers": 4,
        "max_val_steps_per_epoch": 500
    })

    # Get data loaders
    _, val_loader, vocab = get_loaders(config)
    device = torch.device("cuda")

    # 3. Build model and load checkpoint
    model = CaptionModel(
        encoder_type=config["encoder"],
        vocab_size=len(vocab),
        embed_size=config["embed_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["decoder_layers"],
        nhead=config["decoder_heads"],
        max_length=config["max_length"],
    ).to(device)

    ckpt_path = f"/checkpoints/{run_name}/epoch_{epoch:02d}.pt"
    if not os.path.exists(ckpt_path):
        return {"error": f"Checkpoint {ckpt_path} not found"}
        
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    # 4. Generate Captions
    gts_dict = {} # for pycocoevalcap (Bleu/Cider)
    res_dict = {}
    
    candidates = [] # for aac-metrics
    multireferences = []
    
    total = 0
    with torch.no_grad():
        for images, _, captions in tqdm(val_loader, desc=f"Generating Ep {epoch}"):
            images = images.to(device)
            for i in range(images.size(0)):
                if total >= max_samples: break
                
                img_id = total
                true_cap = captions[i]
                
                # Inference
                words, _ = model.generate(images[i:i+1], vocab)
                hyp = " ".join([w for w in words if w not in (vocab.start_token, vocab.pad_token, vocab.unk_token, vocab.end_token)])
                
                # Format for pycocoevalcap
                gts_dict[img_id] = [true_cap]
                res_dict[img_id] = [hyp]
                
                # Format for aac-metrics
                candidates.append(hyp)
                multireferences.append([true_cap])
                
                total += 1
            if total >= max_samples: break

    print(f"Computed captions for {total} images. Running scorers...")

    all_scores = {}

    # 5a. Run pycocoevalcap metrics (Bleu, Rouge, Cider)
    print("Computing Bleu...")
    bleu_score, _ = Bleu(4).compute_score(gts_dict, res_dict)
    for i, s in enumerate(bleu_score, 1):
        all_scores[f"Bleu_{i}"] = s

    print("Computing CIDEr...")
    all_scores["CIDEr"], _ = Cider().compute_score(gts_dict, res_dict)

    print("Computing ROUGE_L...")
    all_scores["ROUGE_L"], _ = Rouge().compute_score(gts_dict, res_dict)

    # 5b. Run aac-metrics (METEOR, SPICE)
    # These will automatically download JARs on the first call
    print("Computing METEOR (robust)...")
    meteor_scorer = METEOR()
    meteor_score, _ = meteor_scorer(candidates, multireferences)
    all_scores["METEOR"] = float(meteor_score["meteor"])

    print("Computing SPICE (robust)...")
    spice_scorer = SPICE()
    spice_score, _ = spice_scorer(candidates, multireferences)
    all_scores["SPICE"] = float(spice_score["spice"])

    return {
        "epoch": epoch,
        "results": all_scores
    }

@app.local_entrypoint()
def main(run_name: str = "proposed", max_samples: int = 5000):
    epochs = [9, 12, 15, 19]
    summary_file = "evaluation_summary_robust.txt"
    
    with open(summary_file, "w") as f:
        f.write(f"ROBUST EVALUATION SUMMARY FOR: {run_name}\n")
        f.write("="*60 + "\n")

    results_list = []
    print(f"🚀 Launching Parallel Robust Evaluation for epochs {epochs}...")
    
    for result in evaluate_epoch_robust.map(
        [run_name]*len(epochs), 
        epochs, 
        [max_samples]*len(epochs)
    ):
        if "error" in result:
            print(f"❌ Error for epoch {result.get('epoch')}: {result['error']}")
            continue
            
        res = result["results"]
        ep = result["epoch"]
        results_list.append((ep, res))
        
        line = f"Epoch {ep:02d}: B-4: {res['Bleu_4']:.4f} | CIDEr: {res['CIDEr']:.4f} | ROUGE_L: {res['ROUGE_L']:.4f} | METEOR: {res['METEOR']:.4f} | SPICE: {res['SPICE']:.4f}"
        print(f"✅ {line}")
        
    results_list.sort(key=lambda x: x[0])
    
    with open(summary_file, "a") as f:
        for ep, res in results_list:
            f.write(f"\nEPOCH {ep:02d}\n")
            f.write("-" * 30 + "\n")
            for metric in sorted(res.keys()):
                f.write(f"  {metric:<12}: {res[metric]:.4f}\n")
            f.write("-" * 30 + "\n")

    print(f"\n🎉 Evaluation Complete! Summary saved to {summary_file}")
