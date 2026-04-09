"""
Train one model variant on Modal GPUs.

Usage:
    # Train baseline (no losses)
    modal run modal_train.py --run-name baseline --epochs 20

    # Train with alignment loss only
    modal run modal_train.py --run-name align_only --epochs 20

    # Train with counterfactual loss only
    modal run modal_train.py --run-name cf_only --epochs 20

    # Train proposed model (both losses)
    modal run modal_train.py --run-name proposed --epochs 20

    # List saved checkpoints
    modal run modal_train.py --action list

    # Download a checkpoint to your laptop
    modal run modal_train.py --action download --run-name baseline --epochs 20
"""
import modal

app = modal.App("caption-training")

# Persistent volumes
data_volume = modal.Volume.from_name("coco-dataset-vol")
checkpoints_volume = modal.Volume.from_name("caption-checkpoints-vol", create_if_missing=True)

# Docker image with all dependencies pre-installed
training_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch>=2.0.0", "torchvision>=0.15.0", "torchaudio>=2.0.0",
        "numpy>=1.21.0", "pandas>=1.3.0", "Pillow>=9.0.0",
        "PyYAML>=5.4.0", "matplotlib>=3.4.0", "scikit-image>=0.18.0",
        "nltk>=3.6.0", "pycocoevalcap>=1.2", "captum>=0.5.0",
        "tensorboard>=2.8.0", "tqdm>=4.62.0", "scipy>=1.7.0",
        "opencv-python-headless>=4.5.0", "requests>=2.26.0",
    )
    .run_commands(
        "python -c \"import nltk; nltk.download('punkt'); nltk.download('punkt_tab')\""
    )
)

# ── Model variant configurations ──────────────────────────
MODEL_CONFIGS = {
    "baseline":   {"use_alignment_loss": False, "use_counterfactual_loss": False},
    "align_only": {"use_alignment_loss": True,  "use_counterfactual_loss": False},
    "cf_only":    {"use_alignment_loss": False, "use_counterfactual_loss": True},
    "proposed":   {"use_alignment_loss": True,  "use_counterfactual_loss": True},
}

# ── Your GitHub repo URL ──────────────────────────────────
REPO_URL = "https://github.com/SwathiHRao28/tdlProject.git"


@app.function(
    image=training_image,
    gpu="A10G",                                # 24 GB VRAM, good price/perf
    volumes={
        "/data": data_volume,                  # COCO dataset
        "/checkpoints": checkpoints_volume,    # Persistent checkpoint storage
    },
    timeout=14400,                             # 4 hours max
)
def train(
    run_name: str = "baseline",
    epochs: int = 20,
    batch_size: int = 32,
    resume: bool = True,
):
    import os, subprocess, yaml, torch, sys, shutil

    # Resolve which losses to use
    loss_config = MODEL_CONFIGS.get(run_name)
    if loss_config is None:
        print(f"❌ Unknown run_name '{run_name}'. Choose from: {list(MODEL_CONFIGS.keys())}")
        return
    
    use_alignment = loss_config["use_alignment_loss"]
    use_counterfactual = loss_config["use_counterfactual_loss"]

    # ── 1. Clone the latest code ──────────────────────────
    repo_dir = "/tmp/project"
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)
    subprocess.run(["git", "clone", "--depth=1", REPO_URL, repo_dir], check=True)
    sys.path.insert(0, repo_dir)
    os.chdir(repo_dir)

    # ── 2. Symlink data so the code finds it at data/coco ─
    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/coco"):
        os.symlink("/data/coco", "data/coco")

    # Verify data exists
    for split in ["train2014", "val2014"]:
        img_dir = f"data/coco/images/{split}"
        if os.path.exists(img_dir):
            count = len(os.listdir(img_dir))
            print(f"📸 {split}: {count} images")
        else:
            print(f"❌ {split} not found at {img_dir}! Run modal_setup_data.py first.")
            return

    # ── 3. Setup checkpoint dirs ──────────────────────────
    local_ckpt_dir = f"checkpoints/{run_name}"
    persistent_ckpt_dir = f"/checkpoints/{run_name}"
    os.makedirs(local_ckpt_dir, exist_ok=True)
    os.makedirs(persistent_ckpt_dir, exist_ok=True)

    # Copy existing checkpoints from volume for resume
    if resume:
        existing = sorted([f for f in os.listdir(persistent_ckpt_dir) if f.endswith(".pt")])
        for f in existing:
            shutil.copy2(os.path.join(persistent_ckpt_dir, f), os.path.join(local_ckpt_dir, f))
            print(f"📂 Restored checkpoint: {f}")

    # ── 4. Override config.yaml ───────────────────────────
    config_path = "configs/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config.update({
        "debug": False,
        "epochs": epochs,
        "batch_size": batch_size,
        "checkpoint_dir": persistent_ckpt_dir, # Write DIRECTLY to the persistent volume!
        "log_dir": f"{persistent_ckpt_dir}/logs", # ⭐ Save TensorBoard logs persistently
        "output_dir": f"outputs/{run_name}",
        "use_alignment_loss": use_alignment,
        "use_counterfactual_loss": use_counterfactual,
        "data_dir": "data/coco",
        "device": "cuda",
        "num_workers": 4,
        "save_every": 1,                 # ⭐ Save checkpoint every 1 epoch!
        # 2000 steps × 32 batch = 64k samples/epoch (good balance of speed vs coverage)
        "max_steps_per_epoch": 2000,
        "max_train_steps_per_epoch": 2000,
        "max_val_steps_per_epoch": 500,
    })

    with open(config_path, "w") as f:
        yaml.dump(config, f)
        
    # ── 4b. Setup Background Commit Thread ────────────────
    # This guarantees that the volume pushes to Modal cloud every 2 minutes
    import threading, time
    def auto_commit():
        while True:
            time.sleep(120)
            try:
                checkpoints_volume.commit()
            except Exception:
                pass
    
    t = threading.Thread(target=auto_commit, daemon=True)
    t.start()

    # ── 5. Print run banner ───────────────────────────────
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\n{'='*60}")
    print(f"🚀 TRAINING: {run_name}")
    print(f"   Alignment Loss:      {use_alignment}")
    print(f"   Counterfactual Loss: {use_counterfactual}")
    print(f"   Epochs:              {epochs}")
    print(f"   Batch Size:          {batch_size}")
    print(f"   GPU:                 {gpu_name}")
    print(f"   Resume:              {resume}")
    print(f"{'='*60}\n")

    # ── 6. Run training ───────────────────────────────────
    result = subprocess.run(
        [sys.executable, "-u", "main.py", "--config", config_path],
        cwd=repo_dir,
    )

    if result.returncode != 0:
        print(f"\n❌ Training failed with exit code {result.returncode}")
        # Still try to save any checkpoints that were created
    
    # ── 7. Copy ALL checkpoints to persistent volume ──────
    print("\n💾 Saving checkpoints to persistent volume...")
    saved_count = 0
    for f in sorted(os.listdir(local_ckpt_dir)):
        if f.endswith(".pt"):
            src = os.path.join(local_ckpt_dir, f)
            dst = os.path.join(persistent_ckpt_dir, f)
            shutil.copy2(src, dst)
            size_mb = os.path.getsize(src) / (1024 * 1024)
            print(f"  ✅ {f} ({size_mb:.1f} MB)")
            saved_count += 1

    if saved_count == 0:
        print("  ⚠️  No checkpoint files found to save!")

    checkpoints_volume.commit()
    print(f"\n🎉 Done! {saved_count} checkpoint(s) saved for '{run_name}'.")


@app.function(
    image=training_image,
    volumes={"/checkpoints": checkpoints_volume},
    timeout=300,
)
def list_checkpoints():
    """List all saved checkpoints across all runs."""
    import os
    base = "/checkpoints"
    print("\n📦 All Saved Checkpoints:")
    print("=" * 50)

    if not os.listdir(base):
        print("  (empty — no training runs yet)")
        return

    for run_name in sorted(os.listdir(base)):
        run_dir = os.path.join(base, run_name)
        if not os.path.isdir(run_dir):
            continue
        files = sorted([f for f in os.listdir(run_dir) if f.endswith(".pt")])
        if files:
            print(f"\n  📁 {run_name}/")
            for f in files:
                size_mb = os.path.getsize(os.path.join(run_dir, f)) / (1024 * 1024)
                print(f"     {f}  ({size_mb:.1f} MB)")
        else:
            print(f"\n  📁 {run_name}/ (no .pt files)")


@app.function(
    image=training_image,
    volumes={"/checkpoints": checkpoints_volume},
    timeout=600,
)
def download_checkpoint(run_name: str, epoch: int) -> bytes:
    """Download a specific checkpoint file as bytes."""
    import os
    path = f"/checkpoints/{run_name}/epoch_{epoch:02d}.pt"
    if not os.path.exists(path):
        available = os.listdir(f"/checkpoints/{run_name}") if os.path.exists(f"/checkpoints/{run_name}") else []
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"Available files: {available}"
        )
    print(f"📥 Reading {path}...")
    with open(path, "rb") as f:
        return f.read()


@app.function(
    image=training_image,
    volumes={"/data": data_volume, "/checkpoints": checkpoints_volume},
    timeout=600,
)
def run_inference_remote(run_name: str, epoch: int) -> bytes:
    """Run inference against a few Validation images directly on the Volume and zip them up!"""
    import os, subprocess, sys, zipfile, io, shutil
    
    repo_dir = "/tmp/project"
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)
    subprocess.run(["git", "clone", "--depth=1", REPO_URL, repo_dir], check=True)
    os.chdir(repo_dir)

    # Symlink data so code works
    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/coco"):
        os.symlink("/data/coco", "data/coco")

    # Pick predefined validation images 
    # (Since modal_setup_data downloaded images/val2014)
    val_images_dir = "data/coco/images/val2014"
    import glob
    test_images = glob.glob(f"{val_images_dir}/*.jpg")[:3]
    
    ckpt_path = f"/checkpoints/{run_name}/epoch_{epoch:02d}.pt"
    
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        for i, img in enumerate(test_images):
            print(f"🔍 Running inference on {img}...")
            subprocess.run([sys.executable, "inference.py", "--image", img, "--checkpoint", ckpt_path])
            
            # Pack outputs
            if os.path.exists("outputs"):
                for out_file in os.listdir("outputs"):
                    if out_file.endswith(".png"):
                        zf.write(os.path.join("outputs", out_file), f"image_{i}_{out_file}")
                        os.remove(os.path.join("outputs", out_file))

    return memory_file.getvalue()

# ═══════════════════════════════════════════════════════════
# LOCAL ENTRYPOINT — Run from your laptop terminal
# ═══════════════════════════════════════════════════════════

@app.local_entrypoint()
def main(
    action: str = "train",           # "train", "list", "download", "infer"
    run_name: str = "baseline",      # "baseline", "align_only", "cf_only", "proposed"
    epochs: int = 20,
    batch_size: int = 32,
):
    """
    Examples:
        modal run modal_train.py --run-name baseline --epochs 20
        modal run modal_train.py --action infer --run-name baseline --epochs 10
    """
    if action == "train":
        print(f"🚀 Launching training for '{run_name}'...")
        train.remote(run_name=run_name, epochs=epochs, batch_size=batch_size)

    elif action == "list":
        list_checkpoints.remote()

    elif action == "download":
        data = download_checkpoint.remote(run_name=run_name, epoch=epochs)
        local_path = f"{run_name}_epoch_{epochs:02d}.pt"
        with open(local_path, "wb") as f:
            f.write(data)
        print(f"✅ Downloaded checkpoint → {local_path} ({len(data) / (1024*1024):.1f} MB)")
        
    elif action == "infer":
        print(f"🖼️ Running cloud inference and zipping visuals...")
        zip_bytes = run_inference_remote.remote(run_name=run_name, epoch=epochs)
        zip_path = f"visuals_{run_name}_epoch_{epochs:02d}.zip"
        with open(zip_path, "wb") as f:
            f.write(zip_bytes)
        print(f"✅ Downloaded inference images → {zip_path}")

    else:
        print(f"❌ Unknown action '{action}'. Use: train, list, download")
