"""
One-time data setup: Downloads COCO 2014 dataset to a Modal Volume.

Usage:
    py -m modal run modal_setup_data.py

This downloads ~20 GB. Takes about 20-30 minutes.
You only need to run this ONCE — the data persists on the volume.
"""
import modal

app = modal.App("coco-data-setup")
volume = modal.Volume.from_name("coco-dataset-vol", create_if_missing=True)

COCO_URLS = {
    "train2014": "http://images.cocodataset.org/zips/train2014.zip",
    "val2014": "http://images.cocodataset.org/zips/val2014.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
}

@app.function(
    image=modal.Image.debian_slim().apt_install("unzip").pip_install("requests", "tqdm"),
    volumes={"/data": volume},
    timeout=7200,           # 2 hours max
)
def download_and_extract():
    import os, subprocess, glob, shutil, requests
    from tqdm import tqdm

    data_dir = "/data/coco"
    os.makedirs(f"{data_dir}/images", exist_ok=True)
    os.makedirs(f"{data_dir}/captions", exist_ok=True)

    for name, url in COCO_URLS.items():
        zip_path = f"/tmp/{name}.zip"
        marker = f"{data_dir}/.done_{name}"

        if os.path.exists(marker):
            print(f"✅ {name} already downloaded, skipping.")
            continue

        # Download
        print(f"\n⬇️  Downloading {name} from {url}...")
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(zip_path, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc=name) as pbar:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        # Extract
        print(f"📦 Extracting {name}...")
        if name == "annotations":
            # Extract to /tmp first, then copy caption files to the volume
            subprocess.run(["unzip", "-o", zip_path, "-d", "/tmp/ann_tmp"], check=True)
            for src_file in glob.glob("/tmp/ann_tmp/annotations/captions_*.json"):
                dest = f"{data_dir}/captions/{os.path.basename(src_file)}"
                shutil.copy2(src_file, dest)  # copy2 works across filesystems (rename does NOT)
                print(f"  → {dest}")
            # Cleanup temp extraction
            shutil.rmtree("/tmp/ann_tmp", ignore_errors=True)
        else:
            # Image zips extract to train2014/ or val2014/ folder
            # Extract directly to the volume
            subprocess.run(["unzip", "-o", zip_path, "-d", f"{data_dir}/images/"], check=True)

        # Mark complete
        with open(marker, "w") as mf:
            mf.write("done")

        # Cleanup zip to free disk space
        os.remove(zip_path)
        print(f"✅ {name} done!")

        # ⭐ COMMIT AFTER EACH DATASET so progress is saved even if next one crashes
        print(f"💾 Committing {name} to volume...")
        volume.commit()
        print(f"💾 {name} committed!\n")

    # Verify final state
    print("\n" + "=" * 50)
    print("📊 VERIFICATION")
    print("=" * 50)
    for split in ["train2014", "val2014"]:
        img_dir = f"{data_dir}/images/{split}"
        if os.path.exists(img_dir):
            count = len(os.listdir(img_dir))
            print(f"  📸 {split}: {count} images")
        else:
            print(f"  ❌ {split}: MISSING!")

    cap_dir = f"{data_dir}/captions"
    if os.path.exists(cap_dir) and os.listdir(cap_dir):
        for f in sorted(os.listdir(cap_dir)):
            size_mb = os.path.getsize(f"{cap_dir}/{f}") / (1024 * 1024)
            print(f"  📝 {f}: {size_mb:.1f} MB")
    else:
        print("  ❌ No caption files found!")

    print("\n🎉 All data saved to Modal Volume 'coco-dataset-vol'!")

@app.local_entrypoint()
def main():
    download_and_extract.remote()

