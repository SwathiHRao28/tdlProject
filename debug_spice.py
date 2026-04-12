import modal
import os
import subprocess

app = modal.App("debug-spice")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "default-jre", "wget")
    .pip_install("pycocoevalcap")
    .run_commands(
        "git clone --depth 1 https://github.com/tylin/coco-caption.git /tmp/coco-caption",
        "PYCOCO_PATH=$(python -c 'import pycocoevalcap; print(pycocoevalcap.__path__[0])') && "
        "cp -r /tmp/coco-caption/pycocoevalcap/spice/* $PYCOCO_PATH/spice/"
    )
)

@app.function(image=image)
def debug():
    import pycocoevalcap
    spice_path = os.path.join(pycocoevalcap.__path__[0], "spice")
    
    print(f"Checking SPICE path: {spice_path}")
    print("Contents:")
    print(subprocess.check_output(["ls", "-R", spice_path]).decode())
    
    print("\nTesting java version:")
    print(subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT).decode())
    
    print("\nTesting SPICE jar help:")
    os.chdir(spice_path)
    try:
        out = subprocess.check_output(["java", "-jar", "spice-1.0.jar", "-help"], stderr=subprocess.STDOUT).decode()
        print(out)
    except subprocess.CalledProcessError as e:
        print(f"Execution failed with code {e.returncode}")
        print(f"Output: {e.output.decode()}")

@app.local_entrypoint()
def main():
    debug.remote()
