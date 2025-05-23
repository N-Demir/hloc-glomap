"""
Sets up an SSH server in a Modal container.

This requires you to `pip install sshtunnel` locally.

After running this with `modal run launch_server.py`, connect to SSH with `ssh -p 9090 root@localhost`,
or from VSCode/Pycharm.

This uses simple password authentication, but you can store your own key in a modal Secret instead.
"""
from enum import Enum
from pathlib import Path
import modal
import threading
import socket
import subprocess
import time

app = modal.App(
    "example-get-started",
    image=modal.Image.from_dockerfile("Dockerfile")
    .apt_install("openssh-server")
    .run_commands(
        "mkdir -p /run/sshd",
        "echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config",
        "echo 'root: ' | chpasswd"
    )
    .workdir("/root")
    .run_commands(
        "git clone https://github.com/N-Demir/hloc-glomap.git"
    )
    .workdir("/root/hloc-glomap")
    .run_commands(
        "pixi install",
        "pixi run post-install",
        gpu="T4"
    ).add_local_dir(".", remote_path="/root/hloc-glomap")
)

# Could add a volume but not sure what I would be using it for
# volumes={"/root/workspace": modal.Volume.from_name("modal-server", create_if_missing=True)}
@app.function(timeout=3600, volumes={"/root/hloc-glomap/data": modal.Volume.from_name("data", create_if_missing=True)}, gpu="T4")
def run_reconstruction(dataset: str, feature: str, matcher: str, mapper: str):
    subprocess.run("git pull", shell=True, check=True)

    subprocess.run(f"gcloud storage rsync -r gs://tour_storage/data/{dataset} data/{dataset}", shell=True, check=True)

    images_dir = Path(f"data/{dataset}/images")
    reconstruction_cmd = [
        "pixi run python tools/reconstruct.py",
        f"--image-dir {images_dir}",
        f"--feature-type {feature}",
        f"--matcher-type {matcher}",
        f"--mapper-cmd {mapper}",
    ]
    reconstruction_cmd = " ".join(reconstruction_cmd)
    subprocess.run(reconstruction_cmd, shell=True, check=True)

    subprocess.run(f"gcloud storage rsync -r data/{dataset} gs://tour_storage/data/{dataset}", shell=True, check=True)

@app.local_entrypoint()
def main(dataset: str):
    """dataset should be like 'tandt/truck' for example"""
    # Run glomap with a few different feature matchers

    feature_matcher_pairs = [
        ("disk", "disk+lightglue"),
        ("aliked-n16", "aliked+lightglue"),
    ]

    for feature, matcher in feature_matcher_pairs:
        # run_reconstruction.remote(dataset, feature, matcher, "glomap")
        run_reconstruction.remote(dataset, feature, matcher, "colmap")