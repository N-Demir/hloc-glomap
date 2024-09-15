import gradio as gr
from pathlib import Path
from lightglue_glomap.scripts import run_command, CONSOLE
from lightglue_glomap.process_data import run_lightglue_glomap


def run_reconstruction_fn(
    input_zip_file: str,
    password: str | None = None,
    progress=gr.Progress(track_tqdm=True),
) -> None:
    zip_path: Path = Path(input_zip_file)
    assert zip_path.exists(), zip_path
    unzip_cmd: list[str] = [  # noqa: E501
        f"unzip {zip_path}",
        f"-d {zip_path.parent}",
    ]
    if password:
        unzip_cmd: list[str] = [  # noqa: E501
            f"unzip -P {password} {zip_path}",
            f"-d {zip_path.parent}",
        ]
    unzip_cmd: str = " ".join(unzip_cmd)
    CONSOLE.print(f"Running command: {unzip_cmd}")
    run_command(cmd=unzip_cmd, verbose=True)
    CONSOLE.print(f"Unzipped {zip_path} to {zip_path.parent}")
    input_dir: Path = zip_path.parent / zip_path.stem
    run_lightglue_glomap(
        image_dir=input_dir / "images", colmap_dir=input_dir / "glomap"
    )


with gr.Blocks() as process_block:
    input_file = gr.File(file_count="single", file_types=["zip"], type="filepath")
    with gr.Accordion(label="Advanced Options", open=False):
        password = gr.Textbox(label="Password", type="password")

    run_btn = gr.Button(value="Run Reconstruction")
    run_btn.click(
        fn=run_reconstruction_fn,
        inputs=(input_file, password),
        outputs=None,
    )
