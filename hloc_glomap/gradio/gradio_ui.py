import os
import tempfile

import gradio as gr
from gradio_rerun import Rerun
from typing import Literal

import rerun as rr
import rerun.blueprint as rrb

from hloc_glomap.process_data import run_hloc_reconstruction
from hloc_glomap.scripts import status, run_command
from pathlib import Path


def cleanup_rrds(pending_cleanup: list[str]) -> None:
    for f in pending_cleanup:
        os.unlink(f)


@rr.thread_local_stream("video_input")
def show_video_rrd(video_file_path: str, pending_cleanup: list[str]) -> str:
    # Log video asset which is referred to by frame references.
    video_asset = rr.AssetVideo(path=video_file_path)
    rr.log("video", video_asset, static=True)

    # Send automatically determined video frame timestamps.
    frame_timestamps_ns = video_asset.read_frame_timestamps_ns()
    rr.send_columns(
        "video",
        # Note timeline values don't have to be the same as the video timestamps.
        times=[rr.TimeNanosColumn("video_time", frame_timestamps_ns)],
        components=[
            rr.VideoFrameReference.indicator(),
            rr.components.VideoTimestamp.nanoseconds(frame_timestamps_ns),
        ],
    )

    # We eventually want to clean up the RRD file after it's sent to the viewer, so tracking
    # any pending files to be cleaned up when the state is deleted.
    temp = tempfile.NamedTemporaryFile(prefix="video_", suffix=".rrd", delete=False)
    pending_cleanup.append(temp.name)

    # blueprint = rrb.Spatial3DView(origin="cube")
    rr.save(temp.name)

    # Just return the name of the file -- Gradio will convert it to a FileData object
    # and send it to the viewer.
    return temp.name


def extract_frames(
    output_dir: Path,
    video_file_path: Path,
    num_frames_target: int,
    verbose: bool = False,
) -> None:
    image_dir = output_dir / "images"
    if image_dir.exists():
        print(f"Output dir {output_dir} already exists, skipping")
        return
    video_cmd = [
        "pixi run video-processing",  # noqa 541
        f"--data '{video_file_path}'",
        f"--output-dir {output_dir}",
        f"--num-frames-target {num_frames_target}",
    ]

    video_cmd = " ".join(video_cmd)
    with status(
        msg="[bold yellow]Running video-processing[/]",
        spinner="circle",
        verbose=verbose,
    ):
        run_command(
            video_cmd,
            verbose=verbose,
        )


@rr.thread_local_stream("reconstruction")
def reconstruction_fn(
    video_file_path: str,
    num_frames_target: int,
    mapper_cmd: Literal["colmap", "glomap"],
    matching_method: Literal["sequential", "vocab_tree", "exhaustive"],
    feature_type: Literal["disk", "xfeat", "aliked-n16"],
    matcher_type: Literal["disk+lightglue", "xfeat+lighterglue", "aliked+lightglue"],
    pending_cleanup: list[str],
    progress=gr.Progress(track_tqdm=True),
):
    # Extract image frames from video file.
    progress(progress=0.1, desc="Extracting frames from video...")
    video_file_path: Path = Path(video_file_path)
    assert video_file_path.exists(), f"Video file {video_file_path} does not exist"
    video_dir = video_file_path.parent

    print(f"Extracting frames from video {video_file_path}...")
    print(f"Parent dir: {video_dir}")

    extract_frames(
        output_dir=video_dir,
        video_file_path=video_file_path,
        num_frames_target=num_frames_target,
    )

    images_dir = video_dir / "images"
    assert images_dir.exists(), f"Images dir {images_dir} does not exist"
    progress(progress=0.15, desc="Running reconstruction... (this may take a while)")
    run_hloc_reconstruction(
        image_dir=images_dir,
        colmap_dir=video_dir,
        matching_method=matching_method,
        feature_type=feature_type,
        matcher_type=matcher_type,
        colmap_cmd=mapper_cmd,
    )
    temp = tempfile.NamedTemporaryFile(
        prefix="reconstruction_", suffix=".rrd", delete=False
    )
    pending_cleanup.append(temp.name)

    blueprint = rrb.Horizontal(
        rrb.Spatial3DView(origin="/"),
        rrb.Vertical(
            rrb.Spatial2DView(origin="/camera/image"),
            rrb.Horizontal(
                rrb.TextDocumentView(origin="/logs"), rrb.TimeSeriesView(origin="/plot")
            ),
        ),
    )
    rr.save(temp.name, default_blueprint=blueprint)

    return temp.name


with gr.Blocks() as demo_block:
    pending_cleanup = gr.State([], time_to_live=10, delete_callback=cleanup_rrds)
    with gr.Row():
        video_file = gr.File(file_count="single", label="Video file")
        run_recon_btn = gr.Button("Reconstruct from Video")
    with gr.Row():
        with gr.Accordion(label="Advanced Options", open=False):
            with gr.Row():
                num_frames_target = gr.Radio(
                    choices=[50, 100, 300, 500],
                    value=50,
                    label="Number of frames to extract",
                )
                mapper_cmd = gr.Radio(
                    choices=["colmap", "glomap"], value="glomap", label="Mapper command"
                )
                matching_method = gr.Radio(
                    choices=["sequential", "vocab_tree", "exhaustive"],
                    value="sequential",
                    label="Matching method",
                )
                feature_type = gr.Radio(
                    choices=["disk", "xfeat", "aliked-n16"],
                    value="aliked-n16",
                    label="Feature type",
                )
                matcher_type = gr.Radio(
                    choices=["aliked+lightglue", "xfeat+lighterglue", "disk+lightglue"],
                    value="aliked+lightglue",
                    label="Matcher type",
                )

    with gr.Row():
        viewer = Rerun(
            streaming=True,
            panel_states={
                "time": "collapsed",
                "blueprint": "hidden",
                "selection": "hidden",
            },
            height=800,
        )

    video_file.upload(
        fn=show_video_rrd, inputs=[video_file, pending_cleanup], outputs=[viewer]
    )

    run_recon_btn.click(
        reconstruction_fn,
        inputs=[
            video_file,
            num_frames_target,
            mapper_cmd,
            matching_method,
            feature_type,
            matcher_type,
            pending_cleanup,
        ],
        outputs=[viewer],
    )

    # gr.Examples(
    #     examples=[
    #         [
    #             "examples/example-vid-vga.mp4",
    #             50,
    #             "glomap",
    #             "sequential",
    #             "aliked-n16",
    #             "aliked+lightglue",
    #             [],
    #         ]
    #     ],
    #     fn=reconstruction_fn,
    #     inputs=[
    #         video_file,
    #         num_frames_target,
    #         mapper_cmd,
    #         matching_method,
    #         feature_type,
    #         matcher_type,
    #         pending_cleanup,
    #     ],
    #     outputs=[viewer],
    #     cache_examples="lazy",
    # )
