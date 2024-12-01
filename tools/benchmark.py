from pathlib import Path
from dataclasses import dataclass
from hloc_glomap.scripts import run_command, status
import tyro


@dataclass
class BenchmarkConfig:
    video_dir: Path
    """Path to the image directory"""


def main(config: BenchmarkConfig) -> None:
    verbose = True
    output_dir_name = config.video_dir.stem.lower().replace(" ", "")
    output_parent_dir = config.video_dir.parent / output_dir_name
    rrd_dir = output_parent_dir / "rrd"
    # Create RRD directory first
    rrd_dir.mkdir(parents=True, exist_ok=True)
    output_dirs: list[Path] = []
    # extract frames from videos at different num_frames_target
    for num_frames_target in [100, 250, 500]:
        # create a bunch of rrds
        output_dir = output_parent_dir / f"{num_frames_target}"
        output_dirs.append(output_dir)
        # check if the output dir exists
        if output_dir.exists():
            print(f"Output dir {output_dir} already exists, skipping")
            continue
        video_cmd = [
            "pixi run video-processing",  # noqa 541
            f"--data '{config.video_dir}'",
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

    # once we have all the frames, we can run the reconstruction on each on with different settings
    # Define feature-matcher pairs
    feature_matcher_pairs = [
        ("disk", "disk+lightglue"),
        ("aliked-n16", "aliked+lightglue"),
        ("xfeat", "xfeat+lighterglue"),
    ]
    mapper_types = [
        "glomap",
        # "colmap",
    ]
    for output_dir in output_dirs:
        for feature_type, matcher_type in feature_matcher_pairs:
            for mapper_type in mapper_types:
                reconstruction_cmd = [
                    "python tools/reconstruct.py",
                    f"--image-dir {output_dir}/images",
                    f"--feature-type {feature_type}",
                    f"--matcher-type {matcher_type}",
                    f"--colmap-cmd {mapper_type}",
                    f"--rerun-config.save {rrd_dir / f'{output_dir.stem}-{feature_type}-{matcher_type}-{mapper_type}.rrd'}",
                ]
                reconstruction_cmd = " ".join(reconstruction_cmd)
                with status(
                    msg=f"[bold yellow]Running reconstruction with {feature_type}/{matcher_type}/{mapper_type}[/]",
                    spinner="circle",
                    verbose=verbose,
                ):
                    run_command(reconstruction_cmd, verbose=verbose)


if __name__ == "__main__":
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            BenchmarkConfig,
            description="Process raw capture into processed one",
        )
    )
