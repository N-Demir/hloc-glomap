from pathlib import Path
from dataclasses import dataclass, field
from hloc_glomap.scripts import run_command, status
import tyro
from enum import Enum


class MapperType(Enum):
    GLOMAP = "glomap"
    COLMAP = "colmap"


class FeatureMatcherPair(Enum):
    DISK = ("disk", "disk+lightglue")
    ALIKED = ("aliked-n16", "aliked+lightglue")
    XFEAT = ("xfeat", "xfeat+lighterglue")

    @property
    def feature(self) -> str:
        return self.value[0]

    @property
    def matcher(self) -> str:
        return self.value[1]


@dataclass
class BenchmarkConfig:
    video_dir: Path
    """Path to the image directory"""
    num_frames_to_extract: list[int] = field(default_factory=lambda: [100, 250])
    """Number of frames to extract from the video"""
    feature_matcher_pairs: list[FeatureMatcherPair] = field(
        default_factory=lambda: [
            FeatureMatcherPair.DISK,
            FeatureMatcherPair.ALIKED,
            FeatureMatcherPair.XFEAT,
        ]
    )
    """Feature and matcher pairs to use for reconstruction"""
    mapper_types: list[MapperType] = field(default_factory=lambda: [MapperType.GLOMAP])


def main(config: BenchmarkConfig) -> None:
    verbose = True
    output_dir_name = config.video_dir.stem.lower().replace(" ", "")
    output_parent_dir = config.video_dir.parent / output_dir_name
    rrd_dir = output_parent_dir / "rrd"
    # Create RRD directory first
    rrd_dir.mkdir(parents=True, exist_ok=True)
    output_dirs: list[Path] = []
    # extract frames from videos at different num_frames_target
    for num_frames_target in config.num_frames_to_extract:
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

    for output_dir in output_dirs:
        for pair in config.feature_matcher_pairs:
            for mapper_type in config.mapper_types:
                reconstruction_cmd = [
                    "python tools/reconstruct.py",
                    f"--image-dir {output_dir}/images",
                    f"--feature-type {pair.feature}",
                    f"--matcher-type {pair.matcher}",
                    f"--colmap-cmd {mapper_type.value}",
                    f"--rerun-config.save {rrd_dir / f'{output_dir.stem}-{pair.feature}-{pair.matcher}-{mapper_type.value}.rrd'}",
                ]
                reconstruction_cmd = " ".join(reconstruction_cmd)
                with status(
                    msg=f"[bold yellow]Running reconstruction with {pair.feature}/{pair.matcher}/{mapper_type.value}[/]",
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
