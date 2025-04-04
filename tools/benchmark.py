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
    image_dir: Path
    """Path to the image directory"""
    feature_matcher_pairs: list[FeatureMatcherPair] = field(
        default_factory=lambda: [
            FeatureMatcherPair.DISK,
            FeatureMatcherPair.ALIKED,
            FeatureMatcherPair.XFEAT,
        ]
    )
    """Feature and matcher pairs to use for reconstruction"""
    mapper_types: list[MapperType] = field(default_factory=lambda: [MapperType.GLOMAP])
    """Mapper types to use for reconstruction"""
    run_splatfacto: bool = True
    """Run nerfstudio splatfacto"""


def main(config: BenchmarkConfig) -> None:
    verbose = True
    output_dir_name = config.image_dir.stem.lower().replace(" ", "")
    output_parent_dir = config.image_dir.parent / output_dir_name
    rrd_dir = output_parent_dir / "rrd"
    # Create RRD directory first
    rrd_dir.mkdir(parents=True, exist_ok=True)
    
    # # extract frames from videos at different num_frames_target
    # output_dirs: list[Path] = []
    # for num_frames_target in config.num_frames_to_extract:
    #     # create a bunch of rrds
    #     output_dir = output_parent_dir / f"{num_frames_target}"
    #     output_dirs.append(output_dir)
    #     if output_dir.exists():
    #         print(f"Output dir {output_dir} already exists, skipping")
    #         continue
    #     video_cmd = [
    #         "pixi run video-processing",  # noqa 541
    #         f"--data '{config.video_dir}'",
    #         f"--output-dir {output_dir}",
    #         f"--num-frames-target {num_frames_target}",
    #     ]

    #     video_cmd = " ".join(video_cmd)
    #     with status(
    #         msg="[bold yellow]Running video-processing[/]",
    #         spinner="circle",
    #         verbose=verbose,
    #     ):
    #         run_command(
    #             video_cmd,
    #             verbose=verbose,
    #         )
    
    # TODO: MY own code replacing that
    output_dir = config.image_dir

    colmap_save_dirs: list[Path] = []
    for pair in config.feature_matcher_pairs:
        for mapper_type in config.mapper_types:
            # Run SFM with the given feature/matcher pair and mapper type
            colmap_dir = f"{mapper_type.value}-{pair.feature}-{pair.matcher}"
            colmap_save_dirs.append(output_dir / colmap_dir)
            reconstruction_cmd = [
                "python tools/reconstruct.py",
                f"--image-dir {output_dir}/images",
                f"--feature-type {pair.feature}",
                f"--matcher-type {pair.matcher}",
                f"--mapper-cmd {mapper_type.value}",
                f"--rerun-config.save {rrd_dir / f'{output_dir.stem}-{colmap_dir}.rrd'}",
            ]
            reconstruction_cmd = " ".join(reconstruction_cmd)
            with status(
                msg=f"[bold yellow]Running reconstruction with {pair.feature}/{pair.matcher}/{mapper_type.value}[/]",
                spinner="circle",
                verbose=verbose,
            ):
                run_command(reconstruction_cmd, verbose=verbose)

            if config.run_splatfacto:
                # Perform Gaussian splatting with the given feature/matcher pair and mapper type log to tensorboard
                splatfacto_cmd = [
                    f"DATA_DIR='{output_dir}' COLMAP_DIR='{colmap_dir}/sparse/0' EXP_NAME='{colmap_dir}' pixi run train-colmap-splat",
                ]
                splatfacto_cmd = " ".join(splatfacto_cmd)

                with status(
                    msg=f"[bold yellow]Running splatfacto with {pair.feature}/{pair.matcher}/{mapper_type.value}[/]",
                    spinner="circle",
                    verbose=verbose,
                ):
                    run_command(
                        splatfacto_cmd,
                        verbose=verbose,
                    )


if __name__ == "__main__":
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            BenchmarkConfig,
            description="Process raw capture into processed one",
        )
    )
