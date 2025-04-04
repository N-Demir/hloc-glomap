from pathlib import Path
from hloc_glomap.process_data import run_hloc_reconstruction
from hloc_glomap.colmap_utils import CameraModel
import rerun as rr
from rerun.recording_stream import RecordingStream
from dataclasses import dataclass
from typing import Literal
import tyro


@dataclass
class RerunConfig:
    application: str = "reconstruction-hloc-glomap"
    """Name of the application"""
    headless: bool = True
    """Wether to spawn a new rerun instance or not"""
    connect: bool = False
    """Wether to connect to an existing rerun instance or not"""
    save: Path | None = None
    """Path to save the rerun data, this will make it so no data is visualized but saved"""

    def __post_init__(self):
        rr.init(application_id=self.application)
        # rr.serve_web(browser=False)

        rec: RecordingStream = rr.get_global_data_recording()  # type: ignore[assignment]

        if self.save is not None:
            rec.save(self.save)
        elif self.connect:
            rec.connect()
        elif not self.headless:
            rec.spawn()


@dataclass
class ReconstructionConfig:
    image_dir: Path
    """Path to the image directory"""
    rerun_config: RerunConfig
    """Rerun configuration"""
    camera_model: CameraModel = CameraModel.OPENCV
    """Camera model to use"""
    verbose: bool = False
    """Verbose output"""
    matching_method: Literal["vocab_tree", "exhaustive", "sequential"] = "sequential"
    """Matching method"""
    feature_type: Literal[
        "sift", "superpoint_aachen", "disk", "xfeat", "aliked-n16"
    ] = "aliked-n16"
    """Feature type"""
    matcher_type: Literal[
        "superglue",
        "NN-ratio",
        "NN-mutual",
        "disk+lightglue",
        "superpoint+lightglue",
        "aliked+lightglue",
        "xfeat+lighterglue",
    ] = "aliked+lightglue"
    """Matcher type"""
    num_matched: int = 50
    """Number of matched features for vocab_tree"""
    use_single_camera_mode: bool = True
    """Use single camera mode"""
    use_loop_closure: bool = True
    """Use loop closure"""
    mapper_cmd: Literal["colmap", "glomap"] = "glomap"
    """Mapper command"""


def main(config: ReconstructionConfig) -> None:
    run_hloc_reconstruction(
        image_dir=config.image_dir,
        sfm_output_dir=Path("sfm_output") / config.image_dir.parent.name
        / f"{config.mapper_cmd}-{config.feature_type}-{config.matcher_type}",
        camera_model=config.camera_model,
        verbose=config.verbose,
        matching_method=config.matching_method,
        feature_type=config.feature_type,
        matcher_type=config.matcher_type,
        num_matched=config.num_matched,
        use_single_camera_mode=config.use_single_camera_mode,
        use_loop_closure=config.use_loop_closure,
        colmap_cmd=config.mapper_cmd,
    )


if __name__ == "__main__":
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            ReconstructionConfig,
            description="Process raw capture into processed one",
        )
    )
