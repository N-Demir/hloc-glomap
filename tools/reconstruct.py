from pathlib import Path
from lightglue_glomap.process_data import run_lightglue_glomap
from lightglue_glomap.colmap_utils import CameraModel
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    args = parser.parse_args()

    images_dir: Path = args.image_dir
    assert images_dir.exists(), images_dir

    run_lightglue_glomap(
        image_dir=images_dir,
        colmap_dir=images_dir.parent / "colmap",
        camera_model=CameraModel.OPENCV,
        verbose=True,
        matching_method="sequential",
        feature_type="superpoint_aachen",
        matcher_type="superglue",
        num_matched=50,
        use_single_camera_mode=True,
    )
