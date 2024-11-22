from pathlib import Path
from lightglue_glomap.process_data import run_lightglue_glomap
from lightglue_glomap.colmap_utils import CameraModel
from argparse import ArgumentParser
import rerun as rr


if __name__ == "__main__":
    name = "reconstruction-lightglue-glomap"
    parser = ArgumentParser(name)
    parser.add_argument("--image-dir", type=Path, required=True)
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, application_id=f"{name}")

    images_dir: Path = args.image_dir
    assert images_dir.exists(), images_dir

    run_lightglue_glomap(
        image_dir=images_dir,
        colmap_dir=images_dir.parent / "test",
        camera_model=CameraModel.OPENCV,
        verbose=True,
        matching_method="sequential",
        feature_type="xfeat",
        matcher_type="NN-mutual",
        num_matched=50,
        use_single_camera_mode=True,
        colmap_cmd="glomap",
    )
    rr.script_teardown(args)
