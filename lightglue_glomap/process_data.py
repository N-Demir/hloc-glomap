from pathlib import Path
from typing import Literal
from lightglue_glomap import pairs_from_sequential
from lightglue_glomap.colmap_utils import CameraModel, colmap_to_json
from lightglue_glomap.scripts import CONSOLE, run_command, status
from hloc.reconstruction import create_empty_db, import_images, get_image_ids
from hloc.triangulation import (
    import_features,
    import_matches,
    estimation_and_geometric_verification,
)
from timeit import default_timer as timer


def run_lightglue_glomap(
    image_dir: Path,
    colmap_dir: Path,
    camera_model: CameraModel = CameraModel.OPENCV,
    verbose: bool = False,
    matching_method: Literal["vocab_tree", "exhaustive", "sequential"] = "vocab_tree",
    feature_type: Literal[
        "sift",
        "superpoint_aachen",
        "disk",
    ] = "superpoint_aachen",
    matcher_type: Literal[
        "superglue",
        "NN-ratio",
        "NN-mutual",
        "disk+lightglue",
    ] = "superglue",
    num_matched: int = 50,
    use_single_camera_mode: bool = True,
) -> None:
    """Runs hloc on the images.

    Args:
        image_dir: Path to the directory containing the images.
        colmap_dir: Path to the output directory.
        camera_model: Camera model to use.
        verbose: If True, logs the output of the command.
        matching_method: Method to use for matching images.
        feature_type: Type of visual features to use.
        matcher_type: Type of feature matcher to use.
        num_matched: Number of image pairs for loc.
        use_single_camera_mode: If True, uses one camera for all frames. Otherwise uses one camera per frame.
    """

    import pycolmap
    from hloc import (  # type: ignore
        extract_features,
        match_features,
        pairs_from_exhaustive,
        pairs_from_retrieval,
    )

    outputs = colmap_dir
    sfm_pairs = outputs / "pairs-netvlad.txt"
    sfm_dir = outputs / "sparse" / "0"
    features = outputs / "features.h5"
    matches = outputs / "matches.h5"

    retrieval_conf = extract_features.confs["netvlad"]  # type: ignore
    feature_conf = extract_features.confs[feature_type]  # type: ignore
    matcher_conf = match_features.confs[matcher_type]  # type: ignore

    references = [p.relative_to(image_dir).as_posix() for p in image_dir.iterdir()]

    start_time = timer()

    extract_features.main(
        feature_conf, image_dir, image_list=references, feature_path=features
    )  # type: ignore
    if matching_method == "exhaustive":
        pairs_from_exhaustive.main(sfm_pairs, image_list=references)  # type: ignore
    elif matching_method == "sequential":
        pairs_from_sequential.main(
            output=sfm_pairs, image_list=references, features=features, overlap=10
        )  # type: ignore
    else:
        retrieval_path = extract_features.main(retrieval_conf, image_dir, outputs)  # type: ignore
        num_matched = min(len(references), num_matched)
        pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=num_matched)  # type: ignore
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)  # type: ignore

    image_options = pycolmap.ImageReaderOptions(camera_model=camera_model.value)  # type: ignore

    if use_single_camera_mode:  # one camera per all frames
        camera_mode = pycolmap.CameraMode.SINGLE  # type: ignore
    else:  # one camera per frame
        camera_mode = pycolmap.CameraMode.PER_IMAGE  # type: ignore

    assert features.exists(), features
    assert sfm_pairs.exists(), sfm_pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / "database.db"

    create_empty_db(database)
    import_images(
        image_dir, database, camera_mode, image_list=None, options=image_options
    )
    image_ids = get_image_ids(database)
    import_features(image_ids, database, features)
    import_matches(
        image_ids,
        database,
        sfm_pairs,
        matches,
        min_match_score=None,
        skip_geometric_verification=False,
    )
    estimation_and_geometric_verification(database, sfm_pairs, verbose)

    colmap_cmd = "glomap"
    # Bundle adjustment
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    mapper_cmd = [
        f"{colmap_cmd} mapper",
        f"--database_path {database}",
        f"--image_path {image_dir}",
        f"--output_path {sparse_dir}",
    ]

    mapper_cmd = " ".join(mapper_cmd)

    with status(
        msg=f"[bold yellow]Running {colmap_cmd} bundle adjustment... (This may take a while)",
        spinner="circle",
        verbose=verbose,
    ):
        run_command(mapper_cmd, verbose=verbose)

    colmap_to_json(
        recon_dir=sfm_dir,
        output_dir=colmap_dir.parent,
    )

    end_time = timer()
    CONSOLE.log(
        f"[bold green]:tada: Done {colmap_cmd} bundle adjustment. Time taken: {end_time - start_time:.2f} seconds."
    )
