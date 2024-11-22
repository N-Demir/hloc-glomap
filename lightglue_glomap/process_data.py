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
from hloc.utils.io import get_keypoints, get_matches
from tqdm import tqdm
from timeit import default_timer as timer

import rerun as rr
import cv2
import numpy as np


def log_features(
    bgr_dict: dict[str, np.ndarray],
    features: Path,
    matching_method: Literal["vocab_tree", "exhaustive", "sequential"],
    max_dim: int = 640,
) -> None:
    # log keypoints
    for idx, (image_name, bgr) in enumerate(
        tqdm(bgr_dict.items(), desc="Logging keypoints")
    ):
        rr.set_time_sequence("features", idx)
        cam_log_path: str = (
            "camera" if matching_method == "sequential" else f"{image_name}"
        )
        keypoints = get_keypoints(features, image_name)

        # reshape bgr and keypoints to vga like resolution max dim 640
        scale = max_dim / max(bgr.shape[:2])
        bgr = cv2.resize(bgr, (0, 0), fx=scale, fy=scale)
        keypoints = keypoints * scale

        rr.log(
            f"{cam_log_path}/image",
            rr.Image(bgr, color_model=rr.ColorModel.BGR).compress(jpeg_quality=50),
        )
        rr.log(f"{cam_log_path}/image/keypoints", rr.Points2D(positions=keypoints))


def log_matches(
    bgr_dict: dict[str, np.ndarray],
    pairs_path: Path,
    matches_path: Path,
    features_path: Path,
    image_ids: dict[str, int],
) -> None:
    with open(str(pairs_path), "r") as f:
        pairs = [p.split() for p in f.readlines()]

    matched = set()
    cam_log_path = "camera"
    # matched_images = []

    rr.reset_time()  # Clears all set timeline info.
    for idx, (name0, name1) in enumerate(
        tqdm(pairs, desc="Logging matches", total=len(pairs))
    ):
        id0, id1 = image_ids[name0], image_ids[name1]
        if len({(id0, id1), (id1, id0)} & matched) > 0:
            continue
        rr.set_time_sequence("matches", idx)
        # load matches
        matches, scores = get_matches(matches_path, name0, name1)

        # load keypoints
        keypoints0 = get_keypoints(features_path, name0)[matches[:, 0]]
        keypoints1 = get_keypoints(features_path, name1)[matches[:, 1]]
        bgr0 = bgr_dict[name0]
        bgr1 = bgr_dict[name1]
        h0, w0, _ = bgr0.shape
        h1, w1, _ = bgr1.shape
        # merge images horizontally
        bgr = cv2.hconcat([bgr0, bgr1])
        keypoints1[:, 0] += w0

        # resize to vga like resolution max dim 640
        max_dim = 640
        scale = max_dim / max(h0, w0, h1, w1)
        bgr = cv2.resize(bgr, (0, 0), fx=scale, fy=scale)
        keypoints0 *= scale
        keypoints1 *= scale

        line_segments = np.stack([keypoints0, keypoints1], axis=1)  # Shape: (n, 2, 2)
        # matched_images.append(bgr)

        # draw matches
        rr.log(
            f"{cam_log_path}/image",
            rr.Image(bgr, color_model=rr.ColorModel.BGR).compress(jpeg_quality=25),
        )
        rr.log(f"{cam_log_path}/image/keypoints0", rr.Points2D(positions=keypoints0))
        rr.log(f"{cam_log_path}/image/keypoints1", rr.Points2D(positions=keypoints1))
        rr.log(
            f"{cam_log_path}/image/matches",
            rr.LineStrips2D(strips=line_segments, colors=(0, 255, 0), radii=0.25),
        )

        matched |= {(id0, id1), (id1, id0)}
    # only sends uncompressed images
    # send_data_columns(matched_images=matched_images, cam_log_path=cam_log_path)


def send_data_columns(matched_images: np.ndarray, cam_log_path: str) -> None:
    # Timeline on which the images are distributed.
    start = timer()
    matched_images = np.stack(matched_images)

    sequence = np.arange(0, len(matched_images))
    # Log the ImageFormat and indicator once, as static.

    format_static = rr.components.ImageFormat(
        width=matched_images.shape[2],
        height=matched_images.shape[1],
        color_model="BGR",
        channel_datatype="U8",
    )
    rr.log(
        f"{cam_log_path}/image",
        [format_static, rr.Image.indicator()],
        static=True,
    )

    # Send all images at once.
    rr.send_columns(
        f"{cam_log_path}/image",
        times=[rr.TimeSequenceColumn("matches", sequence)],
        # Reshape the images so `ImageBufferBatch` can tell that this is several blobs.
        #
        # Note that the `ImageBufferBatch` consumes arrays of bytes,
        # so if you have a different channel datatype than `U8`, you need to make sure
        # that the data is converted to arrays of bytes before passing it to `ImageBufferBatch`.
        components=[
            rr.components.ImageBufferBatch(matched_images.reshape(len(sequence), -1))
        ],
    )

    print(f"Time taken to log matches: {timer() - start:.2f} seconds")


def run_lightglue_glomap(
    image_dir: Path,
    colmap_dir: Path,
    camera_model: CameraModel = CameraModel.OPENCV,
    verbose: bool = False,
    matching_method: Literal["vocab_tree", "exhaustive", "sequential"] = "vocab_tree",
    feature_type: Literal[
        "sift", "superpoint_aachen", "disk", "xfeat"
    ] = "superpoint_aachen",
    matcher_type: Literal[
        "superglue",
        "NN-ratio",
        "NN-mutual",
        "disk+lightglue",
        "superpoint+lightglue",
    ] = "superglue",
    num_matched: int = 50,
    use_single_camera_mode: bool = True,
    colmap_cmd: Literal["colmap", "glomap"] = "glomap",
) -> None:
    """Runs hloc on the images.

    Args:
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

    # if colmap_dir.exists() and colmap_dir.name == "test":
    #     print("Removing existing colmap_dir")
    #     import shutil

    #     shutil.rmtree(colmap_dir)

    outputs = colmap_dir
    sfm_pairs = outputs / "pairs-netvlad.txt"
    sfm_dir = outputs / "sparse" / "0"
    features = outputs / "features.h5"
    matches = outputs / "matches.h5"

    # load bgr images into a dict for logging
    bgr_dict: dict[str, np.ndarray] = {}
    for img_path in sorted(image_dir.iterdir()):
        bgr_dict[img_path.name] = cv2.imread(str(img_path))

    retrieval_conf = extract_features.confs["netvlad"]  # type: ignore
    feature_conf = extract_features.confs[feature_type]  # type: ignore
    matcher_conf = match_features.confs[matcher_type]  # type: ignore

    references: list[str] = sorted(
        [p.relative_to(image_dir).as_posix() for p in image_dir.iterdir()]
    )

    start_time: float = timer()

    extract_features.main(
        feature_conf, image_dir, image_list=references, feature_path=features
    )  # type: ignore

    log_features(bgr_dict=bgr_dict, features=features, matching_method=matching_method)

    if matching_method == "exhaustive":
        pairs_from_exhaustive.main(sfm_pairs, image_list=references)  # type: ignore
    elif matching_method == "sequential":
        pairs_from_sequential.main(
            output=sfm_pairs,
            image_list=references,
            features=features,
            overlap=10,
            quadratic_overlap=False,
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
    image_ids: dict[str, int] = get_image_ids(database)

    log_matches(
        bgr_dict=bgr_dict,
        pairs_path=sfm_pairs,
        matches_path=matches,
        features_path=features,
        image_ids=image_ids,
    )

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

    # Bundle adjustment
    num_images: int = len(list(image_dir.glob("*")))
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    mapper_cmd = [
        f"{colmap_cmd} mapper",
        f"--database_path {database}",
        f"--image_path {image_dir}",
        f"--output_path {sparse_dir}",
    ]
    if colmap_cmd == "glomap":
        mapper_cmd += [
            f"--TrackEstablishment.max_num_tracks {2000*num_images}",
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
