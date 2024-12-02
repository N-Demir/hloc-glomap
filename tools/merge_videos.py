import mmcv
import tyro
from pathlib import Path
from dataclasses import dataclass
from icecream import ic


@dataclass
class MergeVideosConfig:
    videos_parent_dir: Path
    """Path to the parent directory containing the videos"""


def main(config: MergeVideosConfig) -> None:
    assert (
        config.videos_parent_dir.exists()
    ), f"Directory {config.videos_parent_dir} does not exist"
    assert (
        config.videos_parent_dir.is_dir()
    ), f"{config.videos_parent_dir} is not a directory"
    # Get sorted list of videos and properly escape spaces in paths
    videos_list = sorted(
        [
            str(video_path).replace(" ", r"\ ")
            for video_path in config.videos_parent_dir.glob("*.mp4")
        ]
    )

    ic(videos_list)
    mmcv.concat_video(
        video_list=videos_list, out_file=f"{config.videos_parent_dir}/merged.mp4"
    )
    print(f"Merged videos saved to {config.videos_parent_dir}/merged.mp4")


if __name__ == "__main__":
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            MergeVideosConfig,
            description="Merge all videos in a directory into a single video",
        )
    )
