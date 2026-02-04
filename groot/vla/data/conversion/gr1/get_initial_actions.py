from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import groot.vla.common.utils as U
from groot.vla.data.conversion.gr1.constants import (
    INITIAL_ACTIONS_FILENAME,
    TRAINABLE_HDF5_FILENAME,
)
from groot.vla.data.dataset.macro import (
    LE_ROBOT_EPISODE_FILENAME,
    LE_ROBOT_INFO_FILENAME,
    LE_ROBOT_METADATA_DIR,
    LE_ROBOT_MODALITY_FILENAME,
)


def get_initial_actions(data_dir: str | Path):
    hdf5_file = h5py.File(Path(data_dir) / TRAINABLE_HDF5_FILENAME, "r")
    initial_actions = []

    """
    initial_actions: dict[str, dict[str, np.ndarray]]
    0: (the dataset dimension)
        trajectory_name:
          action_key:
            action: np.ndarray
    """
    initial_actions = {}
    for demo_name in hdf5_file["data"].keys():
        demo_group = hdf5_file["data"][demo_name]
        initial_actions[demo_name] = {}
        action_keys = list(demo_group["action"].keys())
        for action_key in action_keys:
            initial_actions[demo_name][action_key] = demo_group["action"][action_key][0]
    return [initial_actions]


def get_initial_actions_from_lerobot(data_dir: str | Path):
    data_dir = Path(data_dir)

    # 1. Get modality for slicing action
    meta_modality_path = data_dir / LE_ROBOT_METADATA_DIR / LE_ROBOT_MODALITY_FILENAME
    meta_modality = U.load_json(meta_modality_path)
    action_keys = meta_modality["action"].keys()

    # 2. Get episode paths
    # 2.1. Get data_path_pattern
    meta_info_path = data_dir / LE_ROBOT_METADATA_DIR / LE_ROBOT_INFO_FILENAME
    meta_info = U.load_json(meta_info_path)
    data_path_pattern = meta_info["data_path"]
    chunk_size = meta_info["chunks_size"]

    # 2.2. Get episode info
    episode_metadata_path = data_dir / LE_ROBOT_METADATA_DIR / LE_ROBOT_EPISODE_FILENAME
    episode_metadata = U.load_jsonl(episode_metadata_path)

    initial_actions = {}
    for episode_info in episode_metadata:
        episode_index = episode_info["episode_index"]
        episode_chunk = episode_index // chunk_size
        episode_path = data_dir / data_path_pattern.format(
            episode_chunk=episode_chunk, episode_index=episode_index
        )
        if not episode_path.exists():
            raise ValueError(f"Episode path {episode_path} does not exist")

        episode_data = pd.read_parquet(episode_path)

        initial_action_concat = episode_data["action"].iloc[0]
        trajectory_id = episode_info["episode_index"]
        initial_actions[trajectory_id] = {}
        for action_key in action_keys:
            start = meta_modality["action"][action_key]["start"]
            end = meta_modality["action"][action_key]["end"]
            initial_actions[trajectory_id][action_key] = initial_action_concat[start:end]
    return [initial_actions]


def save_initial_actions(
    initial_actions: dict[str, dict[str, np.ndarray]], initial_actions_path: str | Path
):
    np.savez(str(initial_actions_path), initial_actions)


def load_initial_actions(initial_actions_path: str | Path):
    """
    initial_actions: list[dict[str, dict[str, np.ndarray]]]
    0: (the first dataset)
        trajectory_name:
          action_key:
            action: np.ndarray
    1: (the second dataset)
        ...
    """
    initial_actions_npz = np.load(str(initial_actions_path), allow_pickle=True)
    initial_actions = []
    initial_actions_array = initial_actions_npz[
        "arr_0"
    ]  # This is the default key when np.savez saves a list
    for dataset_initial_actions in initial_actions_array:
        initial_actions_for_this_dataset = {}
        for trajectory_name, action_dict in dataset_initial_actions.items():
            initial_actions_for_this_dataset[trajectory_name] = action_dict
        initial_actions.append(initial_actions_for_this_dataset)
    return initial_actions


if __name__ == "__main__":
    data_dirs = [
        # "/mnt/amlfs-03/shared/datasets/gr1/real_gr1_arms_only_v1/datasets/real_gr1_arms_only_annotated:gr00t004_1dragonfruit1plate0distractor_Res256Freq20",
        # "/mnt/amlfs-03/shared/datasets/gr1/real_gr1_arms_only_v1/datasets/real_gr1_arms_only_annotated:gr00t004_1dragonfruit1plate2distractor_Res256Freq20",
        # "/mnt/amlfs-03/shared/datasets/gr1/real_gr1_arms_only_v1/datasets/real_gr1_arms_only_annotated:gr00t006_1apple1shelf0distractor_Res256Freq20",
        # "/mnt/amlfs-03/shared/datasets/gr1/real_gr1_arms_only_v1/datasets/real_gr1_arms_only_annotated:gr00t006_1cube1basket0distractor_Res256Freq20",
        # "/mnt/amlfs-03/shared/datasets/gr1/real_gr1_arms_only_v1/datasets/real_gr1_arms_only_annotated:gr00t006_1cup1plate0distractor_Res256Freq20",
        # "/mnt/amlfs-03/shared/datasets/gr1/real_gr1_arms_waist_v1/datasets/real_gr1_arms_waist_annotated:gr00t006_waist_1apple1shelf0distractor_Res256Freq20",
    ]

    # for data_dir in data_dirs:
    #     initial_actions = get_initial_actions(data_dir=data_dir)
    #     save_initial_actions(initial_actions, Path(data_dir) / INITIAL_ACTIONS_FILENAME)

    #     lerobot_dir = data_dir.replace("_v1/datasets", "_v1_lerobot").replace(":", ".")
    #     lerobot_meta_dir = Path(lerobot_dir) / "meta"

    #     if not Path(lerobot_meta_dir).exists():
    #         raise ValueError(f"Lerobot directory {lerobot_meta_dir} does not exist")

    #     save_initial_actions(initial_actions, lerobot_meta_dir / INITIAL_ACTIONS_FILENAME)

    #     # Test loading
    #     loaded_initial_actions = load_initial_actions(Path(data_dir) / INITIAL_ACTIONS_FILENAME)

    lerobot_data_dirs = [
        # "/mnt/amlfs-03/shared/datasets/gr1/gr1_unified_v1/0303/gr1_unified.pnp_remove_static_threshold1_Freq20_new_pipeline",
        # "/mnt/amlfs-03/shared/datasets/gr1/gr1_unified_v1/0303/gr1_unified.industrial_remove_static_threshold1_BGCropRes256Freq20",
        "/mnt/amlfs-03/shared/datasets/gr1/gr1_unified_v1/0302/gr1_unified.pointing_remove_static_threshold1_BGCropRes256Freq20",
    ]

    for lerobot_data_dir in lerobot_data_dirs:
        initial_actions = get_initial_actions_from_lerobot(lerobot_data_dir)
        save_initial_actions(
            initial_actions,
            Path(lerobot_data_dir) / LE_ROBOT_METADATA_DIR / INITIAL_ACTIONS_FILENAME,
        )

        # Test loading
        loaded_initial_actions = load_initial_actions(
            Path(lerobot_data_dir) / LE_ROBOT_METADATA_DIR / INITIAL_ACTIONS_FILENAME
        )

        # loaded_example_initial_actions = load_initial_actions(
        #     "/mnt/amlfs-03/shared/datasets/gr1/gr1_unified_v1/0228/5DC-S_nowaistprefix/gr1_unified.Slice_cutting_board_to_pan_oldsim_dummylang_0222_BGCropRes256Freq20/meta/initial_actions.npz"
        # )

        # print(list(loaded_initial_actions[0].keys())[0])
        # print(list(loaded_example_initial_actions[0].keys())[0])
        # breakpoint()
