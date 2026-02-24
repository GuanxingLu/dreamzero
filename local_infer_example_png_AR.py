#!/usr/bin/env python3
"""Local (no websocket) inference runner for DreamZero with example PNG inputs.

This script follows the same observation schema as socket_test_optimized_AR.py, but
runs inference directly in-process (no client/server).
"""

import argparse
import asyncio
import datetime
import logging
import os
import random
import time
import uuid

import cv2
import numpy as np
import torch
import torch.distributed as dist

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from socket_test_optimized_AR import ARDroidRoboarenaPolicy, WebsocketPolicyServer, init_mesh


logger = logging.getLogger(__name__)


LINGBOT_PROMPT_PRESETS = {
    "franka": "pick bunk",
    "robotwin": "Grab the medium-sized white mug, rotate it, place it on the table, and hook it onto the smooth dark gray rack.",
}

CAMERA_FILES = {
    "cam_high": "observation.images.cam_high.png",
    "cam_left_wrist": "observation.images.cam_left_wrist.png",
    "cam_right_wrist": "observation.images.cam_right_wrist.png",
}


def set_global_determinism(seed: int, enable_deterministic: bool) -> None:
    logger.info("Set global seed=%s deterministic=%s", seed, enable_deterministic)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if enable_deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def _load_png_rgb(path: str, target_hw: tuple[int, int]) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = target_hw
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img.astype(np.uint8)


def _load_preset_images(
    example_root: str,
    preset: str,
    target_hw: tuple[int, int],
) -> dict[str, np.ndarray]:
    preset_dir = os.path.join(example_root, preset)
    if not os.path.isdir(preset_dir):
        raise FileNotFoundError(f"Preset directory not found: {preset_dir}")
    loaded: dict[str, np.ndarray] = {}
    for cam_name, filename in CAMERA_FILES.items():
        path = os.path.join(preset_dir, filename)
        loaded[cam_name] = _load_png_rgb(path, target_hw)
        logger.info("Loaded %s -> %s shape=%s", cam_name, path, loaded[cam_name].shape)
    return loaded


def _build_obs(
    mapped_images: dict[str, np.ndarray],
    prompt: str,
    session_id: str,
    as_chunk: bool,
    frames_per_chunk: int,
) -> dict:
    obs: dict = {}
    for obs_key, image in mapped_images.items():
        if as_chunk:
            obs[obs_key] = np.repeat(image[None, ...], repeats=frames_per_chunk, axis=0)
        else:
            obs[obs_key] = image
    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)
    obs["prompt"] = prompt
    obs["session_id"] = session_id
    return obs


def _log_action(actions: np.ndarray, dt: float, tag: str) -> None:
    assert isinstance(actions, np.ndarray), f"Expected ndarray, got {type(actions)}"
    assert actions.ndim == 2, f"Expected action shape (N, D), got {actions.shape}"
    assert actions.shape[-1] == 8, f"Expected 8 dims (7 joints + 1 gripper), got {actions.shape[-1]}"
    logger.info(
        "[%s] action shape=%s range=[%.5f, %.5f] time=%.3fs",
        tag,
        actions.shape,
        float(actions.min()),
        float(actions.max()),
        dt,
    )


def _resolve_output_dir(model_path: str, index: int, output_dir: str | None) -> str:
    if output_dir:
        return output_dir
    parent_dir = os.path.dirname(model_path)
    checkpoint_name = os.path.basename(model_path)
    date_suffix = datetime.datetime.now().strftime("%Y%m%d")
    return os.path.join(parent_dir, f"real_world_eval_gen_{date_suffix}_{index}", checkpoint_name)


def _run_rank0_local_infer(
    wrapper_policy: ARDroidRoboarenaPolicy,
    args: argparse.Namespace,
    output_dir: str,
) -> None:
    prompt = args.prompt if args.prompt is not None else LINGBOT_PROMPT_PRESETS[args.preset]
    source_images = _load_preset_images(args.example_root, args.preset, (args.height, args.width))
    mapped_images = {
        "observation/exterior_image_0_left": source_images[args.exterior0_source],
        "observation/exterior_image_1_left": source_images[args.exterior1_source],
        "observation/wrist_image_left": source_images[args.wrist_source],
    }

    logger.info(
        "Camera mapping: ext0=%s ext1=%s wrist=%s",
        args.exterior0_source,
        args.exterior1_source,
        args.wrist_source,
    )
    logger.info("Prompt: %s", prompt)

    session_id = str(uuid.uuid4())
    logger.info("Session ID: %s", session_id)

    all_actions: list[np.ndarray] = []

    if not args.skip_initial_single:
        obs = _build_obs(
            mapped_images=mapped_images,
            prompt=prompt,
            session_id=session_id,
            as_chunk=False,
            frames_per_chunk=args.frames_per_chunk,
        )
        t0 = time.time()
        actions = wrapper_policy.infer(obs)
        _log_action(actions, time.time() - t0, "initial_single")
        all_actions.append(actions)

    for i in range(args.num_chunks):
        obs = _build_obs(
            mapped_images=mapped_images,
            prompt=prompt,
            session_id=session_id,
            as_chunk=True,
            frames_per_chunk=args.frames_per_chunk,
        )
        t0 = time.time()
        actions = wrapper_policy.infer(obs)
        _log_action(actions, time.time() - t0, f"chunk_{i:03d}")
        all_actions.append(actions)

    if not args.skip_reset:
        wrapper_policy.reset({})
        logger.info("Reset sent")

    if all_actions:
        actions_np = np.concatenate(all_actions, axis=0)
        npy_path = os.path.join(output_dir, "local_infer_actions.npy")
        pt_path = os.path.join(output_dir, "local_infer_actions.pt")
        np.save(npy_path, actions_np)
        torch.save(torch.from_numpy(actions_np), pt_path)
        logger.info("Saved actions to: %s", npy_path)
        logger.info("Saved actions to: %s", pt_path)


def _run_rank_worker(
    policy: GrootSimPolicy,
    signal_group: dist.ProcessGroup,
    port: int,
) -> None:
    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=port,
        metadata={},
        output_dir=None,
        signal_group=signal_group,
    )
    asyncio.run(server._worker_loop())


def _broadcast_shutdown_to_workers(signal_group: dist.ProcessGroup) -> None:
    signal_tensor = torch.ones(1, dtype=torch.int32, device="cpu")
    dist.broadcast(signal_tensor, src=0, group=signal_group)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local DreamZero PNG inference (no websocket)")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, default=8000, help="Only used for worker helper object.")
    parser.add_argument("--timeout-seconds", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--debug-tensors-dir", default=None)
    parser.add_argument("--example-root", default="example")
    parser.add_argument("--preset", choices=["franka", "robotwin"], default="franka")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--frames-per-chunk", type=int, default=4)
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--num-dit-steps", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--save-debug-tensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug-rank0-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-dit-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-initial-single", action="store_true")
    parser.add_argument("--skip-reset", action="store_true")
    parser.add_argument(
        "--exterior0-source",
        choices=list(CAMERA_FILES.keys()),
        default="cam_high",
    )
    parser.add_argument(
        "--exterior1-source",
        choices=list(CAMERA_FILES.keys()),
        default="cam_right_wrist",
    )
    parser.add_argument(
        "--wrist-source",
        choices=list(CAMERA_FILES.keys()),
        default="cam_left_wrist",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = build_parser().parse_args()

    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"
    os.environ["ATTENTION_BACKEND"] = "TE"
    os.environ["NUM_DIT_STEPS"] = str(args.num_dit_steps)
    os.environ["DREAMZERO_DIT_NUM_LAYERS"] = str(args.num_layers)
    os.environ.setdefault("DREAMZERO_SKIP_COMPONENT_LOADING", "false")
    torch._dynamo.config.recompile_limit = 800

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    set_global_determinism(args.seed + rank, args.enable_deterministic)

    output_dir = _resolve_output_dir(args.model_path, args.index, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    debug_tensors_dir = args.debug_tensors_dir or os.path.join(output_dir, "debug_tensors")
    os.environ["DREAMZERO_DEBUG_SAVE_TENSORS"] = "true" if args.save_debug_tensors else "false"
    os.environ["DREAMZERO_DEBUG_SAVE_ROOT"] = debug_tensors_dir
    os.environ["DREAMZERO_DEBUG_RANK0_ONLY"] = "true" if args.debug_rank0_only else "false"
    if args.save_debug_tensors:
        os.makedirs(debug_tensors_dir, exist_ok=True)
    if rank == 0 and args.save_debug_tensors:
        logger.info("Debug tensors: %s", debug_tensors_dir)

    device_mesh = init_mesh()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    timeout_delta = datetime.timedelta(seconds=args.timeout_seconds)
    signal_group = dist.new_group(backend="gloo", timeout=timeout_delta)
    logger.info("Rank %s/%s initialized signal group", rank, world_size)

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag("oxe_droid"),
        model_path=args.model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
    )

    if hasattr(policy.trained_model, "action_head"):
        action_head = policy.trained_model.action_head
        action_head.seed = args.seed + rank
        action_head.num_inference_steps = args.num_inference_steps
        action_head.cfg_scale = args.cfg_scale
        action_head.debug_save_tensors = args.save_debug_tensors
        action_head.debug_save_root = debug_tensors_dir
        action_head.debug_rank0_only = args.debug_rank0_only
        action_head.debug_rank = rank
        logger.info(
            "Action head runtime config: num_inference_steps=%s cfg_scale=%s num_dit_steps=%s dit_num_layers=%s",
            action_head.num_inference_steps,
            action_head.cfg_scale,
            os.getenv("NUM_DIT_STEPS", "unset"),
            action_head.model.num_layers,
        )

    if rank == 0:
        wrapper_policy = ARDroidRoboarenaPolicy(
            groot_policy=policy,
            signal_group=signal_group,
            output_dir=output_dir,
            debug_save_tensors=args.save_debug_tensors,
            debug_root=debug_tensors_dir,
            debug_rank=rank,
            debug_rank0_only=args.debug_rank0_only,
        )
    else:
        wrapper_policy = None

    try:
        if rank == 0:
            _run_rank0_local_infer(wrapper_policy, args, output_dir)  # type: ignore[arg-type]
            if world_size > 1:
                _broadcast_shutdown_to_workers(signal_group)
        else:
            _run_rank_worker(policy, signal_group, args.port)
    finally:
        try:
            dist.barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    main()
