#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from transformers import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export DreamZero training-side preprocessing tensors for one fixed sample."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--droid-data-root", required=True)
    parser.add_argument("--dit-version", required=True)
    parser.add_argument("--text-encoder-pretrained-path", required=True)
    parser.add_argument("--image-encoder-pretrained-path", required=True)
    parser.add_argument("--vae-pretrained-path", required=True)
    parser.add_argument("--tokenizer-path", default="google/umt5-xxl")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-resolution-width", type=int, default=320)
    parser.add_argument("--image-resolution-height", type=int, default=176)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--action-horizon", type=int, default=24)
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--num-frame-per-block", type=int, default=2)
    parser.add_argument("--num-action-per-block", type=int, default=24)
    parser.add_argument("--num-state-per-block", type=int, default=1)
    parser.add_argument("--max-chunk-size", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument(
        "--disable-augment",
        action="store_true",
        help="Export tensors with eval-mode transforms to disable training-time video/language augmentation.",
    )
    return parser.parse_args()


def build_cfg(args: argparse.Namespace):
    repo_root = Path(__file__).resolve().parent.parent
    config_dir = repo_root / "groot" / "vla" / "configs"
    overrides = [
        "report_to=none",
        "wandb_project=dreamzero",
        f"output_dir={repo_root / 'outputs' / 'alignment_export_tmp'}",
        "data=dreamzero/droid_relative",
        "model=dreamzero/vla",
        "model/dreamzero/action_head=wan_flow_matching_action_tf",
        "model/dreamzero/transform=dreamzero_cotrain",
        "train_architecture=lora",
        f"num_frames={args.num_frames}",
        f"action_horizon={args.action_horizon}",
        "state_horizon=1",
        f"num_views={args.num_views}",
        f"num_frame_per_block={args.num_frame_per_block}",
        f"num_action_per_block={args.num_action_per_block}",
        f"num_state_per_block={args.num_state_per_block}",
        f"seed={args.seed}",
        "bf16=true",
        "tf32=true",
        "eval_bf16=true",
        "per_device_train_batch_size=1",
        "max_steps=1",
        "save_total_limit=5",
        "save_strategy=no",
        "dataloader_num_workers=0",
        "dataloader_pin_memory=false",
        "train_dataset.dataset_kwargs.video_backend=torchvision_av",
        f"max_chunk_size={args.max_chunk_size}",
        "frame_seqlen=880",
        f"droid_data_root={args.droid_data_root}",
        f"dit_version={args.dit_version}",
        f"text_encoder_pretrained_path={args.text_encoder_pretrained_path}",
        f"image_encoder_pretrained_path={args.image_encoder_pretrained_path}",
        f"vae_pretrained_path={args.vae_pretrained_path}",
        f"tokenizer_path={args.tokenizer_path}",
        f"image_resolution_width={args.image_resolution_width}",
        f"image_resolution_height={args.image_resolution_height}",
        "language_dropout_prob=0.0",
    ]
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="conf", overrides=overrides)


def cache_single_trajectory(dataset, episode_index: int) -> None:
    trajectory_ids = [episode_index]
    if hasattr(dataset, "frames_to_load"):
        (
            dataset.cached_shard,
            dataset.shard_start_indices,
            dataset.cached_df,
            dataset.frame_indices_map,
        ) = dataset.get_shard(
            trajectory_ids,
            dataset.modality_keys,
            dataset.all_video_paths,
            dataset.all_parquet_paths,
            dataset.frames_to_load,
            dataset.video_backend,
            dataset.video_backend_kwargs,
        )
        return

    dataset.cached_shard, dataset.shard_start_indices, dataset.cached_df = dataset.get_shard(
        trajectory_ids,
        dataset.modality_keys,
        dataset.all_video_paths,
        dataset.all_parquet_paths,
        dataset.video_backend,
        dataset.video_backend_kwargs,
        dataset.fps,
    )


def load_alignment_component_weights(action_head, args: argparse.Namespace) -> None:
    print("[alignment] load text/image/vae weights", flush=True)
    action_head.text_encoder.load_state_dict(
        torch.load(args.text_encoder_pretrained_path, map_location="cpu")
    )
    action_head.image_encoder.model.load_state_dict(
        torch.load(args.image_encoder_pretrained_path, map_location="cpu"),
        strict=False,
    )
    action_head.vae.model.load_state_dict(torch.load(args.vae_pretrained_path, map_location="cpu"))


def disable_video_augmentation_transforms(dataset) -> None:
    transforms = getattr(dataset, "transforms", None)
    if transforms is None:
        raise RuntimeError("dataset has no transforms to modify")
    disabled = 0
    for transform in getattr(transforms, "transforms", []):
        if transform.__class__.__name__ in {
            "VideoCrop",
            "VideoColorJitter",
            "VideoRandomGrayscale",
            "VideoRandomPosterize",
            "VideoRandomErasing",
        }:
            transform.eval()
            disabled += 1
    if disabled == 0:
        raise RuntimeError("no video augmentation transforms found to disable")


def get_deterministic_language(step_data: dict) -> str:
    value = step_data.get("annotation.language.language_instruction")
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    if value is not None:
        return str(value)
    task = step_data.get("task")
    if task is None:
        raise RuntimeError("missing deterministic language source in step_data")
    return str(task)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training alignment export")

    torch.cuda.set_device(0)
    os.environ["DREAMZERO_DIT_NUM_LAYERS"] = str(args.num_layers)
    os.environ["DREAMZERO_SKIP_DIFFUSION_MODEL_INIT"] = "true"
    cfg = build_cfg(args)
    set_seed(args.seed)

    print("[alignment] instantiate train_dataset", flush=True)
    train_dataset = instantiate(cfg.train_dataset)
    print("[alignment] instantiate data_collator", flush=True)
    data_collator = instantiate(cfg.data_collator)
    print("[alignment] instantiate action_head", flush=True)
    action_head = instantiate(cfg.model.config.action_head_cfg).eval()
    load_alignment_component_weights(action_head, args)
    print("[alignment] action_head ready", flush=True)

    dataset = train_dataset.datasets[0]
    if args.disable_augment:
        print("[alignment] disable video augmentation transforms", flush=True)
        disable_video_augmentation_transforms(dataset)
    print(f"[alignment] cache episode {args.episode_index}", flush=True)
    cache_single_trajectory(dataset, args.episode_index)
    try:
        indices = {
            key: delta_indices + args.step_index for key, delta_indices in dataset.delta_indices.items()
        }
        print(f"[alignment] load step_data episode={args.episode_index} step={args.step_index}", flush=True)
        step_data = dataset.get_step_data(args.episode_index, indices)
        if step_data is None:
            raise RuntimeError(
                f"empty sample for episode_index={args.episode_index}, step_index={args.step_index}"
            )
        print("[alignment] apply transforms", flush=True)
        transformed = dataset.transforms(step_data)
        if args.disable_augment:
            transformed["text"] = get_deterministic_language(step_data)
    finally:
        dataset.delete_cached_shard()

    print("[alignment] collate batch", flush=True)
    batch = data_collator([transformed])
    action_inputs = action_head.prepare_input(batch)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[alignment] extract tensors", flush=True)
    with torch.inference_mode():
        action_head.save_training_alignment_tensors(
            str(output_dir),
            action_head.extract_training_alignment_tensors_sequential(action_inputs),
        )
    print("[alignment] save metadata", flush=True)

    metadata = {
        "episode_index": args.episode_index,
        "step_index": args.step_index,
        "num_layers": args.num_layers,
        "image_resolution_width": args.image_resolution_width,
        "image_resolution_height": args.image_resolution_height,
        "num_frames": args.num_frames,
        "action_horizon": args.action_horizon,
        "disable_augment": args.disable_augment,
        "raw_keys": sorted(step_data.keys()),
        "transformed_keys": sorted(transformed.keys()),
    }
    (output_dir / "sample_meta.json").write_text(json.dumps(metadata, indent=2))
    (output_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    main()
