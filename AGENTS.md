# DreamZero Codebase Overview

## Snapshot
- Core implementation lives in `groot/vla/` (configs, datasets/transforms, model, trainer).
- Operational scripts live in `scripts/` (training, data conversion, local inference launch).
- Evaluation/websocket integration is in `eval_utils/` and vendored `sim-evals/`.

## Training File Flow
1. Optional data conversion:
   - `scripts/data/convert_droid.py`
2. Training launch scripts:
   - `scripts/train/droid_training.sh` (LoRA)
   - `scripts/train/droid_training_full_finetune.sh` (full finetune)
3. Entrypoint:
   - `groot/vla/experiment/experiment.py` (`@hydra.main`)
4. Experiment assembly:
   - `groot/vla/experiment/base.py` (create model, dataset, trainer; run train loop)
5. Config wiring:
   - `groot/vla/configs/conf.yaml`
   - `groot/vla/configs/data/dreamzero/droid_relative.yaml`
   - `groot/vla/configs/model/dreamzero/vla.yaml`
   - `groot/vla/configs/model/dreamzero/action_head/wan_flow_matching_action_tf.yaml`
   - `groot/vla/configs/model/dreamzero/transform/dreamzero_cotrain.yaml`
6. Data loading + transforms:
   - `groot/vla/data/dataset/lerobot_sharded.py`
   - `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py`
7. Model forward path:
   - `groot/vla/model/dreamzero/base_vla.py`
   - `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`

## Inference File Flow
1. Distributed server entrypoint:
   - `socket_test_optimized_AR.py`
2. Policy/model load:
   - `groot/vla/model/n1_5/sim_policy.py` (`GrootSimPolicy`)
3. Robot-format wrapper + frame accumulation:
   - `ARDroidRoboarenaPolicy` in `socket_test_optimized_AR.py`
4. Inference call chain:
   - `ARDroidRoboarenaPolicy.infer(...)`
   - `GrootSimPolicy.lazy_joint_forward_causal(...)`
   - `VLA.lazy_joint_video_action_causal(...)`
   - `WANPolicyHead.lazy_joint_video_action(...)`
5. Local non-websocket path:
   - `scripts/run_launch_dreamzero_local_png_sync.sh`
   - `local_infer_example_png_AR.py`
6. Simulation evaluation client path:
   - `eval_utils/run_sim_eval.py`
   - `eval_utils/policy_client.py` (client)
   - `eval_utils/policy_server.py` (server protocol)

