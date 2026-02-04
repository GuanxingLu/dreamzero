from dataclasses import dataclass, field
from typing import Tuple

from hydra.utils import instantiate
import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree

BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


# config
@dataclass
class VLAConfig(PretrainedConfig):
    model_type = "vla"
    backbone_cfg: PretrainedConfig = field(
        default=None, metadata={"help": "Backbone configuration."}
    )

    action_head_cfg: PretrainedConfig = field(
        default=None, metadata={"help": "Action head configuration."}
    )

    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})

    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


# real model
class VLA(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = VLAConfig
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    see discussion at https://nvidia.slack.com/archives/C07T1V7L886/p1732550624654139
    """

    def __init__(
        self,
        config: VLAConfig,
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)
        super().__init__(config)
        self.backbone = instantiate(config.backbone_cfg)
        self.action_head = instantiate(config.action_head_cfg)
        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

        self.rank = dist.get_rank() if dist.is_initialized() else None

    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline

        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1]
                in [
                    self.action_horizon,
                    2 * self.action_horizon,
                ]  # 2 * self.action_horizon for Q learning
                and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f"\n{action.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{action.shape=}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):

        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

    def forward(
        self,
        inputs: dict,
        validate: bool = True,
    ) -> BatchFeature:

        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)
        if validate:
            self.validate_data(action_head_outputs, backbone_outputs, is_training=True)

        print(f"[RANK {self.rank} HEARTBEAT] Forward done")

        return action_head_outputs

    def get_action(
        self,
        inputs: dict,
        num_action_samples: int = 1,
        inference_batch_size: int = 32,
        validate: bool = True,
    ) -> BatchFeature:
        """
        Sample actions from the model.

        Args:
            inputs (dict): dict where each input has a leading batch dimension
            num_action_samples (int): if specified, this function will sample num_action_samples actions for each item in the batch
            inference_batch_size (int): only used if num_action_samples > 1, this is the used batch size for each forward pass
            validate (bool): whether to validate the inputs and outputs
        """
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.get_action(
            backbone_outputs,
            action_inputs,
            num_action_samples=num_action_samples,
            inference_batch_size=inference_batch_size,
        )
        if validate:
            self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            # Only cast to self.compute_dtype if the tensor is floating
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            else:
                # Keep original dtype
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs

    @classmethod
    def from_pretrained_for_tuning(
        cls,
        pretrained_model_name_or_path: str,
        tune_backbone_kwargs: dict,
        tune_action_head_kwargs: dict,
        vla_override_kwargs={},
        backbone_override_kwargs={},
        action_head_override_kwargs={},
        expand_action_head_kwargs=None,
    ):
        print(
            f"Loading pretrained VLA from {pretrained_model_name_or_path} with VLA override kwargs: {vla_override_kwargs}"
        )
        print(f"Tune backbone kwargs: {tune_backbone_kwargs}")
        print(f"Tune action head kwargs: {tune_action_head_kwargs}")
        pretrained_vla = cls.from_pretrained(pretrained_model_name_or_path, **vla_override_kwargs)
        pretrained_vla.config.update(vla_override_kwargs)
        if "action_horizon" in vla_override_kwargs:
            pretrained_vla.config.action_head_cfg["config"]["action_horizon"] = vla_override_kwargs[
                "action_horizon"
            ]

        # Expand action head features if requested
        if expand_action_head_kwargs is not None:
            # Handle action dimension expansion
            if "expand_action_dim" in expand_action_head_kwargs:
                expand_dim_kwargs = expand_action_head_kwargs["expand_action_dim"]
                old_action_dim = expand_dim_kwargs.get("old_action_dim")
                new_action_dim = expand_dim_kwargs.get("new_action_dim")
                if old_action_dim is not None and new_action_dim is not None:
                    if old_action_dim == new_action_dim:
                        print(
                            f"Action dimension already matches target dimension {new_action_dim}, skipping expansion"
                        )
                    else:
                        print(
                            f"Expanding action dimension from {old_action_dim} to {new_action_dim}"
                        )
                        pretrained_vla.expand_action_dimension(old_action_dim, new_action_dim)
                else:
                    print(
                        "Warning: expand_action_dim provided but old_action_dim or new_action_dim is missing"
                    )

        pretrained_vla.backbone.set_trainable_parameters(**tune_backbone_kwargs)
        pretrained_vla.action_head.set_trainable_parameters(**tune_action_head_kwargs)

        print(f"Backbone cfg override kwargs: {backbone_override_kwargs}")
        print(f"Action head cfg override kwargs: {action_head_override_kwargs}")
        pretrained_vla.backbone.set_override_kwargs(**backbone_override_kwargs)
        pretrained_vla.action_head.set_override_kwargs(**action_head_override_kwargs)

        return pretrained_vla

    @classmethod
    def from_pretrained_with_wrapped_action_head(
        cls,
        pretrained_model_name_or_path: str,
        config: VLAConfig,
        tune_backbone_kwargs: dict,
        tune_action_head_kwargs: dict,
    ):
        """
        from_pretrained_with_wrapped_action_head:
            This method is usually used for wrapping around a pretrained action head with a new action head.
            Common use cases:
            - Residual model
            - Value model
            - Q model
            - ...
        """
        # 1. Load the pretrained VLAModel
        pretrained_vla = cls.from_pretrained(pretrained_model_name_or_path)
        pretrained_backbone_cfg_dict = pretrained_vla.config.backbone_cfg
        # Update config.backbone_cfg with the pretrained backbone config
        config.backbone_cfg = pretrained_backbone_cfg_dict

        # 2. Instantiate a new VLAModel
        vla = cls(config)

        # 3. Replace the backbone in the new VLAModel with the pretrained backbone
        vla.backbone = pretrained_vla.backbone
        vla.action_head.base_model = pretrained_vla.action_head

        # 4. Tune the backbone and action head
        print(f"Tune backbone kwargs: {tune_backbone_kwargs}")
        print(f"Tune action head kwargs: {tune_action_head_kwargs}")
        vla.backbone.set_trainable_parameters(**tune_backbone_kwargs)
        # Note that the action head is with residual model
        vla.action_head.set_trainable_parameters(**tune_action_head_kwargs)

        # 5. Return the new VLAModel
        return vla

    def expand_action_dimension(self, old_action_dim, new_action_dim):
        """
        Expand action dimension by copying weights from existing dimensions.
        This method should be called after loading a pretrained model but before fine-tuning
        when the target task requires more action dimensions than the pretrained model.

        Args:
            old_action_dim: Original action dimension from pretrained model
            new_action_dim: New (larger) action dimension for current task
        """
        if hasattr(self.action_head, "expand_action_dimension"):
            self.action_head.expand_action_dimension(old_action_dim, new_action_dim)
            # Update VLA config
            self.action_dim = new_action_dim
            self.config.action_dim = new_action_dim

            # Update any nested config that might contain action_dim
            self.config.action_head_cfg["config"]["action_dim"] = new_action_dim
            self.config.action_head_cfg["config"]["max_action_dim"] = new_action_dim
        else:
            raise NotImplementedError(
                f"Action head {type(self.action_head)} does not support action dimension expansion"
            )


class CotrainVLA(VLA):

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:
        if "cotrain" in inputs and inputs["cotrain"]:
            return self.backbone.cotrain(inputs)
        return super().forward(inputs)


def create_vla_with_pretrained_action_head(pretrained_vla_path: str, config: VLAConfig):
    # 1. Instantiate a new VLAModel
    vla = VLA(config)

    # 2. Load the pretrained VLAModel
    pretrained_vla = VLA.from_pretrained(pretrained_vla_path)

    # 3. Replace the action head in the new VLAModel with the pretrained action head
    vla.action_head = pretrained_vla.action_head

    # 4. Replace the action head config in the new VLAModel with the pretrained action head config
    vla.config.action_head_cfg = pretrained_vla.config.action_head_cfg

    # 5. Return the new VLAModel
    return vla


def create_vla_with_pretrained_backbone(
    pretrained_model_name_or_path: str,
    config: VLAConfig,
    tune_backbone_kwargs: dict,
    tune_action_head_kwargs: dict,
):
    # 1. Load the pretrained VLAModel
    pretrained_vla = VLA.from_pretrained(pretrained_model_name_or_path)
    pretrained_backbone_cfg_dict = pretrained_vla.config.backbone_cfg
    # Update config.backbone_cfg with the pretrained backbone config
    config.backbone_cfg = pretrained_backbone_cfg_dict

    # 2. Instantiate a new VLAModel
    vla = VLA(config)

    # 3. Replace the backbone in the new VLAModel with the pretrained backbone
    vla.backbone = pretrained_vla.backbone

    # 4. Tune the backbone and action head
    print(f"Tune backbone kwargs: {tune_backbone_kwargs}")
    print(f"Tune action head kwargs: {tune_action_head_kwargs}")
    vla.backbone.set_trainable_parameters(**tune_backbone_kwargs)
    vla.action_head.set_trainable_parameters(**tune_action_head_kwargs)

    # 6. Return the new VLAModel
    return vla


# register
AutoConfig.register("vla", VLAConfig)
AutoModel.register(VLAConfig, VLA)
