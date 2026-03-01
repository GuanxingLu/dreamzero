# DreamZero 模型结构 - Tensor流向图

## 推理入口
```
scripts/run_launch_dreamzero_local_png_sync.sh
  → local_infer_example_png_AR.py::main()
  → GrootSimPolicy (groot/vla/model/n1_5/sim_policy.py)
  → ARDroidRoboarenaPolicy (socket_test_optimized_AR.py)
```

## Tensor流向

### 1. 输入处理
```python
# ARDroidRoboarenaPolicy._convert_observation()
输入:
  observation/exterior_image_0_left: (H, W, 3) uint8
  observation/exterior_image_1_left: (H, W, 3) uint8
  observation/wrist_image_left: (H, W, 3) uint8
  observation/joint_position: (7,) float32
  observation/gripper_position: (1,) float32
  prompt: str

帧累积后:
  video.exterior_image_1_left: (T, H, W, 3)  # T=1 or 4
  video.exterior_image_2_left: (T, H, W, 3)
  video.wrist_image_left: (T, H, W, 3)
  state.joint_position: (1, 7) float64
  state.gripper_position: (1, 1) float64
  annotation.language.action_text: str
```

### 2. WANPolicyHead.forward()
**类**: `WANPolicyHead` (groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py)

```python
# 输入
videos: (B, T, H, W, C) uint8
text: str
state: (B, 1, 7+1) float64
action: (B, A, 8) float32  # training only

# Step 1: 视频预处理
videos = rearrange(videos, "b t h w c -> b c t h w")  # (B, C, T, H, W)
videos = normalize(videos / 255.0)  # [-1, 1]

# Step 2: 文本编码
# self.text_encoder (T5)
prompt_embs = self.encode_prompt(text, attention_mask)
# → (B, max_length, text_dim)

# Step 3: 视频编码 (VAE)
# self.vae
latents = self.encode_video(videos, tiled=True)
# → (B, latent_channels, T, H_latent, W_latent)
latents = latents.transpose(1, 2)  # (B, T, C, H, W)

# Step 4: 图像编码 (CLIP)
# self.image_encoder
image = videos[:, :, :1]  # 首帧
clip_feas, ys, _ = self.encode_image(image, num_frames, height, width)
# clip_feas: (B, clip_dim)
# ys: (B, y_dim)

# Step 5: 噪声采样 (训练时)
noise = torch.randn_like(latents)  # (B, T, C, H, W)
timestep_id = sample_timesteps()  # (B, T)
noisy_latents = self.scheduler.add_noise(latents, noise, timestep_id)

# Step 6: DiT前向传播
# self.model (CausalWanModel)
model_output = self.model(
    hidden_states=noisy_latents,  # (B, T, C, H, W)
    timestep=timestep_id,         # (B, T)
    encoder_hidden_states=prompt_embs,  # (B, L, D)
    clip_feas=clip_feas,          # (B, clip_dim)
    ys=ys,                        # (B, y_dim)
    actions=noisy_actions,        # (B, A, 8)
    states=state_features,        # (B, S, 8)
    embodiment_id=embodiment_id,  # (B,)
)
# → model_output.sample: (B, T, C, H, W)
# → model_output.action_pred: (B, A, 8)
```

### 3. CausalWanModel.forward()
**类**: `CausalWanModel` (groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py)

```python
# 输入
hidden_states: (B, T, C, H, W)  # 噪声视频latent
timestep: (B, T)
encoder_hidden_states: (B, L, text_dim)  # 文本特征
actions: (B, A, 8)  # 噪声动作
states: (B, S, 8)   # 状态
embodiment_id: (B,)

# Step 1: Patch Embedding
# self.patch_embed
x = self.patch_embed(hidden_states)
# (B, T, C, H, W) → (B, T, H*W, D)
# D = hidden_size (e.g., 1536)

# Step 2: 位置编码
# self.pos_embed_spatial (RoPE for spatial)
# self.pos_embed_temporal (RoPE for temporal)
freqs_spatial = rope_params(...)  # (H*W, D/2)
freqs_temporal = rope_params(...)  # (T, D/2)

# Step 3: Action/State编码
# self.action_encoder (MultiEmbodimentActionEncoder)
action_emb = self.action_encoder(actions, timestep, embodiment_id)
# (B, A, 8) → (B, A, D)

# self.state_encoder (MultiEmbodimentActionEncoder)
state_emb = self.state_encoder(states, timestep, embodiment_id)
# (B, S, 8) → (B, S, D)

# Step 4: 构建序列
# Token序列: [first_frame | image_blocks | action_registers | state_registers]
first_frame = x[:, 0]  # (B, H*W, D)
image_blocks = x[:, 1:]  # (B, T-1, H*W, D)
image_blocks = rearrange(image_blocks, "b t hw d -> b (t hw) d")

sequence = torch.cat([
    first_frame,      # (B, H*W, D)
    image_blocks,     # (B, (T-1)*H*W, D)
    action_emb,       # (B, A, D)
    state_emb,        # (B, S, D)
], dim=1)
# → (B, total_len, D)
# total_len = H*W + (T-1)*H*W + A + S

# Step 5: Transformer Blocks
# self.blocks (nn.ModuleList of CausalWanAttentionBlock)
for block in self.blocks:
    sequence = block(
        sequence,
        freqs_spatial,
        freqs_temporal,
        encoder_hidden_states,  # cross-attention
    )
# → (B, total_len, D)

# Step 6: 输出解码
# self.final_layer (CausalHead)
output = self.final_layer(sequence)

# 分离图像和动作
image_len = H*W + (T-1)*H*W
image_output = output[:, :image_len]  # (B, T*H*W, C*patch_size^2)
action_output = output[:, image_len:image_len+A]  # (B, A, D)

# Unpatchify
# self.unpatchify
video_pred = self.unpatchify(image_output)
# (B, T*H*W, C*patch_size^2) → (B, T, C, H, W)

# Action decoder
# self.action_decoder (CategorySpecificMLP)
action_pred = self.action_decoder(action_output, embodiment_id)
# (B, A, D) → (B, A, 8)

# 返回
return {
    'sample': video_pred,      # (B, T, C, H, W)
    'action_pred': action_pred # (B, A, 8)
}
```

### 4. 推理输出
```python
# WANPolicyHead 推理模式
# 使用 FlowMatchScheduler 进行去噪

# 初始化随机噪声
latents = torch.randn(B, T, C, H, W)
actions = torch.randn(B, A, 8)

# 迭代去噪 (num_inference_steps 步)
for t in scheduler.timesteps:
    # 预测噪声
    model_output = self.model(
        hidden_states=latents,
        timestep=t,
        encoder_hidden_states=prompt_embs,
        actions=actions,
        states=state_features,
        ...
    )

    # 更新latents和actions
    latents = scheduler.step(model_output.sample, t, latents)
    actions = scheduler.step(model_output.action_pred, t, actions)

# 解码latents (VAE decoder)
videos = self.vae.decode(latents)

# 最终动作输出
final_actions = actions  # (B, A, 8)
# A = action_horizon (e.g., 32)
# 8 = 7 joints + 1 gripper
```

## 关键类定义

### 核心模型类
- `WANPolicyHead`: 主动作头 (groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py:207)
- `CausalWanModel`: DiT主模型 (groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py:1218)
- `CausalWanAttentionBlock`: Transformer块 (wan_video_dit_action_casual_chunk.py:1087)
- `CausalWanSelfAttention`: 因果自注意力 (wan_video_dit_action_casual_chunk.py:188)
- `CausalHead`: 输出头 (wan_video_dit_action_casual_chunk.py:1190)

### 编码器类
- `MultiEmbodimentActionEncoder`: 动作编码器 (wan_video_dit_action_casual_chunk.py:57)
- `CategorySpecificLinear`: 具身体特定线性层 (wan_video_dit_action_casual_chunk.py:31)
- `CategorySpecificMLP`: 具身体特定MLP (wan_video_dit_action_casual_chunk.py:45)

### 调度器类
- `FlowMatchScheduler`: Flow matching调度器 (groot/vla/model/dreamzero/modules/flow_match_scheduler.py)
- `FlowUniPCMultistepScheduler`: 多步调度器 (groot/vla/model/dreamzero/modules/flow_unipc_multistep_scheduler.py)

## Tensor形状速查

```
输入:
  images: (B, T, H, W, 3) uint8
  text: str
  state: (B, 1, 8) float64

编码后:
  latents: (B, T, C_latent, H_latent, W_latent)  # C_latent=16, H_latent=H/8, W_latent=W/8
  prompt_embs: (B, max_length, text_dim)  # text_dim=4096 (T5)
  clip_feas: (B, clip_dim)  # clip_dim=1024

DiT内部:
  sequence: (B, total_len, D)  # D=1536
  total_len = H_latent*W_latent*T + action_horizon + state_horizon

输出:
  video_pred: (B, T, C_latent, H_latent, W_latent)
  action_pred: (B, action_horizon, 8)  # action_horizon=32
```
