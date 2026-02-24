from groot.vla.model.dreamzero.modules.wan2_1_attention import AttentionModule
from groot.vla.model.n1_5.modules.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)
from groot.vla.model.dreamzero.modules.wan2_1_submodule import (
    WanRMSNorm,
    rope_action_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 frame_seqlen,
                 local_attn_size=-1,
                 sink_size=0,
                 num_frame_per_block=1,
                 qk_norm=True,
                 eps=1e-6,
                 num_action_per_block=32,
                 num_state_per_block=1):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.num_frame_per_block = num_frame_per_block
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 21 * frame_seqlen if local_attn_size == -1 else local_attn_size * frame_seqlen
        self.frame_seqlen = frame_seqlen
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block
        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.attn = AttentionModule(num_heads=self.num_heads, head_dim=self.head_dim)
        self.causal_attn = AttentionModule(num_heads=self.num_heads, head_dim=self.head_dim, causal=True)

    def _blockwise_causal_flash_attn(self, q, k, v, frame_seqlen, num_frame_per_block=1, 
                                       action_horizon=None, state_horizon=None, 
                                       num_action_per_block=None, num_state_per_block=None,
                                       visualize_mask=False):
        """
        Implement blockwise causal attention using flash_attention.
        Matches the pattern from _prepare_blockwise_causal_attn_mask:
        
        Structure:
        - First image: conditioning only, cannot attend to anything
        - Image blocks: can attend to first image + previous image blocks + current action block + current state block
        - Action blocks: can attend to previous image blocks + current image block + current state block + first image
        - State blocks: conditioning only, cannot attend to anything
        
        Args:
            q, k, v: Query, key, value tensors [B, L, num_heads, head_dim]
            frame_seqlen: Number of tokens per frame
            num_frame_per_block: Number of frames per attention block
            action_horizon: Total number of action tokens (if None, no action/state tokens)
            state_horizon: Total number of state tokens (if None, no action/state tokens)
            num_action_per_block: Number of action tokens per block
            num_state_per_block: Number of state tokens per block
            visualize_mask: If True, print the attention mask pattern
        
        Returns:
            Attention output [B, L, num_heads, head_dim]
        """
        b, total_len, n, d = q.shape
        
        # Check if we have action/state tokens
        has_action_state = (action_horizon is not None and state_horizon is not None)
        
        if not has_action_state:
            # OPTIMIZED: Simple blockwise causal attention (without action/state tokens)
            num_frames = total_len // frame_seqlen
            block_size = frame_seqlen * num_frame_per_block
            num_blocks = (num_frames - 1) // num_frame_per_block
            
            # Handle edge case when sequence is too short (no blocks to process)
            if num_blocks <= 0:
                # Process entire sequence as a single block
                return self.attn(q, k, v)
            
            # OPTIMIZATION: For global attention, process all blocks in one call with causal masking
            if self.local_attn_size == -1:
                # Single flash_attention call with causal=True for all blocks at once
                # This is much faster than looping!
                return self.causal_attn(q, k, v)
            
            # With local attention, still need loop but optimize it
            # Pre-allocate output tensor
            output = torch.empty_like(q)
            
            # Pre-compute block boundaries
            block_starts = [frame_seqlen + i * block_size for i in range(num_blocks)]
            block_ends = [min(start + block_size, total_len) for start in block_starts]
            kv_starts = [max(0, end - self.local_attn_size * frame_seqlen) for end in block_ends]
            
            for block_idx in range(num_blocks):
                block_start = block_starts[block_idx]
                block_end = block_ends[block_idx]
                kv_start = kv_starts[block_idx]
                
                output[:, block_start:block_end] = self.attn(
                    q[:, block_start:block_end],
                    k[:, kv_start:block_end],
                    v[:, kv_start:block_end]
                )
            
            return output

        assert action_horizon is not None and state_horizon is not None
        assert num_action_per_block is not None and num_state_per_block is not None

        # Multi-modal structure: [first image] [image blocks] [action blocks] [state blocks]
        # Calculate block structure
        first_image_len = frame_seqlen
        action_len = action_horizon
        state_len = state_horizon
        image_blocks_len = total_len - first_image_len - action_len - state_len
        
        num_image_blocks = image_blocks_len // (num_frame_per_block * frame_seqlen)
        num_action_blocks = action_horizon // num_action_per_block
        num_state_blocks = state_horizon // num_state_per_block

        assert num_image_blocks == num_action_blocks == num_state_blocks
        
        # Token ranges
        first_image_start = 0
        first_image_end = first_image_len
        image_blocks_start = first_image_end
        image_blocks_end = image_blocks_start + image_blocks_len
        action_start = image_blocks_end
        action_end = action_start + action_len
        state_start = action_end
        state_end = state_start + state_len
        
        # OPTIMIZED: Pre-allocate output tensor and pre-compute all indices
        output = torch.empty_like(q)
        
        # Process first image (conditioning, can only self-attend)
        output[:, first_image_start:first_image_end] = self.attn(
            q[:, first_image_start:first_image_end],
            k[:, first_image_start:first_image_end],
            v[:, first_image_start:first_image_end]
        )
        
        # Pre-compute all block indices for image blocks
        image_block_starts = [image_blocks_start + i * num_frame_per_block * frame_seqlen for i in range(num_image_blocks)]
        image_block_ends = [image_blocks_start + (i + 1) * num_frame_per_block * frame_seqlen for i in range(num_image_blocks)]
        if self.local_attn_size != -1:
            image_kv_starts = [max(image_blocks_start, end - self.local_attn_size * frame_seqlen) for end in image_block_ends]
        else:
            image_kv_starts = [image_blocks_start] * num_image_blocks
        
        # Pre-compute action and state block indices
        action_block_starts = [action_start + i * num_action_per_block for i in range(num_action_blocks)]
        action_block_ends = [action_start + (i + 1) * num_action_per_block for i in range(num_action_blocks)]
        state_block_starts = [state_start + i * num_state_per_block for i in range(num_state_blocks)]
        state_block_ends = [state_start + (i + 1) * num_state_per_block for i in range(num_state_blocks)]
        
        # Process each image block
        for block_idx in range(num_image_blocks):
            block_start = image_block_starts[block_idx]
            block_end = image_block_ends[block_idx]
            image_kv_start = image_kv_starts[block_idx]
            action_block_start = action_block_starts[block_idx]
            action_block_end = action_block_ends[block_idx]
            state_block_start = state_block_starts[block_idx]
            state_block_end = state_block_ends[block_idx]
            
            # Build context: first image + relevant image blocks + current action + current state
            k_context = torch.cat([
                k[:, first_image_start:first_image_end],  # First image
                k[:, image_kv_start:block_end],  # Image blocks
                k[:, action_block_start:action_block_end],  # Current action block
                k[:, state_block_start:state_block_end]  # Current state block
            ], dim=1)
            v_context = torch.cat([
                v[:, first_image_start:first_image_end],
                v[:, image_kv_start:block_end],
                v[:, action_block_start:action_block_end],
                v[:, state_block_start:state_block_end]
            ], dim=1)
            
            output[:, block_start:block_end] = self.attn(
                q[:, block_start:block_end], k_context, v_context
            )
        
        # Process each action block
        for block_idx in range(num_action_blocks):
            action_block_start = action_block_starts[block_idx]
            action_block_end = action_block_ends[block_idx]
            image_block_end = image_block_ends[block_idx]
            state_block_start = state_block_starts[block_idx]
            state_block_end = state_block_ends[block_idx]
            
            # Determine image context range
            if self.local_attn_size != -1:
                image_kv_start = max(image_blocks_start, image_block_end - self.local_attn_size * frame_seqlen)
            else:
                image_kv_start = image_blocks_start
            
            # Build context
            k_context = torch.cat([
                k[:, first_image_start:first_image_end],  # First image
                k[:, image_kv_start:image_block_end],  # Image blocks
                k[:, action_block_start:action_block_end],  # Current action block
                k[:, state_block_start:state_block_end]  # Current state block
            ], dim=1)
            v_context = torch.cat([
                v[:, first_image_start:first_image_end],
                v[:, image_kv_start:image_block_end],
                v[:, action_block_start:action_block_end],
                v[:, state_block_start:state_block_end]
            ], dim=1)
            
            output[:, action_block_start:action_block_end] = self.attn(
                q[:, action_block_start:action_block_end], k_context, v_context
            )
        
        # Process state blocks (conditioning, can only self-attend)
        for block_idx in range(num_state_blocks):
            state_block_start = state_block_starts[block_idx]
            state_block_end = state_block_ends[block_idx]
            
            output[:, state_block_start:state_block_end] = self.attn(
                q[:, state_block_start:state_block_end],
                k[:, state_block_start:state_block_end],
                v[:, state_block_start:state_block_end]
            )
        
        return output

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        freqs_action: torch.Tensor,
        freqs_state: torch.Tensor,
        action_register_length: int | None,
    ) -> torch.Tensor:
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        roped_query = rope_action_apply(
            x=q,
            freqs=freqs,
            freqs_action=freqs_action,
            freqs_state=freqs_state,
            action_register_length=action_register_length,
            num_action_per_block=self.num_action_per_block,
            num_state_per_block=self.num_state_per_block,
        ).type_as(v)
        roped_key = rope_action_apply(
            x=k,
            freqs=freqs,
            freqs_action=freqs_action,
            freqs_state=freqs_state,
            action_register_length=action_register_length,
            num_action_per_block=self.num_action_per_block,
            num_state_per_block=self.num_state_per_block,
        ).type_as(v)

        # Calculate dynamic action and state horizons
        if action_register_length is not None:
            chunk_size = action_register_length // (self.num_action_per_block + self.num_state_per_block)
            action_horizon = chunk_size * self.num_action_per_block
            state_horizon = chunk_size * self.num_state_per_block
        else:
            action_horizon = None
            state_horizon = None

        # Use blockwise causal flash attention without massive padding
        visualize = False
        x = self._blockwise_causal_flash_attn(
            roped_query, roped_key, v, self.frame_seqlen, self.num_frame_per_block,
            action_horizon=action_horizon,
            state_horizon=state_horizon,
            num_action_per_block=self.num_action_per_block if action_register_length else None,
            num_state_per_block=self.num_state_per_block if action_register_length else None,
            visualize_mask=visualize)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 frame_seqlen,
                 local_attn_size=-1,
                 sink_size=0,
                 num_frame_per_block=1,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 num_action_per_block=32,
                 num_state_per_block=1):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(
            dim=dim,
            num_heads=num_heads,
            frame_seqlen=frame_seqlen,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            num_frame_per_block=num_frame_per_block,
            qk_norm=qk_norm,
            eps=eps,
            num_action_per_block=num_action_per_block,
            num_state_per_block=num_state_per_block,
        )
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        freqs: torch.Tensor,
        freqs_action: torch.Tensor,
        freqs_state: torch.Tensor,
        action_register_length: int | None,
        context: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        # self-attention
        y = self.self_attn(
            x=(self.norm1(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)),
            freqs=freqs,
            freqs_action=freqs_action,
            freqs_state=freqs_state,
            action_register_length=action_register_length,
        )
        x = x + (y * e[2].squeeze(2))

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, e):
            x = x + self.cross_attn(self.norm3(x), context)
            y = self.ffn(
                (self.norm2(x) * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            )
            x = x + (y * e[5].squeeze(2))
            return x

        x = cross_attn_ffn(x, context, e)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 frame_seqlen=220,
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 max_chunk_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 num_frame_per_block=1, 
                 action_dim=32,
                 num_registers=8,
                 max_state_dim=64,
                 max_num_embodiments=32,
                 hidden_size=1024,
                 diffusion_model_pretrained_path=None,
                 num_action_per_block=32,
                 num_state_per_block=1):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.frame_seqlen = frame_seqlen
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = max_chunk_size * num_frame_per_block + 1 if max_chunk_size != -1 else -1
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.num_frame_per_block = num_frame_per_block
        self.diffusion_model_pretrained_path = diffusion_model_pretrained_path
        self.action_dim = action_dim
        self.num_registers = num_registers
        self.max_state_dim = max_state_dim
        self.max_num_embodiments = max_num_embodiments
        self.hidden_size = hidden_size
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block

        max_num_embodiments = 1

        self.state_encoder = CategorySpecificMLP(
            num_categories=max_num_embodiments,
            input_dim=max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=action_dim,
            hidden_size=self.dim,
            num_embodiments=max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=max_num_embodiments,
            input_dim=dim,
            hidden_dim=self.hidden_size,
            output_dim=action_dim,
        )

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads, frame_seqlen,
                                    self.local_attn_size, sink_size, num_frame_per_block, qk_norm, cross_attn_norm, eps,
                                    num_action_per_block, num_state_per_block)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        
        self.freqs_action = rope_params(1024*10, d)
        self.freqs_state = rope_params(1024, d)
        self.freqs = [
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ]
        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = True
        self.independent_first_frame = False if self.num_frame_per_block == 1 else True

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def _forward_blocks(
        self,
        x: torch.Tensor,
        seq_len: int,
        freqs: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: torch.Tensor | None,
        embodiment_id: torch.Tensor | None,
        action: torch.Tensor | None,
        timestep_action: torch.Tensor | None,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""
        Forward pass through the diffusion model blocks.
        """
        x = x.flatten(start_dim=2).transpose(1, 2)

        B = x.shape[0]
        F = timestep.shape[1]

        if action is not None:
            embodiment_id = torch.tensor([0], device=x.device).repeat(x.shape[0])
            action_features = self.action_encoder(action, timestep_action, embodiment_id)
            state_features = self.state_encoder(state, embodiment_id)
            action_register = torch.cat([action_features, state_features], dim=1)
            action_length = action_features.shape[1]
            action_register_length = action_register.shape[1]
            x = torch.cat([x, action_register], dim=1)
        else:
            action_features = None
            state_features = None
            action_length = 0
            action_register_length = None

        # time embeddings
        timestep = timestep.unsqueeze(-1).expand(B, F, seq_len // F).reshape(B, -1)

        if action is not None:
            assert timestep_action is not None
            assert state_features is not None
            stride = timestep_action.shape[1] // state_features.shape[1]
            timestep_state = timestep_action[:, ::stride]
            timestep = torch.cat([timestep, timestep_action, timestep_state], dim=1)

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x))
        e = e.unflatten(dim=0, sizes=(B, -1))
        e0 = self.time_projection(e)
        e0 = e0.unflatten(dim=2, sizes=(6, self.dim))

        # context
        context = self.text_embedding(context)

        if clip_feature is not None:
            clip_embedding = self.img_emb(clip_feature)
            context = torch.cat([clip_embedding, context], dim=1)

        for block in self.blocks:
            x = block(
                x=x,
                e=e0,
                freqs=freqs,
                freqs_action=self.freqs_action,
                freqs_state=self.freqs_state,
                context=context,
                action_register_length=action_register_length,
            )

        if action is not None:
            action_noise_pred = x[:, seq_len: seq_len + action_length]
            action_noise_pred = self.action_decoder(action_noise_pred, embodiment_id)
        else:
            action_noise_pred = None

        # Build a tensor that contains only video tokens per sample with length = max(video_lens)
        x_video = x[:, :seq_len]
        e_video = e[:, :seq_len]

        # Unpatchify video-only tokens
        x_video = self.head(x_video, e_video.unsqueeze(2))

        return x_video, action_noise_pred

    def _forward_train(
        self,
        x,
        timestep,
        timestep_action,
        context,
        seq_len,
        y=None,
        clip_feature=None,
        action=None,
        state=None,
        embodiment_id=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_feature is not None and y is not None

        if y is not None:
            x = torch.cat([x, y.to(dtype=x.dtype)], dim=1)

        # embeddings
        x = self.patch_embedding(x)

        grid_size = torch.tensor(x.shape[2:], dtype=torch.long)
        freqs = self._create_freqs(
            grid_size=grid_size,
            start_frame=0,
        )

        x = x.flatten(start_dim=2).transpose(1, 2)
        assert x.shape[1] == seq_len

        B = x.shape[0]
        F = timestep.shape[1]

        # time embeddings
        if action is not None:
            embodiment_id = torch.tensor([0], device=x.device).repeat(x.shape[0])
            action_features = self.action_encoder(action, timestep_action, embodiment_id)
            action_length = action_features.shape[1]
            state_features = self.state_encoder(state, embodiment_id)
            action_register = torch.cat([action_features, state_features], dim=1)
            action_register_length = action_register.shape[1]
            x = torch.cat([x, action_register], dim=1)
        else:
            action_features = None
            action_length = None
            state_features = None
            action_register = None
            action_register_length = None

        # time embeddings
        timestep = timestep.unsqueeze(-1).expand(B, F, seq_len // F).reshape(B, -1)

        if action is not None:
            assert timestep_action is not None
            assert state_features is not None
            stride = timestep_action.shape[1] // state_features.shape[1]
            timestep_state = timestep_action[:, ::stride]
            timestep = torch.cat([timestep, timestep_action, timestep_state], dim=1)

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x))
        e = e.unflatten(dim=0, sizes=(B, -1))
        e0 = self.time_projection(e)
        e0 = e0.unflatten(dim=2, sizes=(6, self.dim))

        # context
        assert context.shape[1] == self.text_len
        context = self.text_embedding(context)

        if clip_feature is not None:
            clip_embedding = self.img_emb(clip_feature)
            context = torch.cat([clip_embedding, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            freqs=freqs,
            freqs_action=self.freqs_action,
            freqs_state=self.freqs_state,
            action_register_length=action_register_length,
            context=context,
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        if action is not None:
            action_noise_pred = x[:, seq_len: seq_len + action_length]
            action_noise_pred = self.action_decoder(action_noise_pred, embodiment_id)
        else:
            action_noise_pred = None

        # Build a tensor that contains only video tokens per sample with length = max(video_lens)
        x_video = x[:, :seq_len]
        e_video = e[:, :seq_len]

        # Unpatchify video-only tokens
        x_video = self.head(x_video, e_video.unsqueeze(2))
        video_noise_pred = self.unpatchify(x_video, grid_size)

        return video_noise_pred, action_noise_pred

    def forward(
        self,
        *args,
        **kwargs
    ):
        return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_size):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (Tensor):
                Patchified features, with shape [B, L, C_out * prod(patch_size)].
            grid_size (Tensor):
                Spatial-temporal grid dimensions before patching, with shape [3]
                (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            Tensor:
                Reconstructed video tensors with shape [B, C_out, F, H / 8, W / 8]
        """
        B = x.shape[0]
        c = self.out_dim
        grid_size = grid_size.tolist()
        assert x.shape[1] == math.prod(grid_size)
        x = x.view(B, *grid_size, *self.patch_size, c)
        x = torch.einsum('bfhwpqrc->bcfphqwr', x)
        x = x.reshape(B, c, *[i * j for i, j in zip(grid_size, self.patch_size)])
        return x

    def _create_freqs(
        self,
        grid_size: torch.Tensor,
        start_frame: int,
    ):
        device = self.patch_embedding.weight.device
        if any(freq.device != device for freq in self.freqs):
            self.freqs = [freq.to(device) for freq in self.freqs]
        if self.freqs_action.device != device:
            self.freqs_action = self.freqs_action.to(device)
        if self.freqs_state.device != device:
            self.freqs_state = self.freqs_state.to(device)

        f, h, w = grid_size.tolist()
        freqs = torch.cat(
            [
                self.freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1
        ).reshape(f * h * w, 1, -1)

        return freqs

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
