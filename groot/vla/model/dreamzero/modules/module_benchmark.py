import torch
import time
import contextlib
from torch.profiler import profile, ProfilerActivity
from groot.vla.model.dreamzero.modules.wan2_1_attention import AttentionModule
from groot.vla.model.dreamzero.modules.wan_video_dit_causal_chunk import CausalWanModel


def benchmark_wan_model(n_iters, n_warmup_iters):
    wan_model = CausalWanModel(
        model_type='i2v',
        patch_size=(1, 2, 2),
        frame_seqlen=880,
        text_len=512,
        in_dim=36,
        dim=5120,
        ffn_dim=13824,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=40,
        num_layers=40,
        max_chunk_size=6,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-06,
        num_frame_per_block=2,
        hidden_size=1024,
        diffusion_model_pretrained_path=None,
    )
    print("WanModel initialized")
    wan_model.to(dtype=torch.bfloat16, device="cuda")
    print("Moved WanModel to device")
    wan_model.train()

    def train_func(**kwargs):
        kwargs_cloned = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                kwargs_cloned[key] = value.detach().clone().requires_grad_(True)
            elif isinstance(value, int):
                kwargs_cloned[key] = value
            else:
                raise ValueError(f"Unsupported type: {type(value)}")

        output = wan_model(**kwargs_cloned)
        output = torch.nn.functional.silu(output)
        output = output.sum(dim=tuple(range(1, output.ndim)))

        loss = torch.nn.functional.mse_loss(output, torch.zeros_like(output))
        loss.backward()

    timestep = [[
        997.7805, 853.9290, 853.9290, 924.8179, 924.8179, 801.6499, 801.6499,
        878.4186, 878.4186, 829.4078, 829.4078, 838.2740, 838.2740,
    ]]
    kwargs = dict(
        x=torch.randn(1, 16, 13, 44, 80, dtype=torch.bfloat16, device="cuda"),
        context=torch.randn(1, 512, 4096, dtype=torch.bfloat16, device="cuda"),
        timestep=torch.tensor(timestep, dtype=torch.float32, device="cuda"),
        seq_len=11440,
        clean_x=torch.randn(1, 16, 13, 44, 80, dtype=torch.bfloat16, device="cuda"),
        y=torch.randn(1, 20, 13, 44, 80, dtype=torch.bfloat16, device="cuda"),
        clip_feature=torch.randn(1, 257, 1280, dtype=torch.bfloat16, device="cuda"),
    )

    start_time = time.time()
    for _ in range(n_warmup_iters):
        train_func(**kwargs)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Warmup done, took {end_time - start_time:.2f} seconds")

    enable_profiling = False

    if enable_profiling:
        profile_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
        )
    else:
        profile_context = contextlib.nullcontext()

    start_time = time.time()

    with profile_context as prof:
        for _ in range(n_iters):
            train_func(**kwargs)
    torch.cuda.synchronize()

    end_time = time.time()

    if enable_profiling:
        prof.export_chrome_trace(f"benchmark/wan_model.json.gz")

    time_per_iter = (end_time - start_time) / n_iters
    print(f"Wan Model time per iter: {time_per_iter:.2f} sec")

def benchmark_single_attention(q_len, kv_len, n_heads, head_dim, backend, compiled, n_iters, n_warmup_iters, n_ops):
    attn_module = AttentionModule(num_heads=n_heads, head_dim=head_dim, backend=backend)
    attn_module.train()

    def forward_loop(q, k, v):
        attn_input = q
        for _ in range(n_ops):
            attn_input = attn_module(attn_input, k, v)
            attn_input = torch.nn.functional.silu(attn_input)
        return attn_input

    if compiled:
        forward_loop = torch.compile(forward_loop, fullgraph=True, mode="reduce-overhead")

    def train_loop(q, k, v):
        q = q.detach().clone().requires_grad_(True)
        k = k.detach().clone().requires_grad_(True)
        v = v.detach().clone().requires_grad_(True)
        output = forward_loop(q, k, v)
        output = output.sum(dim=tuple(range(1, output.ndim)))
        loss = torch.nn.functional.mse_loss(output, torch.zeros_like(output))
        loss.backward()

    q = torch.randn(1, q_len, n_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, kv_len, n_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, kv_len, n_heads, head_dim, dtype=torch.bfloat16, device="cuda")

    for _ in range(n_warmup_iters):
        train_loop(q, k, v)
    torch.cuda.synchronize()

    enable_profiling = False

    if enable_profiling:
        profile_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
        )
    else:
        profile_context = contextlib.nullcontext()

    start_time = time.time()

    with profile_context as prof:
        for _ in range(n_iters):
            train_loop(q, k, v)
    torch.cuda.synchronize()
    
    end_time = time.time()

    if enable_profiling:
        prof.export_chrome_trace(f"benchmark/{backend}_{kv_len}.json.gz")

    time_per_iter = (end_time - start_time) / n_iters / n_ops
    # assert torch.allclose(out, out[0], rtol=1e-4, atol=1e-4)

    print(f"Backend: {backend}, compiled: {compiled}, q_len: {q_len}, kv_len: {kv_len}, time per iter: {time_per_iter * 1e6:.2f} us")

def benchmark_attention(n_iters, n_warmup_iters):
    n_ops = 100
    n_heads = 40
    head_dim = 128
    q_len = 1760
    kv_lens = [q_len // 2 + q_len * num_blocks for num_blocks in range(1, 7)]
    compiled_list = [True]
    backend_list = ["TE"]

    for compiled in compiled_list:
        for backend in backend_list:
            for kv_len in kv_lens:
                benchmark_single_attention(
                    q_len=q_len,
                    kv_len=kv_len,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    backend=backend,
                    compiled=compiled,
                    n_iters=n_iters,
                    n_warmup_iters=n_warmup_iters,
                    n_ops=n_ops,
                )


if __name__ == "__main__":
    n_iters = 3
    n_warmup_iters = 3
    # benchmark_attention(n_iters=n_iters, n_warmup_iters=n_warmup_iters)
    benchmark_wan_model(n_iters=n_iters, n_warmup_iters=n_warmup_iters)
