# DreamZero: World Action Models Are Zero-Shot Policies

DreamZero is a World Action Model that jointly predicts actions and videos, achieving strong zero-shot performance on unseen tasks. This release package contains everything needed to load a pretrained DreamZero model and run distributed inference via a WebSocket server.

## Testing Out DreamZero in Simulation with API
We provide an inference script that directly evaluates a hosted DreamZero-DROID policy on [`sim_evals`](https://github.com/arhanjain/sim-evals). To test out the policy, first request access to the API via this form (insert link). Then, follow these instructions to install [`sim_evals`](https://github.com/arhanjain/sim-evals) and launch evaluation.

```bash
# Clone repository
git clone --recurse-submodules https://github.com/arhanjain/sim-evals.git
cd sim-evals

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Activate uv environment
uv sync
source .venv/bin/activate

# [Optional] update pytorch versions
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129

# Download assets (may need to export HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> first)
uvx hf download owhan/DROID-sim-environments --repo-type dataset --local-dir assets

# Run eval script
cd ..
python droid_sim_evals/run_eval.py --host <API_HOST> --port <API_PORT> 
```

The outputs are saved in `runs` directory.


## Quick Start

### Prerequisites

- **Python**: 3.11
- **Hardware**: Multi-GPU setup (tested on GB200, H100)
  - Minimum: 2 GPUs for distributed inference
- **CUDA**: Compatible GPU with CUDA support

### Installation

1. **Create conda environment:**
```bash
conda create -n dreamzero python=3.11
conda activate dreamzero
```

2. **Install dependencies (PyTorch 2.8+ with CUDA 12.9+):**
```bash
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129
```

3. **Install flash attention:**
```bash
MAX_JOBS=8 pip install --no-build-isolation flash-attn
```

4. **[GB200 ONLY, SKIP FOR H100] Install Transformer Engine:**
```bash
pip install --no-build-isolation transformer_engine[pytorch]
```

## Running the Inference Server

### Command Overview

The inference server uses PyTorch distributed training utilities to parallelize the model across multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 socket_test_optimized_AR.py --port 5000 --enable-dit-cache --model-path /path/to/your/checkpoint
```

To verify the server is working, run a test client. The first few inferences will take a few minutes to warm up. After warming up, inference takes ~0.6s on GB200 and ~3s on H100.

```
python test_client_AR.py --port 5000
```

### Command-line Arguments

- `--port`: Port number for the WebSocket server (default: 8000)
- `--model-path`: Path to the pretrained model checkpoint directory
- `--enable-dit-cache`: Enable caching in DiT layers for faster inference (recommended)
- `--max-chunk-size`: Override max_chunk_size for inference (optional)
- `--timeout-seconds`: Server timeout in seconds (default: 50000)
- `--index`: Index for output directory naming (default: 0)


### Performance

- **Inference Time** (server-side, 2x GB200): ~1.5 seconds per action prediction
- **Distributed Setup**: Rank 0 handles WebSocket communication, all ranks participate in model inference


### Output

The server saves:
- **Videos**: Generated video predictions as MP4 files in `{model_path}/real_world_eval_gen_{date}_{index}/{checkpoint_name}/`
- **Input observations**: Saved per message in `{output_dir}/inputs/{msg_index}_{timestamp}/`


## Citation

If you use DreamZero in your research, please cite:

```bibtex
@article{dreamzero2025,
  title={DreamZero: World Action Models Are Zero-Shot Policies},
  author={GROOT-Dreams Team},
  journal={arXiv preprint},
  year={2025}
}
```

## License

[License Here]

## Support

For issues and questions:
- Check the troubleshooting section above
- Review server logs for detailed error messages
- Verify your checkpoint is compatible with this release
