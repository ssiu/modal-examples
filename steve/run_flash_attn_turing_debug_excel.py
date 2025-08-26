import modal
from modal import App, Image
import subprocess
import sys
from pathlib import Path

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
os = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{os}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git", "curl", "sudo")
    .pip_install(
        "torch",
        "setuptools",
        "ninja",
        "wheel",
        "pytest",
        "pandas",
        "numpy",
        "openpyxl"
    )
)

def execute_command(command: str):
    command_args = command.split(" ")
    print(f"{command_args = }")
    subprocess.run(command_args, stdout=sys.stdout, stderr=subprocess.STDOUT)


app = App(image=image)

VOLUME_NAME = "flash-attn-outputs"
outputs = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
OUTPUTS_PATH = Path("/outputs")

#@app.function(gpu="T4", container_idle_timeout=60, cpu=16.0, memory=65536)
@app.function(gpu="T4", cpu=16.0, memory=65536, volumes={OUTPUTS_PATH: outputs})
def run_extension():

    # flash attention develop
    import os
    import torch
    execute_command("git clone https://github.com/ssiu/flash-attention-turing.git")
    os.chdir("flash-attention-turing")
    execute_command("git checkout add_tests")
    #execute_command("git submodule update --init --recursive")
    #execute_command("git clone https://github.com/NVIDIA/cutlass.git")
    os.environ["CXX"] = "g++"
    os.environ["CC"] = "gcc"
    # execute_command("ls /root/flash-attention-turing/csrc/")
    # execute_command("ls /root/flash-attention-turing/csrc/cutlass/")
    #execute_command("ls /root/flash-attention-turing/csrc/cutlass/include/cute/tensor.hpp")
    execute_command("pip install -v .")
    print(torch.version.cuda)
    print(torch.__version__)
    # execute_command("compute-sanitizer python utils/test_flash_backward.py 1 128 1 128")

    import torch.nn.functional as F
    from torch.nn.attention import bias

    import numpy as np
    import pandas as pd

    from flash_attn_turing import (
        fwd,
        bwd
    )

    def causal_lower_right(seqlen_q, seqlen_k, device):
        """
        Build a lower-right causal mask for seqlen_q >= seqlen_k.
        Returns a boolean tensor of shape (seqlen_q, seqlen_k)
        """
        diagonal_offset = seqlen_k - seqlen_q
        mask = torch.tril(
            torch.ones((seqlen_q, seqlen_k), dtype=torch.bool, device=device),
            diagonal=diagonal_offset,
        )
        return mask  # bool tensor (seqlen_q, seqlen_k)

    def attention_ref(query, key, value, d_output, causal=False):
        query_torch = query.permute(0, 2, 1, 3).contiguous().clone()
        key_torch = key.permute(0, 2, 1, 3).contiguous().clone()
        value_torch = value.permute(0, 2, 1, 3).contiguous().clone()

        batch_size, nheads, seqlen_q, d = query_torch.size()
        _, _, seqlen_k, _ = key_torch.size()

        scores = torch.matmul(query_torch, key_torch.transpose(-2, -1)) / (d ** 0.5)

        if causal:
            # attn_mask = torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=attn.device).tril(diagonal=0)

            attn_mask = causal_lower_right(seqlen_q, seqlen_k, device=scores.device)
            attn_mask = attn_mask.view(1, 1, seqlen_q, seqlen_k)  # broadcast over batch & heads
            scores = scores.masked_fill(~attn_mask, float("-inf"))
            # mask_val = torch.finfo(attn.dtype).min / 2
            # attn = attn.masked_fill(~attn_mask, mask_val)
            # attn = attn.masked_fill(~attn_mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = torch.where(attn.isnan(), torch.zeros_like(attn), attn)

        # if causal:
        #     for i in range(32):
        #         print(f"attention matrix {attn[0, 0, 0, i]}")
        output_torch = torch.matmul(attn, value_torch)
        output_torch = output_torch.permute(0, 2, 1, 3).contiguous().clone()

        return output_torch


    device = "cuda"

    batch_size = 4
    nheads = 4
    seqlen_q = 256
    seqlen_k = 128
    d = 128
    causal = True
    dtype = torch.float16
    query = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    key = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype)
    value = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype)
    d_output = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)

    output, l = fwd(query, key, value, causal)

    output_torch = attention_ref(query, key, value, d_output, causal)

    output_max_diff = (output - output_torch).abs().max().item()
    output_mean_diff = (output - output_torch).abs().mean().item()
    idx = torch.argmax((output - output_torch).abs())
    coords = np.unravel_index(idx.item(), (output - output_torch).shape)

    output_max = output.abs().max().item()
    output_mean = output.abs().mean().item()

    output_torch_max = output_torch.abs().max().item()
    output_torch_mean = output_torch.abs().mean().item()

    print(f"seqlen_q = {seqlen_q}, seqlen_k = {seqlen_k}, d = {d}, causal = {causal}")

    print(f"output max: {output_max}")
    print(f"output mean: {output_mean}")
    print(f"coordinate: {coords}")

    print(f"output_torch max: {output_torch_max}")
    print(f"output_torch mean: {output_torch_mean}")

    print(f"output max diff: {output_max_diff}")
    print(f"output mean diff: {output_mean_diff}")


    for i in range(batch_size):
        for j in range(nheads):
            output_subset = output[i, :, j, :].cpu()
            output_torch_subset = output_torch[i, :, j, :].cpu()
            output_subset_flat = output_subset.flatten()
            output_torch_subset_flat = output_torch_subset.flatten()
            output_diff = output_subset_flat - output_torch_subset_flat
            data = {
                'output': output_subset_flat.numpy(),
                'output_torch': output_torch_subset_flat.numpy(),
                'diff': output_diff.numpy()
            }
            df = pd.DataFrame(data)
            df.to_excel(OUTPUTS_PATH /f"seqlen_q_{seqlen_q}_seqlen_k_{seqlen_k}_batch_{i}_nhead_{j}.xlsx", index=False)
    # if seqlen_q == 1024 and d == 128:
    #     for i in range(32):
    #         print(32*i, output[0,32*i,0,127], output_torch[0,32*i,0,127])

    #
    # with open(OUTPUTS_PATH /"test.txt", "w") as f:
    #     f.write(f"{output_max_diff}, {output_mean_diff}")

    # assert output_max_diff <= 1e-2
    # assert output_mean_diff <= 1e-4