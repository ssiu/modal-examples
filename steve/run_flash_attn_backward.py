import modal
from modal import App, Image
import subprocess
import sys

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
os = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{os}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git", "curl", "sudo")
    .pip_install(
        "torch",
        "torchvision"
    )
)

vol = modal.Volume.from_name("steve")

app = App(image=image)

@app.function(gpu="A100", cpu=8.0, memory=32768, volumes={"/data": vol})
def run_profiler():

    import torch
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel
    from torch._C import _SDPBackend as SDPBackend
    from torch.profiler import profile, record_function, ProfilerActivity


    # (batch_size, num_heads, seq_len, head_dim)

    batch_size, num_heads, seq_len, head_dim = 4, 32, 4096, 128
    Q = torch.rand(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
    K = torch.rand(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
    V = torch.rand(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
    dO = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32, device="cuda")
    Q.requires_grad = True
    K.requires_grad = True
    V.requires_grad = True

    #

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        # with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
        #     output =  F.scaled_dot_product_attention(query, key, value)
        output = F.scaled_dot_product_attention(Q, K, V)
        output.backward(dO)
        print(Q.grad.dtype)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    with open("/data/profiler_output.txt", "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))

