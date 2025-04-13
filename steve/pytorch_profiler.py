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

@app.function(gpu="A100", container_idle_timeout=60, cpu=8.0, memory=32768, volumes={"/data": vol})
def run_profiler():
    # import torch
    # import torchvision.models as models
    # from torch.profiler import profile, record_function, ProfilerActivity
    #
    # model = models.resnet18()
    # inputs = torch.randn(5, 3, 224, 224)
    #
    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    #     with record_function("model_inference"):
    #         model(inputs)
    #
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # import torch
    # from torch.profiler import profile, record_function, ProfilerActivity
    #
    # A = torch.rand(1024, 1024, device='cuda', dtype=torch.float16)
    # B = torch.rand(1024, 1024, device='cuda', dtype=torch.float16)
    #
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     C = torch.matmul(A, B)
    #
    # print(C.dtype)
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    import torch
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel
    from torch._C import _SDPBackend as SDPBackend
    from torch.profiler import profile, record_function, ProfilerActivity
    # (batch_size, num_heads, seq_len, head_dim)
    query = torch.rand(32, 8, 16384, 64, dtype=torch.float16, device="cuda")
    key = torch.rand(32, 8, 16384, 64, dtype=torch.float16, device="cuda")
    value = torch.rand(32, 8, 16384, 64, dtype=torch.float16, device="cuda")
    #

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        # with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
        #     output =  F.scaled_dot_product_attention(query, key, value)
        output = F.scaled_dot_product_attention(query, key, value)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    with open("/data/profiler_output.txt", "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))

    # with open("/data/profiler_output_full.txt", "w") as f:
    #     for event in prof.key_averages():
    #         # Convert each profiling event to a string
    #         f.write(str(event) + "\n")
    #
    # vol.commit()