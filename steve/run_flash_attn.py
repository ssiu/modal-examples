import modal
from modal import App, Image

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
os = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{os}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
    .pip_install(  # required to build flash-attn
        "ninja",
        "packaging",
        "wheel",
        "torch",
    )
    .run_commands(  # add flash-attn
        "pip install flash-attn==2.5.8 --no-build-isolation"
    )
)

app = App(image=image)

@app.function(gpu="A10G", image=image)
def run_flash_attn():
    import torch
    from flash_attn import flash_attn_func

    batch_size, seqlen, nheads, headdim, nheads_k = 2, 4, 3, 16, 3

    q = torch.randn(batch_size, seqlen, nheads, headdim, dtype=torch.float16).to("cuda")
    k = torch.randn(batch_size, seqlen, nheads_k, headdim, dtype=torch.float16).to("cuda")
    v = torch.randn(batch_size, seqlen, nheads_k, headdim, dtype=torch.float16).to("cuda")

    out = flash_attn_func(q, k, v)
    print(out)
    assert out.shape == (batch_size, seqlen, nheads, headdim)