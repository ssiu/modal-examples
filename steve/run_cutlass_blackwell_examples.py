import modal
from modal import App, Image
import subprocess
import sys
from pathlib import Path

cuda_version = "12.8.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
os = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{os}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install("git", "curl", "sudo")
    .uv_pip_install(
        "torch",
        "setuptools",
        "ninja",
        "wheel",
        "pytest",
        "pandas",
        "openpyxl",
        # "cuda-python",
        # "nvidia-cutlass-dsl"        
    )
)

def execute_command(command: str):
    command_args = command.split(" ")
    print(f"{command_args = }")
    subprocess.run(command_args, stdout=sys.stdout, stderr=subprocess.STDOUT)


app = App(image=image)

# VOLUME_NAME = "flash-attn-outputs"
# outputs = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
# OUTPUTS_PATH = Path("/outputs")

#@app.function(gpu="T4", container_idle_timeout=60, cpu=16.0, memory=65536)
# @app.function(gpu="B200", cpu=16.0, memory=65536, volumes={OUTPUTS_PATH: outputs}, timeout=3600)
@app.function(gpu="B200", timeout=3600)
def run_extension():

    # flash attention develop
    import os
    import torch
    # execute_command("git clone https://github.com/ssiu/flash-attention-turing.git")
    # os.chdir("flash-attention-turing")
    # execute_command("git checkout arbitrary_seqlen")
    #execute_command("git submodule update --init --recursive")
    #execute_command("git clone https://github.com/NVIDIA/cutlass.git")
    os.environ["CXX"] = "g++"
    os.environ["CC"] = "gcc"
    print(torch.version.cuda)
    print(torch.__version__)
    print(torch.cuda.get_device_name(0))

    # execute_command("ls -l /usr/local/")
    execute_command("nvcc --version")
    execute_command("nvidia-smi")
    execute_command("pip install cuda-python==12.9")
    execute_command("pip install nvidia-cutlass-dsl")
    execute_command("git clone https://github.com/ssiu/cutlass.git")

    os.chdir("cutlass")
    # execute_command("python examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_0.py")

    # execute_command("python examples/python/CuTeDSL/blackwell/dense_gemm.py --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32 --mma_tiler_mn 256,128 --cluster_shape_mn 2,1 --mnkl 8192,8192,8192,1 --use_tma_store --use_2cta_instrs")
    execute_command("python examples/python/CuTeDSL/blackwell/dense_gemm.py --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32 --mma_tiler_mn 128,128 --cluster_shape_mn 2,1 --mnkl 384,7168,2304,1 --use_tma_store")

