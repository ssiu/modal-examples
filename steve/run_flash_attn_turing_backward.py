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
        "pytest"
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

    #execute_command("pytest -s test_flash_attn.py::test_flash_attn_fwd_id_matrix")
    execute_command("pytest -s test_flash_attn.py::test_flash_attn_fwd")