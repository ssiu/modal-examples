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
@app.function(gpu="T4", cpu=8, memory=32768, volumes={OUTPUTS_PATH: outputs}, timeout=6000)
def run_extension():

    # flash attention develop
    import os
    import torch
    execute_command("git clone https://github.com/ssiu/flash-attention-turing.git")
    os.chdir("flash-attention-turing")
    execute_command("git checkout final_clean_up")

    subprocess.run(
    ["python", "-c", "import torch; print(torch.__version__)"],
    stdout=sys.stdout,
    stderr=subprocess.STDOUT,
    check=True,
    )

    # execute_command("git clone https://github.com/ssiu/flash-attention")
    # os.chdir("flash-attention")
    # execute_command("git checkout turing")
    # os.chdir("turing")
    

    os.environ["CXX"] = "g++"
    os.environ["CC"] = "gcc"

    execute_command("pip install -v .")
    print(torch.version.cuda)
    print(torch.__version__)


    execute_command("pytest -s -vv --tb=short -rfE test_flash_attn.py::test_flash_attn")
    execute_command("pytest -s -vv --tb=short -rfE test_flash_attn.py::test_flash_attn_varlen")
    # execute_command("pytest -s -vv --tb=short -rfE test_flash_attn.py::test_flash_attn_qkv")
    # execute_command("pytest -s -vv --tb=short -rfE test_flash_attn.py::test_flash_attn_kv")
    # execute_command("pytest -s -vv --tb=short -rfE test_flash_attn.py::test_flash_attn_varlen_qkv")
    # execute_command("pytest -s -vv --tb=short -rfE test_flash_attn.py::test_flash_attn_varlen_kv")

    # execute_command("pytest -vv --tb=short -rfE test_flash_attn.py::test_flash_attn")
    # execute_command("pytest -vv --tb=short -rfE test_flash_attn.py::test_flash_attn_varlen")
    # execute_command("pytest -vv --tb=short -rfE test_flash_attn.py::test_flash_attn_qkvpacked")
    # execute_command("pytest -vv --tb=short -rfE test_flash_attn.py::test_flash_attn_kvpacked")
    # execute_command("pytest -vv --tb=short -rfE test_flash_attn.py::test_flash_attn_varlen_qkvpacked")
    # execute_command("pytest -vv --tb=short -rfE test_flash_attn.py::test_flash_attn_varlen_kvpacked")


    # execute_command("pytest -q --disable-warnings --tb=short test_flash_attn.py::test_flash_attn_qkvpacked")
    # execute_command("pytest -q --disable-warnings --tb=short test_flash_attn.py::test_flash_attn_kvpacked")
    # execute_command("pytest -q --disable-warnings --tb=short test_flash_attn.py::test_flash_attn_varlen_qkvpacked")
    # execute_command("pytest -q --disable-warnings --tb=short test_flash_attn.py::test_flash_attn_varlen_kvpacked")
    outputs.commit()
