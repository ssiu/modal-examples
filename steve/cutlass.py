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
)

app = App(image=image)

def execute_command(command: str):
    command_args = command.split(" ")
    print(f"{command_args = }")
    subprocess.run(command_args, stdout=sys.stdout, stderr=subprocess.STDOUT)


@app.function(gpu="T4", container_idle_timeout=60, cpu=8.0, memory=32768)
def f():
    import os
    execute_command("git clone https://github.com/ssiu/cuda.git")
    #execute_command("pwd")

    # # flash attention turing
    # os.chdir("cuda/flash_attn_turing")
    # execute_command("git clone https://github.com/NVIDIA/cutlass.git")
    # execute_command("make launch_flash_attn.o")
    # execute_command("./launch_flash_attn.o")

    # cutlass learning
    os.chdir("cuda/cutlass")
    execute_command("git clone https://github.com/NVIDIA/cutlass.git")
    execute_command("make tensor_op.o")
    execute_command("./tensor_op.o")


    # # gemm experiments
    # os.chdir("cuda/gemm")
    # execute_command("git clone https://github.com/NVIDIA/cutlass.git")
    # execute_command("make launch_sm75_gemm.o")
    # execute_command("./launch_sm75_gemm.o 1024 1024 1024")




