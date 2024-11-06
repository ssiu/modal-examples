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
    import subprocess
    import os
    #execute_command("sudo ls /root")
    execute_command("pwd")
    execute_command("git clone https://github.com/NVIDIA/cutlass.git")
    execute_command("git clone https://github.com/ssiu/cuda.git")
    os.chdir("cuda/cutlass")
    execute_command("nvcc -lineinfo -o sm75_gemm_16x8x8.o -std=c++17 -arch=sm_75 -I/root/cutlass/include -I/root/cutlass/tools/util/include sm75_gemm_16x8x8.cu")
    execute_command("./sm75_gemm_16x8x8.o")
    # execute_command("./dev/download_starter_pack.sh")
    # execute_command("make test_gpt2fp32cu")
    # execute_command("./test_gpt2fp32cu")
    # #execute_command("which ncu")
    # execute_command("sudo ncu ./test_gpt2fp32cu")
    # # execute_command("sudo /usr/local/cuda/bin/ncu -f --print-level info --target-processes all --set full --import-source on -o test ./test_gpt2fp32cu")
    # # # execute_command("nsys profile -o test --stats=true -t cuda,osrt,nvtx --force-overwrite=true ./test_gpt2fp32cu")

