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
        "setuptools",
        "ninja",
        "wheel"
    )
)

def execute_command(command: str):
    command_args = command.split(" ")
    print(f"{command_args = }")
    subprocess.run(command_args, stdout=sys.stdout, stderr=subprocess.STDOUT)


app = App(image=image)

#@app.function(gpu="T4", container_idle_timeout=60, cpu=16.0, memory=65536)
@app.function(gpu="T4", scaledown_window=60, cpu=8.0, memory=32768)
def run_extension():
    # official pytorch extension example
    # https://github.com/pytorch/extension-cpp
    # import os
    # execute_command("git clone https://github.com/pytorch/extension-cpp.git")
    # os.chdir("extension-cpp")
    # #execute_command("EXPORT CXX=g++")
    # execute_command("pip install numpy expecttest")
    # os.environ["CXX"] = "g++"
    # os.environ["CC"] = "gcc"
    # execute_command("pip install .")
    # execute_command("python test/test_extension.py")


    # steve's pytorch extension example
    # https://github.com/ssiu/extension-cpp

    # import os
    # execute_command("git clone https://github.com/ssiu/extension-cpp.git")
    # os.chdir("extension-cpp")
    # #execute_command("EXPORT CXX=g++")
    # execute_command("pip install numpy expecttest")
    # os.environ["CXX"] = "g++"
    # os.environ["CC"] = "gcc"
    # execute_command("pip install .")
    # execute_command("python test/test_gemm.py")

    # simple example for testing 2 tensor outputs
    import os
    execute_command("git clone https://github.com/ssiu/cuda.git")
    os.chdir("cuda/cpp_extension_simple")
    os.environ["CXX"] = "g++"
    os.environ["CC"] = "gcc"
    execute_command("pip install .")
    execute_command("python test.py")

    # # simple example from chatgpt
    # import os
    # execute_command("git clone https://github.com/ssiu/cuda.git")
    # os.chdir("cuda/cpp_extension_test")
    # execute_command("git clone https://github.com/NVIDIA/cutlass.git")
    # os.environ["CXX"] = "g++"
    # os.environ["CC"] = "gcc"
    #
    # execute_command("pip install .")
    # execute_command("python test.py")

    # flash attention
    # import os
    # execute_command("git clone https://github.com/ssiu/cuda.git")
    # os.chdir("cuda/flash_attn_turing")
    # execute_command("git clone https://github.com/NVIDIA/cutlass.git")
    # os.environ["CXX"] = "g++"
    # os.environ["CC"] = "gcc"
    #
    #
    # execute_command("pip install .")
    # execute_command("python test_flash.py 4 4096 32 128")


