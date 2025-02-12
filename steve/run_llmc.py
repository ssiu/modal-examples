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


@app.function(gpu="A10G", container_idle_timeout=300, cpu=8.0, memory=32768)
def f():
    import subprocess
    import os
    execute_command("git clone https://github.com/karpathy/llm.c.git")
    os.chdir("llm.c")
    execute_command("chmod u+x ./dev/download_starter_pack.sh")
    execute_command("./dev/download_starter_pack.sh")
    execute_command("make test_gpt2fp32cu")
    execute_command("sudo /usr/local/cuda/bin/ncu ./test_gpt2fp32cu")
