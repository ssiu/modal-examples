import os
import subprocess
import sys
from pathlib import Path

import modal


CUDA_VERSION = "12.4.0"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu22.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

OUTPUT_VOLUME_NAME = "flash-attn-outputs"
OUTPUT_PATH = Path("/outputs")
WORKDIR = Path("/root/ncu-minimal")

app = modal.App("test-ncu-minimal")

outputs = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.11")
    .apt_install("sudo")
    .pip_install("torch")
)


def run_command(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command))
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )


def _write_workload_script(script_path: Path) -> None:
    script_path.write_text(
        """
import json
import torch

torch.manual_seed(0)
assert torch.cuda.is_available(), "CUDA is required for this NCU example"

device = "cuda"
dtype = torch.float16

a = torch.randn(4096, 4096, device=device, dtype=dtype)
b = torch.randn(4096, 4096, device=device, dtype=dtype)

for _ in range(5):
    out = a @ b
torch.cuda.synchronize()

summary = {
    "device_name": torch.cuda.get_device_name(0),
    "cuda_version": torch.version.cuda,
    "torch_version": torch.__version__,
    "output_shape": list(out.shape),
    "output_dtype": str(out.dtype),
}
print("WORKLOAD_SUMMARY=" + json.dumps(summary, sort_keys=True))
""".strip()
        + "\n"
    )


@app.function(
    image=image,
    gpu="T4",
    cpu=8.0,
    memory=32768,
    timeout=1800,
    volumes={OUTPUT_PATH: outputs},
)
def run_ncu(
    set_name: str = "full",
) -> dict:
    WORKDIR.mkdir(parents=True, exist_ok=True)

    workload_script = WORKDIR / "workload.py"
    _write_workload_script(workload_script)

    ncu_report = OUTPUT_PATH / "ncu_minimal_report"
    ncu_stdout = OUTPUT_PATH / "ncu_minimal_stdout.txt"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    command = [
        "sudo",
        "/usr/local/cuda/bin/ncu",
        "--set",
        set_name,
        "--target-processes",
        "all",
        "--kernel-name-base",
        "demangled",
        "--force-overwrite",
        "-o",
        str(ncu_report),
        sys.executable,
        str(workload_script),
    ]

    print(f"Python executable: {sys.executable}")
    run_command(["nvidia-smi"])
    run_command(["ls", "-l", "/usr/local/cuda/bin/ncu"])

    with ncu_stdout.open("w") as log_file:
        process = subprocess.run(
            command,
            cwd=str(WORKDIR),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    stdout_text = ncu_stdout.read_text()
    print(stdout_text)

    report_files = sorted(p.name for p in OUTPUT_PATH.glob("ncu_minimal_report*"))
    outputs.commit()

    return {
        "returncode": process.returncode,
        "report_prefix": str(ncu_report),
        "report_files": report_files,
        "stdout_path": str(ncu_stdout),
        "stdout_tail": stdout_text[-4000:],
    }


@app.local_entrypoint()
def main(set_name: str = "full"):
    result = run_ncu.remote(set_name=set_name)
    print(result)
