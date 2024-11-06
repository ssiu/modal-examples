# https://huggingface.co/Qwen/Qwen2-7B-Instruct
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
        "transformers>=4.40.0",
        "accelerate"
    )
    .run_commands(  # add flash-attn
        "pip install flash-attn --no-build-isolation"
    )
)

app = App(name="qwen2", image=image)

@app.function(gpu="A10G", image=image, cpu=8.0, memory=32768)
def run_qwen2():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2-7B-Instruct"
    device = "cuda"  # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(response)


