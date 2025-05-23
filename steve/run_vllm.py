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
        "torch==2.5.1",
        "torchvision",
        "vllm"
    )
)

def execute_command(command: str):
    command_args = command.split(" ")
    print(f"{command_args = }")
    subprocess.run(command_args, stdout=sys.stdout, stderr=subprocess.STDOUT)


vol = modal.Volume.from_name("steve")

app = App(image=image)

@app.function(gpu="A100", scaledown_window=60, cpu=8.0, memory=32768, volumes={"/data": vol})
def run_vllm():

    import os
    from vllm import LLM, SamplingParams
    from torch.profiler import profile, record_function, ProfilerActivity

    os.environ["HF_TOKEN"] = "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"

#     prompt = """
#     WESTMORELAND
#
# O that we now had here
# But one ten thousand of those men in England
# That do no work to-day!
#
# KING HENRY V
#
# What’s he that wishes so?
# My cousin Westmoreland? No, my fair cousin:
# If we are mark’d to die, we are enow
# To do our country loss; and if to live,
# The fewer men, the greater share of honour.
# God’s will! I pray thee, wish not one man more.
# By Jove, I am not covetous for gold,
# Nor care I who doth feed upon my cost;
# It yearns me not if men my garments wear;
# Such outward things dwell not in my desires:
# But if it be a sin to covet honour,
# I am the most offending soul alive.
# No, faith, my coz, wish not a man from England:
# God’s peace! I would not lose so great an honour
# As one man more, methinks, would share from me
# For the best hope I have. O, do not wish one more!
# Rather proclaim it, Westmoreland, through my host,
# That he which hath no stomach to this fight,
# Let him depart; his passport shall be made
# And crowns for convoy put into his purse:
# We would not die in that man’s company
# That fears his fellowship to die with us.
# This day is called the feast of Crispian:
# He that outlives this day, and comes safe home,
# Will stand a tip-toe when the day is named,
# And rouse him at the name of Crispian.
# He that shall live this day, and see old age,
# Will yearly on the vigil feast his neighbours,
# And say ‘To-morrow is Saint Crispian:’
# Then will he strip his sleeve and show his scars.
# And say ‘These wounds I had on Crispin’s day.’
# Old men forget: yet all shall be forgot,
# But he’ll remember with advantages
# What feats he did that day: then shall our names.
# Familiar in his mouth as household words
# Harry the king, Bedford and Exeter,
# Warwick and Talbot, Salisbury and Gloucester,
# Be in their flowing cups freshly remember’d.
# This story shall the good man teach his son;
# And Crispin Crispian shall ne’er go by,
# From this day to the ending of the world,
# But we in it shall be remember’d;
# We few, we happy few, we band of brothers;
# For he to-day that sheds his blood with me
# Shall be my brother; be he ne’er so vile,
# This day shall gentle his condition:
# And gentlemen in England now a-bed
# Shall think themselves accursed they were not here,
# And hold their manhoods cheap whiles any speaks
# That fought with us upon Saint Crispin’s day.
#     """

    # Sample prompts.
    prompts = [
        # prompt
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1000)

    # Create an LLM.
    llm = LLM(model="meta-llama/Llama-3.2-3B-Instruct", dtype="float16", max_model_len=54864)
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    # for output in outputs:
    #     prompt = output.prompt
    #     generated_text = output.outputs[0].text
    #     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        outputs = llm.generate(prompts, sampling_params)
        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
