import torch
import glob
import shutil
import os

input_path = "/home/rkumar/LLaVA-brdcst/checkpoints/llava-v1.5-7b-pretrain-hf"
output_path = "/home/rkumar/LLaVA-brdcst/checkpoints/llava-v1.5-projectors/"

os.makedirs(output_path, exist_ok=True)

state_dict = torch.load(f"{input_path}/mm_projector.bin")
original_path = "/home/rkumar/LLaVA-brdcst/checkpoints/llava-v1.5-lora-merged-run2/"

for i in range(1, 3):
    filename = f"pytorch_model-0000{i}-of-00002.bin"
    original = torch.load(original_path + filename)

    for k in state_dict.keys():
        if k in original:
            print(filename, k)
            original[k] = state_dict[k]

    torch.save(original, output_path + filename)

for filename in glob.glob(original_path + "*.json"):
    shutil.copy(filename, output_path)

for filename in glob.glob(original_path + "*.model"):
    shutil.copy(filename, output_path)