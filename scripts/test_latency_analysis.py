from model_chunking.models.qwen2 import Qwen2ChunkingForCausalLM, Qwen2ChunkingConfig, Qwen2Tokenizer, Qwen2Config, infer_chunking_device_map, infer_chunking_device_map_from_config
from transformers import Qwen2ForCausalLM
from tqdm.auto import tqdm

model_name = "Qwen/Qwen2.5-7B-Instruct"
num_gpu = 2
batch_size=16

# # pre-chunking: []; chunking [[0, 1, ..., 11], [12, 13, ..., 23]], post-chunking: []
config = Qwen2ChunkingConfig.from_pretrained(model_name, num_layers_per_chunk=6, num_chunks=2, chunking_mode="sequential_with_shared_start_and_end", aggregation_mode="mean", use_adapters=False, distillation=True)

# config = Qwen2ChunkingConfig.from_pretrained(model_name, num_layers_per_chunk=12, num_chunks=2, chunking_mode="uniform_with_shared_start_and_end", aggregation_mode="mean", use_adapters=False)

# # pre-chunking: [0,1,...,3]; chunking [[4, 6, ..., 18], [5, 7, ..., 19]], post-chunking: [20, ..., 23]
# config = Qwen2ChunkingConfig.from_pretrained(model_name, num_layers_per_chunk=8, num_chunks=2, chunking_mode="uniform_with_shared_start_and_end", aggregation_mode="mean", use_adapters=False)

# pre-chunking: [0,1,...,7]; chunking [[8, 10, ..., 14], [9, 11, ..., 15]], post-chunking: [16, ..., 23]
# config = Qwen2ChunkingConfig.from_pretrained(model_name, num_layers_per_chunk=4, num_chunks=2, chunking_mode="uniform_with_shared_start_and_end", aggregation_mode="mean", use_adapters=False)

# # pre-chunking: [0,1,...,8]; chunking [[9, 11, 13], [10, 12, 14]], post-chunking: [15, ..., 23]
# config = Qwen2ChunkingConfig.from_pretrained(model_name, num_layers_per_chunk=3, num_chunks=2, chunking_mode="uniform_with_shared_start_and_end", aggregation_mode="mean", use_adapters=False)

# # # pre-chunking: [0,1,...,9]; chunking [[10, 12], [11, 13]], post-chunking: [14, ..., 23]
# config = Qwen2ChunkingConfig.from_pretrained(model_name, num_layers_per_chunk=2, num_chunks=2, chunking_mode="uniform_with_shared_start_and_end", aggregation_mode="mlp", use_adapters=True)

# # # pre-chunking: [0,1,..., 11,]; chunking [], post-chunking: [12, ..., 23] # same as no chunking
# config = Qwen2ChunkingConfig.from_pretrained(model_name, num_layers_per_chunk=0, num_chunks=None, chunking_mode="uniform_with_shared_start_and_end", aggregation_mode="mean", use_adapters=False)


# config = Qwen2ChunkingConfig.from_pretrained(model_name, num_layers_per_chunk=6, num_chunks=4, chunking_mode="uniform_with_shared_start_and_end", aggregation_mode="mean", use_adapters=False)

# 2 gpus

device_map = infer_chunking_device_map_from_config(config, num_gpu)
model = Qwen2ChunkingForCausalLM.from_pretrained(
    model_name,
    config=config,
    torch_dtype="auto",
    device_map=device_map
)

# model = Qwen2ForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

print(model.hf_device_map)

tokenizer = Qwen2Tokenizer.from_pretrained(model_name)

prompt = "Can you teach to how to use python to draw a turtle?"
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)


import datasets
dataset = datasets.load_dataset("cais/mmlu", "all", split='test')
questions = dataset['question'][:64]

import torch
import time
start = time.time()

def batchify(lst, batch_size):
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]

for batch_questions in tqdm(batchify(questions, batch_size)):
    messages = [[
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": question}
    ] for question in batch_questions]
    texts = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer(texts, return_tensors="pt", padding=True, padding_side='left').to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=5,
        do_sample=False
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)

end = time.time()
print("Time taken: ", end - start)


"""
Experiment 1: Qwen2.5-7B-Instruct

with setting max_new_tokens=5, run time estimation experiment for:
1. num_chunks=2,
2. num_gpu=2,
3. num_layers_per_chunk=0, 2, 4, 8, 12
4. chunking_mode="uniform_with_shared_start_and_end"
5. aggregation_mode="mean"
6. batch_size=2, 4, 8, 16, 32, 64

(Note, when num_layers_per_chunk=0, num_chunks=None, chunking_mode="uniform_with_shared_start_and_end", it is equivalent to no chunking)

Results:

"""