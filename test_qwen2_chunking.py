from model_chunking.models.qwen2 import Qwen2ChunkingForCausalLM, Qwen2ChunkingConfig, Qwen2Tokenizer, Qwen2Config
from tqdm.auto import tqdm

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
original_config = Qwen2Config.from_pretrained(model_name)

# Divisors for Qwen2.5-0.5B = 2, 3, 4, 6, 8, 12
# divisor = 1 should decompose to the original model
divisors = [1, 2, 3, 4, 6, 8, 12, 24]
for divisor in tqdm(divisors):
    config = Qwen2ChunkingConfig.from_pretrained(model_name, num_layers_per_chunk=original_config.num_hidden_layers // divisor, chunking_mode="sequential", aggregation_mode="mean")

    # chunking_mode: "sequential" or "uniform"
    # sequential: [[1,2,3], [4,5,6], ...]
    # uniform: [[1,4,7], [2,5,8], [3,6,9], ...]

    # aggregation_mode: "mlp" or "mean"
    # mlp: use a MLP to aggregate the outputs of each chunk
    # mean: use the mean of the outputs of each chunk

    model = Qwen2ChunkingForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    tokenizer = Qwen2Tokenizer.from_pretrained(model_name)

    prompt = "Tell me a short story."
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

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("-" * 20)
    print("Divisor = ", divisor)
    print("-" * 20)
    print(response)
    print("-" * 20)

    with open("test_qwen2_chunking.txt", "a") as f:
        f.write("-" * 20 + "\n")
        f.write("Divisor = " + str(divisor) + "\n")
        f.write("-" * 20 + "\n")
        f.write(response + "\n")
        f.write("-" * 20 + "\n")
        f.write("\n")