# model-chunking
Project for CS 854 (Model Serving Systems for GenAI), uWaterloo, Fall 2024.

## Installation
```bash
# in your project env
conda create -n model-chunking python=3.10
conda activate model-chunking
pip install -e .
pip install flash-attn --no-build-isolation
```

## Usage
```bash
python test_qwen2.py
```

## Training
### setup
```bash
cd LLaMA-Factory && pip install -e .
```
Then add `import model_chunking` to `LLaMA-Factory/src/llamafactory/launcher.py`, it looks like this:
```python
import model_chunking # our models registered here
from llamafactory.train.tuner import run_exp  # use absolute import


def launch():
    run_exp()


if __name__ == "__main__":
    launch()
```

Then we can use our custom chunking models in LLaMA-Factory.

In the `./LLaMA-Factory/examples/train_lora/llama3_lora_sft.yaml`, change the `model_name_or_path` to be `DongfuJiang/Qwen2.5-0.5B-Instruct`. Then we can train the model.

- Lora
```bash
cd LLaMA-Factory && llamafactory-cli train ../training_configs/qwen2_chunking_lora.yaml
```

### How is `DongfuJiang/Qwen2.5-0.5B-Instruct` configured?
I simply change the `config.json` after duplicate the `Qwen/Qwen2.5-0.5B-Instruct`

```json
{
  "architectures": [
    "Qwen2ChunkingForCausalLM" # before it was "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 896,
  "initializer_range": 0.02,
  "intermediate_size": 4864,
  "max_position_embeddings": 32768,
  "max_window_layers": 21,
  "model_type": "qwen2_chunking", # before it was "qwen2"
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "sliding_window": 32768,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.43.1",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}
```

To add other custom configs, add them like this:
```json
{
    ...,
    "num_layers_per_chunk": 3,
    "chunking_mode": "sequential",
    "aggregation_mode": "mean",
    "use_adapters": false
}
```
Then apply the changes to your huggingface model.


## TODO
- [x] Write evaluation script in terms of perplexity, QA accuracy, etc.
- [x] Enable models to be run in parallel on GPUs.

## Full MMLU Results

In our paper, we provide a summary of our MMLU results, while the complete results across all 57 tasks are presented below. Notably, we outperform the baseline in key areas, including Machine Learning, Medical Genetics, Business Ethics, Human Aging, Marketing, and World Religions.

|Task                                           |Metric|SFT k=2, m=12||Stderr|SFT k=2, m=6||Stderr|CPT k=2, m=12||Stderr|CPT k=2, m=6||Stderr|k=0, m=0 (Baseline)||Stderr|
|-----------------------------------------------|------|-------------------|------------------------|--------------------|------------------|-----------------------|-------------------|------|----------|------|------------------|-----------------------|-------------------|-------------------------|------------------------------|--------------------------|
|mmlu:_average                           |qem   |0.2239             |±                       |0.0311              |0.2148            |±                      |0.0307             |0.2312|±         |0.0315|0.2303            |±                      |0.0314             |0.2931                   |±                             |0.0339                    |
|mmlu:abstract_algebra                   |qem   |0.2200             |±                       |0.0416              |0.2100            |±                      |0.0409             |0.2200|±         |0.0416|0.2200            |±                      |0.0416             |0.3400                   |±                             |0.0476                    |
|mmlu:anatomy                            |qem   |0.1778             |±                       |0.0330              |0.1778            |±                      |0.0330             |0.1852|±         |0.0336|0.1852            |±                      |0.0336             |0.2741                   |±                             |0.0385                    |
|mmlu:astronomy                          |qem   |0.1579             |±                       |0.0297              |0.1711            |±                      |0.0306             |0.1776|±         |0.0311|0.1776            |±                      |0.0311             |0.3421                   |±                             |0.0386                    |
|mmlu:business_ethics                    |qem   |0.3000             |±                       |0.0461              |0.2900            |±                      |0.0456             |0.3000|±         |0.0461|0.3000            |±                      |0.0461             |0.2300                   |±                             |0.0423                    |
|mmlu:clinical_knowledge                 |qem   |0.1925             |±                       |0.0243              |0.2075            |±                      |0.0250             |0.2151|±         |0.0253|0.2151            |±                      |0.0253             |0.2566                   |±                             |0.0269                    |
|mmlu:college_biology                    |qem   |0.2569             |±                       |0.0365              |0.2292            |±                      |0.0351             |0.2569|±         |0.0365|0.2500            |±                      |0.0362             |0.2500                   |±                             |0.0362                    |
|mmlu:college_chemistry                  |qem   |0.2000             |±                       |0.0402              |0.1600            |±                      |0.0368             |0.2000|±         |0.0402|0.2000            |±                      |0.0402             |0.2700                   |±                             |0.0446                    |
|mmlu:college_computer_science           |qem   |0.2600             |±                       |0.0441              |0.2500            |±                      |0.0435             |0.2600|±         |0.0441|0.2600            |±                      |0.0441             |0.3700                   |±                             |0.0485                    |
|mmlu:college_mathematics                |qem   |0.2000             |±                       |0.0402              |0.1900            |±                      |0.0394             |0.2100|±         |0.0409|0.1900            |±                      |0.0394             |0.3200                   |±                             |0.0469                    |
|mmlu:college_medicine                   |qem   |0.2023             |±                       |0.0306              |0.1850            |±                      |0.0296             |0.2081|±         |0.0310|0.2081            |±                      |0.0310             |0.2890                   |±                             |0.0346                    |
|mmlu:college_physics                    |qem   |0.2157             |±                       |0.0409              |0.2157            |±                      |0.0409             |0.2157|±         |0.0409|0.2157            |±                      |0.0409             |0.3039                   |±                             |0.0458                    |
|mmlu:computer_security                  |qem   |0.2700             |±                       |0.0446              |0.2600            |±                      |0.0441             |0.2800|±         |0.0451|0.2800            |±                      |0.0451             |0.3000                   |±                             |0.0461                    |
|mmlu:conceptual_physics                 |qem   |0.2596             |±                       |0.0287              |0.2383            |±                      |0.0279             |0.2638|±         |0.0288|0.2638            |±                      |0.0288             |0.3319                   |±                             |0.0308                    |
|mmlu:econometrics                       |qem   |0.2368             |±                       |0.0400              |0.2281            |±                      |0.0395             |0.2368|±         |0.0400|0.2368            |±                      |0.0400             |0.2895                   |±                             |0.0427                    |
|mmlu:electrical_engineering             |qem   |0.2345             |±                       |0.0353              |0.2345            |±                      |0.0353             |0.2414|±         |0.0357|0.2414            |±                      |0.0357             |0.3862                   |±                             |0.0406                    |
|mmlu:elementary_mathematics             |qem   |0.2090             |±                       |0.0209              |0.1878            |±                      |0.0201             |0.2090|±         |0.0209|0.2090            |±                      |0.0209             |0.3042                   |±                             |0.0237                    |
|mmlu:formal_logic                       |qem   |0.2698             |±                       |0.0397              |0.2619            |±                      |0.0393             |0.2857|±         |0.0404|0.2857            |±                      |0.0404             |0.3016                   |±                             |0.0410                    |
|mmlu:global_facts                       |qem   |0.1800             |±                       |0.0386              |0.1800            |±                      |0.0386             |0.1800|±         |0.0386|0.1700            |±                      |0.0378             |0.2600                   |±                             |0.0441                    |
|mmlu:high_school_biology                |qem   |0.1710             |±                       |0.0214              |0.1645            |±                      |0.0211             |0.1774|±         |0.0217|0.1774            |±                      |0.0217             |0.2774                   |±                             |0.0255                    |
|mmlu:high_school_chemistry              |qem   |0.1379             |±                       |0.0243              |0.1379            |±                      |0.0243             |0.1527|±         |0.0253|0.1527            |±                      |0.0253             |0.3202                   |±                             |0.0328                    |
|mmlu:high_school_computer_science       |qem   |0.2300             |±                       |0.0423              |0.2400            |±                      |0.0429             |0.2500|±         |0.0435|0.2500            |±                      |0.0435             |0.2400                   |±                             |0.0429                    |
|mmlu:high_school_european_history       |qem   |0.2121             |±                       |0.0319              |0.2061            |±                      |0.0316             |0.2182|±         |0.0323|0.2121            |±                      |0.0319             |0.3212                   |±                             |0.0365                    |
|mmlu:high_school_geography              |qem   |0.1717             |±                       |0.0269              |0.1717            |±                      |0.0269             |0.1768|±         |0.0272|0.1768            |±                      |0.0272             |0.2525                   |±                             |0.0310                    |
|mmlu:high_school_government_and_politics|qem   |0.1813             |±                       |0.0278              |0.1865            |±                      |0.0281             |0.1969|±         |0.0287|0.1969            |±                      |0.0287             |0.2487                   |±                             |0.0312                    |
|mmlu:high_school_macroeconomics         |qem   |0.1949             |±                       |0.0201              |0.1872            |±                      |0.0198             |0.2026|±         |0.0204|0.2026            |±                      |0.0204             |0.2615                   |±                             |0.0223                    |
|mmlu:high_school_mathematics            |qem   |0.2111             |±                       |0.0249              |0.2037            |±                      |0.0246             |0.2111|±         |0.0249|0.2111            |±                      |0.0249             |0.3556                   |±                             |0.0292                    |
|mmlu:high_school_microeconomics         |qem   |0.2059             |±                       |0.0263              |0.2017            |±                      |0.0261             |0.2101|±         |0.0265|0.2101            |±                      |0.0265             |0.2563                   |±                             |0.0284                    |
|mmlu:high_school_physics                |qem   |0.1921             |±                       |0.0322              |0.1854            |±                      |0.0317             |0.1987|±         |0.0326|0.1987            |±                      |0.0326             |0.2715                   |±                             |0.0363                    |
|mmlu:high_school_psychology             |qem   |0.1890             |±                       |0.0168              |0.1633            |±                      |0.0158             |0.1927|±         |0.0169|0.1908            |±                      |0.0168             |0.2844                   |±                             |0.0193                    |
|mmlu:high_school_statistics             |qem   |0.1528             |±                       |0.0245              |0.1343            |±                      |0.0233             |0.1528|±         |0.0245|0.1528            |±                      |0.0245             |0.3287                   |±                             |0.0320                    |
|mmlu:high_school_us_history             |qem   |0.2304             |±                       |0.0296              |0.2304            |±                      |0.0296             |0.2500|±         |0.0304|0.2500            |±                      |0.0304             |0.3186                   |±                             |0.0327                    |
|mmlu:high_school_world_history          |qem   |0.2574             |±                       |0.0285              |0.2447            |±                      |0.0280             |0.2700|±         |0.0289|0.2700            |±                      |0.0289             |0.2996                   |±                             |0.0298                    |
|mmlu:human_aging                        |qem   |0.3139             |±                       |0.0311              |0.3139            |±                      |0.0311             |0.3139|±         |0.0311|0.3139            |±                      |0.0311             |0.2780                   |±                             |0.0301                    |
|mmlu:human_sexuality                    |qem   |0.2519             |±                       |0.0381              |0.2443            |±                      |0.0377             |0.2595|±         |0.0384|0.2595            |±                      |0.0384             |0.4351                   |±                             |0.0435                    |
|mmlu:international_law                  |qem   |0.2314             |±                       |0.0385              |0.2314            |±                      |0.0385             |0.2397|±         |0.0390|0.2397            |±                      |0.0390             |0.1901                   |±                             |0.0358                    |
|mmlu:jurisprudence                      |qem   |0.2407             |±                       |0.0413              |0.2222            |±                      |0.0402             |0.2593|±         |0.0424|0.2593            |±                      |0.0424             |0.2778                   |±                             |0.0433                    |
|mmlu:logical_fallacies                  |qem   |0.2147             |±                       |0.0323              |0.1902            |±                      |0.0308             |0.2209|±         |0.0326|0.2209            |±                      |0.0326             |0.3129                   |±                             |0.0364                    |
|mmlu:machine_learning                   |qem   |0.3036             |±                       |0.0436              |0.2946            |±                      |0.0433             |0.3125|±         |0.0440|0.3125            |±                      |0.0440             |0.2500                   |±                             |0.0411                    |
|mmlu:management                         |qem   |0.1748             |±                       |0.0376              |0.1553            |±                      |0.0359             |0.1748|±         |0.0376|0.1748            |±                      |0.0376             |0.3398                   |±                             |0.0469                    |
|mmlu:marketing                          |qem   |0.2906             |±                       |0.0297              |0.2393            |±                      |0.0280             |0.2906|±         |0.0297|0.2906            |±                      |0.0297             |0.2607                   |±                             |0.0288                    |
|mmlu:medical_genetics                   |qem   |0.3000             |±                       |0.0461              |0.2900            |±                      |0.0456             |0.3000|±         |0.0461|0.3000            |±                      |0.0461             |0.2100                   |±                             |0.0409                    |
|mmlu:miscellaneous                      |qem   |0.2363             |±                       |0.0152              |0.2324            |±                      |0.0151             |0.2375|±         |0.0152|0.2375            |±                      |0.0152             |0.2669                   |±                             |0.0158                    |
|mmlu:moral_disputes                     |qem   |0.2312             |±                       |0.0227              |0.2023            |±                      |0.0216             |0.2486|±         |0.0233|0.2486            |±                      |0.0233             |0.3150                   |±                             |0.0250                    |
|mmlu:moral_scenarios                    |qem   |0.2380             |±                       |0.0142              |0.2380            |±                      |0.0142             |0.2380|±         |0.0142|0.2380            |±                      |0.0142             |0.2726                   |±                             |0.0149                    |
|mmlu:nutrition                          |qem   |0.2222             |±                       |0.0238              |0.2255            |±                      |0.0239             |0.2255|±         |0.0239|0.2255            |±                      |0.0239             |0.3464                   |±                             |0.0272                    |
|mmlu:philosophy                         |qem   |0.1736             |±                       |0.0215              |0.1415            |±                      |0.0198             |0.1865|±         |0.0221|0.1865            |±                      |0.0221             |0.3312                   |±                             |0.0267                    |
|mmlu:prehistory                         |qem   |0.2037             |±                       |0.0224              |0.2037            |±                      |0.0224             |0.2160|±         |0.0229|0.2130            |±                      |0.0228             |0.2994                   |±                             |0.0255                    |
|mmlu:professional_accounting            |qem   |0.2305             |±                       |0.0251              |0.1915            |±                      |0.0235             |0.2340|±         |0.0253|0.2340            |±                      |0.0253             |0.2411                   |±                             |0.0255                    |
|mmlu:professional_law                   |qem   |0.2216             |±                       |0.0106              |0.2145            |±                      |0.0105             |0.2458|±         |0.0110|0.2458            |±                      |0.0110             |0.2301                   |±                             |0.0108                    |
|mmlu:professional_medicine              |qem   |0.1838             |±                       |0.0235              |0.1838            |±                      |0.0235             |0.1838|±         |0.0235|0.1838            |±                      |0.0235             |0.2684                   |±                             |0.0269                    |
|mmlu:professional_psychology            |qem   |0.2418             |±                       |0.0173              |0.2206            |±                      |0.0168             |0.2500|±         |0.0175|0.2484            |±                      |0.0175             |0.2533                   |±                             |0.0176                    |
|mmlu:public_relations                   |qem   |0.2182             |±                       |0.0396              |0.1909            |±                      |0.0376             |0.2182|±         |0.0396|0.2182            |±                      |0.0396             |0.1545                   |±                             |0.0346                    |
|mmlu:security_studies                   |qem   |0.1755             |±                       |0.0244              |0.1878            |±                      |0.0250             |0.1878|±         |0.0250|0.1878            |±                      |0.0250             |0.3673                   |±                             |0.0309                    |
|mmlu:sociology                          |qem   |0.2388             |±                       |0.0301              |0.2438            |±                      |0.0304             |0.2438|±         |0.0304|0.2438            |±                      |0.0304             |0.3781                   |±                             |0.0343                    |
|mmlu:us_foreign_policy                  |qem   |0.2400             |±                       |0.0429              |0.2800            |±                      |0.0451             |0.2800|±         |0.0451|0.2800            |±                      |0.0451             |0.3500                   |±                             |0.0479                    |
|mmlu:virology                           |qem   |0.2831             |±                       |0.0351              |0.2651            |±                      |0.0344             |0.2831|±         |0.0351|0.2831            |±                      |0.0351             |0.3373                   |±                             |0.0368                    |
|mmlu:world_religions                    |qem   |0.3216             |±                       |0.0358              |0.3041            |±                      |0.0353             |0.3216|±         |0.0358|0.3216            |±                      |0.0358             |0.2865                   |±                             |0.0347                    |
