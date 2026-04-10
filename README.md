# OpenChat: Advancing Open-source Language Models with Mixed-Quality Data

<div align="center">
  <img src="assets/logo_new.png" style="width: 65%">
</div>

<p align="center">
  <a href="https://openchat.team">Online Demo</a> •
  <a href="https://discord.gg/pQjnXvNKHY">Discord</a> •
  <a href="https://huggingface.co/openchat">Huggingface</a> •
  <a href="https://arxiv.org/pdf/2309.11235.pdf">Paper</a>
</p>

**🔥 The first 7B model that Achieves Comparable Results with ChatGPT (March)! 🔥**

**🤖 #1 Open-source model on MT-bench scoring 7.81, outperforming 70B models 🤖**

<div style="display: flex; justify-content: center; align-items: center">
  <img src="assets/openchat.png" style="width: 45%;">
  <img src="assets/openchat_grok.png" style="width: 45%;">
</div>

OpenChat is an innovative library of open-source language models, fine-tuned with [C-RLFT](https://arxiv.org/pdf/2309.11235.pdf) - a strategy inspired by offline reinforcement learning. Our models learn from mixed-quality data without preference labels, delivering exceptional performance on par with ChatGPT, even with a 7B model. Despite our simple approach, we are committed to developing a high-performance, commercially viable, open-source large language model, and we continue to make significant strides toward this vision.

[![DOI](https://zenodo.org/badge/645397533.svg)](https://zenodo.org/badge/latestdoi/645397533)

## <a id="models"></a> Models

Our latest model, OpenChat 3.5, is a highly capable model fine-tuned using C-RLFT with Mistral 7B as the base, on a collection of publicly available high-quality instruction data. For older version models such as OpenChat 3.2 SUPER, please refer to [Legacy Models](#legacy-models).

For inference with Huggingface Transformers, follow the conversation template provided below.

<details>
  <summary>Conversation templates (click to expand)</summary>

```python
import transformers
tokenizer = transformers.AutoTokenizer.from_pretrained("openchat/openchat_3.5")

# Single-turn
tokens = tokenizer("GPT4 Correct User: Hello<|end_of_turn|>GPT4 Correct Assistant:").input_ids
assert tokens == [1, 420, 6316, 28781, 3198, 3123, 1247, 28747, 22557, 32000, 420, 6316, 28781, 3198, 3123, 21631, 28747]

# Multi-turn
tokens = tokenizer("GPT4 Correct User: Hello<|end_of_turn|>GPT4 Correct Assistant: Hi<|end_of_turn|>GPT4 Correct User: How are you today?<|end_of_turn|>GPT4 Correct Assistant:").input_ids
assert tokens == [1, 420, 6316, 28781, 3198, 3123, 1247, 28747, 22557, 32000, 420, 6316, 28781, 3198, 3123, 21631, 28747, 15359, 32000, 420, 6316, 28781, 3198, 3123, 1247, 28747, 1602, 460, 368, 3154, 28804, 32000, 420, 6316, 28781, 3198, 3123, 21631, 28747]

# Coding Mode
tokens = tokenizer("Code User: Implement quicksort using C++<|end_of_turn|>Code Assistant:").input_ids
assert tokens == [1, 7596, 1247, 28747, 26256, 2936, 7653, 1413, 334, 1680, 32000, 7596, 21631, 28747]
```

</details>

## <a id="installation"></a> Installation

First, make sure your nvidia driver version is at least 580. Check it by

```bash
cat /proc/driver/nvidia/version
```

If the version less than 580, you could reinstalled it by

```bash
sudo apt purge "^nvidia*"
sudo apt purge "^libnvidia*"
sudo apt autoremove
sudo apt install nvidia-driver-580
# reboot after this
```

Second, make sure cuda that you use in your environment is CUDA 13.0. You could install cuda by

```bash
sudo apt install cuda-toolkit-13-0
```

If CUDA 13.0 not exist in your package manager, add it first

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/<distro>/<arch>/cuda-keyring_1.1-1_all.deb # example distro=ubuntu2204, arch=x86_64
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
```

Then, after make sure CUDA 13.0 already installed in `/usr/local/cuda-13.0`, add it to your path and ld library path in `~/.bashrc` or `~/.profile`

```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Then, log out your terminal session and re-log in. Check CUDA version by

```bash
nvcc --version
```

To use OpenChat, you need to install PyTorch. If you encounter compatibility problems, you can try to create a new `conda` environment following the instructions below.

```bash
conda create -y --name openchat
conda activate openchat

conda install -y python=3.11
pip3 install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu130

pip3 install git+http://git.blackeye.id/fahadh/openchat.git
```

<details>
  <summary>In addition to PyPI, you can also install from source (click to expand)</summary>

```bash
git clone http://git.blackeye.id/fahadh/openchat.git
cd openchat

pip3 install --upgrade pip  # enable PEP 660 support
pip3 install -e .
```
</details>

## <a id="training"></a> OpenChat Model Training

The OpenChat training system utilizes padding-free training and the [Multipack Sampler](https://github.com/imoneoi/multipack_sampler), achieving a **3~10x** speedup compared to the conventional padded training.

## Choose a base model

OpenChat supports Llama 2 and Mistral models. Please first choose a base model to fit your needs. Each base model has a corresponding weight repo, model type, and recommended batch size as listed below, they should be filled into `BASE_REPO`, `MODEL_TYPE`, and `BATCH_SIZE` in the following instructions.

| Base Model | Size | Weights (with EOT token)          | Model Type              | Recommended Batch Size per GPU (8xA100 80GB) |
|------------|------|-----------------------------------|-------------------------|--------------------------------------|
| Mistral    | 7B   | `imone/Mistral_7B_with_EOT_token` | `mistral` | 83968                                |
| Llama 2    | 7B   | `imone/LLaMA2_7B_with_EOT_token`  | `llama`         | 83968                                |
| Llama 2    | 13B  | `imone/Llama2_13B_with_EOT_token` | `llama`         | 36864                                |

Note: The OpenChat conversation template requires an `<|end_of_turn|>` special token. The base model specified must include this token. Our provided weights are the original base weights with this token added. If you want to add them manually, use the `convert_llama_weights_to_hf_add_tokens.py` or `mistral_add_tokens.py` in the `scripts` directory.

## Installing DeepSpeed, Flash Attention, Flash Linear Attention, and Causal Conv1d

First, ensure that the CUDA `nvcc` compiler is available in your environment. If it is not, install the CUDA toolkit that matches the version used by PyTorch.

Next, install DeepSpeed:

```bash
pip install deepspeed
```

> Better to set environment variable `DS_BUILD_CPU_ADAM=1` and `DS_BUILD_FUSED_ADAM=1` before installing deepspeed to build the optimizer.

Then, install flash attention:

```bash
pip install flash-attn==2.8.3 --no-build-isolation
```

Additionally, install ring flash attention:

```bash
pip install ring_flash_attn@git+https://github.com/zhuzilin/ring-flash-attention
```

Next, install Flash Linear Attention:

```bash
pip install flash-linear-attention==0.5.0
```

Then, install causal conv 1d

```bash
pip install causal-conv1d==1.6.1
```


### Preparing Your Data

To utilize the OpenChat trainer, prepare your SFT data into a JSON Lines format where each line corresponds to a `Conversation` object:

```python
class Message(BaseModel):
    role: str     # Must be "user" or "assistant"
    content: str  # Message content
    weight: Optional[float] = None  # Loss weight for this message. Typically 0 for user and 1 for assistant to supervise assistant's responses only
    name: Optional[str] = None # Additional name for the role


class Conversation(BaseModel):
    items: List[Message]  # All messages within the conversation
    condition: str = ""  # C-RLFT condition, can be any string or empty.
    system: str = ""  # System message for this conversation

    images: Optional[List[str]] = None # Path to images relative to dataset directory
    videos: Optional[List[str]] = None # Path to videos relative to dataset directory
```

For basic SFT, assign `weight` as `0` for human messages and `1` for assistant responses.

SFT example:

```json
{"items":[{"role":"user","content":"Hello","weight":0.0},{"role":"assistant","content":"Hi","weight":1.0},{"role":"user","content":"How are you today?","weight":0.0},{"role":"assistant","content":"I'm fine.","weight":1.0}],"system":""}
{"items":[{"role":"user","content":"Who are you?","weight":0.0},{"role":"assistant","content":"I'm OpenChat.","weight":1.0}],"system":"You are a helpful assistant named OpenChat."}
```

For C-RLFT, `condition` should be set as the class the conversation belongs to (e.g. `GPT3` or `GPT4`). The `weight` is assigned as `0` for human messages and `w` for assistant responses, where `w` is the weight of the class (e.g. `0.1` for `GPT3` and `1` for `GPT4`, as found in our C-RLFT paper).

C-RLFT example:

```json
{"items":[{"role":"user","content":"What is C-RLFT?","weight":0.0},{"role":"assistant","content":"C-RLFT is a method for improving open-source LLMs with mixed-quality data.","weight":1.0}],"condition":"GPT4","system":""}
{"items":[{"role":"user","content":"What is C-RLFT?","weight":0.0},{"role":"assistant","content":"I don't know.","weight":0.1}],"condition":"GPT3","system":""}
```

### Pre-tokenizing the Dataset

You'll then need to pre-tokenize the dataset using the command (please specify a filename as `PRETOKENIZED_DATA_OUTPUT_PATH` to store the pretokenized dataset):

```bash
python -m ochat.data.generate_dataset --model-type MODEL_TYPE --model-path BASE_REPO --in-files data.jsonl --out-prefix PRETOKENIZED_DATA_OUTPUT_PATH
```

### Launching the OpenChat Trainer

You can now launch the OpenChat trainer using the command below. Training a 13B model requires eight A/H100s with 80GB VRAM, while a 7B model can be trained with four A/H100s with 80GB VRAM or eight A/H100s with 40GB VRAM.

For hyperparameters, we recommend first setting the batch size to the recommended batch size. If OOM occurs, try setting it to the exact maximum that VRAM can hold and as a multiple of `2048`.
Other hyperparameters have been carefully selected as the default. Furthermore, the learning rate is automatically determined based on the [inverse square-root rule](https://arxiv.org/abs/2006.09092).

<details>

<summary>Training Commands (click to expand)</summary>

```bash
NUM_GPUS=8

deepspeed --num_gpus=$NUM_GPUS --module ochat.training_deepspeed.train \
          --model_path BASE_REPO \
          --data_prefix PRETOKENIZED_DATA_OUTPUT_PATH \
          --save_path PATH_TO_SAVE_MODEL \
          --batch_max_len BATCH_SIZE \
          --epochs 5 \
          --save_every 1 \
          --deepspeed \
          --deepspeed_config ochat/training_deepspeed/deepspeed_config.json
```

> Change `ochat.training_deepspeed.train` to `ochat.training_deepspeed.train_offload` if you want to use deepspeed ZERO offloading

</details>

<details>

<summary>Training Commands LORA (click to expand)</summary>

```bash
NUM_GPUS=8

deepspeed --num_gpus=$NUM_GPUS --module ochat.training_deepspeed.train \
          --model_path BASE_REPO \
          --data_prefix PRETOKENIZED_DATA_OUTPUT_PATH \
          --save_path PATH_TO_SAVE_MODEL \
          --batch_max_len BATCH_SIZE \
          --epochs 50 \
          --save_every 10 \
          --use_lora \
          --lora_alpha 32 \
          --lora_r 32 \
          --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
          --deepspeed \
          --deepspeed_config ochat/training_deepspeed/deepspeed_config.json
```

> Change `ochat.training_deepspeed.train` to `ochat.training_deepspeed.train_offload` if you want to use deepspeed ZERO offloading

</details>

You can find checkpoints of all epochs in `PATH_TO_SAVE_MODEL`. Then you may evaluate each epoch and choose the best one.

## Limitations

**Foundation Model Limitations**
Despite its advanced capabilities, OpenChat is still bound by the limitations inherent in its foundation models. These limitations may impact the model's performance in areas such as:

 - Complex reasoning
 - Mathematical and arithmetic tasks
 - Programming and coding challenges

**Hallucination of Non-existent Information**
OpenChat may sometimes generate information that does not exist or is not accurate, also known as "hallucination". Users should be aware of this possibility and verify any critical information obtained from the model.

**Safety**
OpenChat may sometimes generate harmful, hate speech, biased responses, or answer unsafe questions. It's crucial to apply additional AI safety measures in use cases that require safe and moderated responses.

## License

Our OpenChat 3.5 code and models are distributed under the Apache License 2.0.

## Contact

💌 We are a student team from Tsinghua University, working on OpenChat, a project that requires additional computing power or LLMs API keys for further development. If you are interested in our project and would like to offer support, please feel free to reach out to us:

* Wang Guan [imonenext at gmail dot com]
* Cheng Sijie [csj23 at mails dot tsinghua dot edu dot cn]

We look forward to hearing from you and collaborating on this exciting project!

## Citation

```
@article{wang2023openchat,
  title={OpenChat: Advancing Open-source Language Models with Mixed-Quality Data},
  author={Wang, Guan and Cheng, Sijie and Zhan, Xianyuan and Li, Xiangang and Song, Sen and Liu, Yang},
  journal={arXiv preprint arXiv:2309.11235},
  year={2023}
}
```

## Acknowledgements

We extend our heartfelt gratitude to Alignment Lab AI, Nous Research, and Pygmalion AI for their substantial contributions to data collection and model training.

Special thanks go to Changling Liu from GPT Desk Pte. Ltd., Qiying Yu at Tsinghua University, Baochang Ma, and Hao Wan from 01.AI company for their generous provision of resources. We are also deeply grateful to Jianxiong Li and Peng Li at Tsinghua University for their insightful discussions.

Furthermore, we appreciate the developers behind the following projects for their significant contributions to our research: [Mistral](https://mistral.ai/), [Chain-of-Thought Hub](https://github.com/FranxYao/chain-of-thought-hub), [Llama 2](https://ai.meta.com/llama/), [Self-Instruct](https://arxiv.org/abs/2212.10560), [FastChat (Vicuna)](https://github.com/lm-sys/FastChat), [Alpaca](https://github.com/tatsu-lab/stanford_alpaca.git), and [StarCoder](https://github.com/bigcode-project/starcoder). Their work has been instrumental in driving our research forward.
