# local-chatbot
Implementation of a local chatbot with CLI interface
Follow the instructions below to run this app locally (clone this repo and cd into it).

### Create conda environment and install dependencies

```bash
conda create -n "chatbot" python=3.12
conda activate chatbot
pip install -r requirements.txt
```

### Launch local OpenAi-compatible server

```bash
vllm serve [Qwen/Qwen3-0.6B] --trust-remote-code [--max_model_len 1000]
```

### On a new terminal window, run the main script

```bash
python main.py
```

```bash
usage: main.py [-h] [--model MODEL] [--openai_api_key OPENAI_API_KEY] [--openai_api_base OPENAI_API_BASE]
               [--max_tokens MAX_TOKENS] [--temperature TEMPERATURE]

Chatbot parameters

options:
  -h, --help            show this help message and exit
  --model MODEL
  --openai_api_key OPENAI_API_KEY
  --openai_api_base OPENAI_API_BASE
  --max_tokens MAX_TOKENS
  --temperature TEMPERATURE
```
