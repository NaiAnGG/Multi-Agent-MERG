# A Multi-Agent Framework with Structured Reasoning and Reflective Refinement for Multimodal Empathetic Response Generation

This repository contains the implementation of a multi-agent framework for multimodal empathetic response generation. The framework samples visual evidence from dialogue videos and uses structured reasoning stages for multimodal analysis, scene integration, response-strategy planning, response generation, and reflective refinement. It writes per-sample predictions and evaluation summaries for reproducible experiments.

## Requirements

- Python 3.9 or newer
- An accessible Ollama server with a vision-capable model installed

Install the Python packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Resources

- [Ollama](https://ollama.com/): local model serving used by the pipeline.
- [IEMOCAP](https://sail.usc.edu/iemocap/): interactive emotional dyadic motion capture dataset.
- [MELD](https://affective-meld.github.io/): multimodal emotionally annotated dialogue dataset.

## Configure Ollama

Copy the example configuration and edit the model and server URL for your environment:

```bash
cp OAI_CONFIG_LIST.example OAI_CONFIG_LIST
```

## Run

From the repository root:

```bash
python cap_main.py \
  --csv_path /path/to/dialogues.csv \
  --video_dir /path/to/videos \
  --oai_config ./OAI_CONFIG_LIST \
  --save_path ./outputs/results.jsonl
```

## Project files

- `cap_main.py`: command-line entry point and sample processing pipeline.
- `cap_agents.py`: Ollama client, video frame handling, and agent logic.
- `cap_prompt.py`: prompts used by the agents.

