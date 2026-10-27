# MMM

MMM is a video-grounded, multi-agent dialogue response generation and evaluation pipeline. It samples video frames, analyzes multimodal context, plans a response strategy, generates a response, and writes per-sample results and evaluation summaries.

## Requirements

- Python 3.9 or newer
- An accessible Ollama server with a vision-capable model installed
- The project data in the expected format (see below)
- `eval_qwen7_compat.py` and the model packages it requires. This evaluator module is imported by `cap_main.py` but is not included in this repository; add it or adapt the evaluation import before running.

Install the Python packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configure Ollama

Copy the example configuration and edit the model and server URL for your environment:

```bash
cp OAI_CONFIG_LIST.example OAI_CONFIG_LIST
```

The configuration is a JSON array. Each entry needs `model` and `base_url`; `api_key` is accepted for compatibility, but this client calls Ollama's local API and does not use it as an authentication secret. Keep private credentials out of this file and do not commit your local `OAI_CONFIG_LIST`.

Start Ollama, make sure the configured model is available, and verify the server URL is reachable from the machine running this project.

## Data

The input CSV must include these columns: `dialogue_name`, `sentence_id`, `speaker`, `utterance`, `emotion`, and `session_id`. Videos are expected as `<dialogue_name>.avi` under the video directory. The code expects data to be supplied separately; this repository does not include datasets or videos. Check the dataset's license and terms before using or redistributing it.

## Run

From the repository root:

```bash
python cap_main.py \
  --csv_path /path/to/dialogues.csv \
  --video_dir /path/to/videos \
  --oai_config ./OAI_CONFIG_LIST \
  --save_path ./outputs/results.jsonl
```

The output is JSON Lines. A summary JSON is written next to the results file by default. Use `python cap_main.py --help` to see all options.

## Project files

- `cap_main.py`: command-line entry point and sample processing pipeline.
- `cap_agents.py`: Ollama client, video frame handling, and agent logic.
- `cap_prompt.py`: prompts used by the agents.
- `OAI_CONFIG_LIST.example`: safe configuration template.
- `eval_qwen7_compat.py`: required by the entry point but currently not included.

## License

No license has been added yet. Add a license only after confirming you have the rights to publish the code and any included assets.
