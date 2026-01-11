# MLUX-Reactly
A ReAct inspired Agent from scratch using Ollama (topic AI6, supervisor Dr. Marian Lux).

## Getting Started

### A virtual environment is recommended:
```sh
python3 -m venv .venv
source .venv/bin/activate
```

### Install all dependencies:
```sh
pip install -r requirements.txt
```

### Setup Ollama
You also have to setup Ollama. See the [README of Ollama](https://github.com/ollama/ollama/blob/main/README.md) for that.

### Pull Ollama Models
The agent uses by default the Ollama model `qwen2.5:7b-instruct-q8_0`, but can be configured via the `llm` keyword argument.
The demo RAG-tool included within this repository uses the embedding model `nomic-embed-text`.
```sh
ollama pull qwen2.5:7b-instruct-q8_0
ollama pull nomic-embed-text
```

### Run simple chatbot
```sh
python3 test/chat.py
```

## Evaluation
To evaluate the agent, you can use `eval.py` like this:
```sh
python3 test/eval.py -agents <agent names> -tests <test names with test params>
```

For example, use
```sh
python3 test/eval.py -agents reactly llama-react -tests hotpot/train:100:3
```
to run the 'hotpot' test with the examples from index 100 to 103 on both the Reactly and the Llama-ReAct agents.