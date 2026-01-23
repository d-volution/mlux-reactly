# MLUX-Reactly Agent Framework
A ReAct inspired Agent developed from scratch using Ollama. This agent uses task decomposition and uses provided tools.

## Installation

### A virtual environment is recommended:
```sh
python3 -m venv .venv
source .venv/bin/activate
```

### From PIP
```sh
pip install mlux-reactly
```

### From Source
```sh
pip install git+https://github.com/d-volution/mlux-reactly.git
```

### Setting up Ollama
You also have to setup Ollama. See the [README of Ollama](https://github.com/ollama/ollama/blob/main/README.md) for that.

The agent uses by default the Ollama model `qwen2.5:7b-instruct-q8_0`, but can be configured via the `llm` keyword argument.
The demo RAG-tool included within this repository uses the embedding model `nomic-embed-text`.
```sh
ollama pull qwen2.5:7b-instruct-q8_0
ollama pull nomic-embed-text
```

## Usage
```python
from typing import Annotated
from mlux_reactly import ReactlyAgent, LLM

def count_substr(a: Annotated[str, "Some string"], b: Annotated[str, "substring to be counted"]) -> int:
    """This tool calculates how often string b occures in string a"""
    if not b:
        return 0
    return sum(1 for i in range(len(a) - len(b) + 1) if a[i:i+len(b)] == b)

agent = ReactlyAgent(tools=[count_substr], llm=LLM("qwen2.5:7b-instruct-q8_0"))

answer = agent.query("How many times does the letter l occure in 'artificial general intelligence'?")
print(f"agent answer: {answer}")
```

---

## When running from Git Repo

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