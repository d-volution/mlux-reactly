from __future__ import annotations
from typing import Any, List
import json
import logging
from pathlib import Path
from llama_index.core import VectorStoreIndex, Document, SimpleDirectoryReader
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding # type: ignore
from ..types import Tool

logging.getLogger("httpx").setLevel(logging.WARNING)

class RetrieverTool(Tool):
    _name: str
    _desc: str
    retriever: BaseRetriever
    
    def __init__(self, retriever: BaseRetriever):
        self._name = "retriever"
        self._desc = """
The retriever tool can retrieve certain documents (mostly locally on the same machine). Documents can be queried by search terms. Kinda like a RAG.

Use the tool like this:
---
Action: retriever
Action Input: { "input": "query for the topic" }
---

The tool will respond like this:
---
Observation: [{"score": score of result, "content": "content of document"}, ...more results...]
---
Each result (document) will be one object in the list.

"""
        self.retriever = retriever

    @staticmethod
    def from_directory_path(path: str = "./test-files") -> RetrieverTool:
        documents: List[Document] = SimpleDirectoryReader(
            path, 
            #file_metadata=lambda x: {"filename": x}
        ).load_data()

        for doc in documents:
            print(f"- doc {doc.get_doc_id()} meta: {doc.get_metadata_str()}")

        index = VectorStoreIndex.from_documents(documents, embed_model=OllamaEmbedding(
            model_name="qwen2.5:7b-instruct-q8_0"
        ))
        retriever = index.as_retriever()
        return RetrieverTool(retriever)


    @property
    def name(self) -> str:
        return self._name
    
    @property
    def desc(self) -> str:
        return self._desc

    def run(self, input: dict[str, Any]) -> tuple[str, bool]:
        query: str = input["input"]
        #return "[]", True
        results = self.retriever.retrieve(query)
        results_dics = [{
            #'name': result.__doc__.lower(),
            'score': result.get_score(),
            'content': result.get_content(),
        } for result in results]
        print(f"# retriever: {len(results)} results")
        return json.dumps(results_dics), True


