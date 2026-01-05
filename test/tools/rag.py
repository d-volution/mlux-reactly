from typing import Any, Annotated, List, Dict, Sequence
import logging
import json
from llama_index.core import VectorStoreIndex, Document, SimpleDirectoryReader
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.embeddings.ollama import OllamaEmbedding # type: ignore


logging.getLogger("httpx").setLevel(logging.WARNING)

def make_rag_for_folder(documents_path: str):
    documents: List[Document] = SimpleDirectoryReader(
        documents_path, 
        file_metadata=lambda x: {"filename": x}
    ).load_data()
    return make_rag_for_documents(documents)



def make_rag_for_documents(documents: Sequence[Document]):
    index = VectorStoreIndex.from_documents(documents, embed_model=OllamaEmbedding(
        model_name="nomic-embed-text"
    ))
    retriever: BaseRetriever = index.as_retriever()
    return make_rag_for_retriever(retriever)



def make_rag_for_retriever(retriever: BaseRetriever):
    def rag(query: Annotated[str, """string to query for document chunks"""]) -> List[Dict[str, Any]]:
        """A RAG tool for retrieving text documents."""

        results = retriever.retrieve(query)

        results_as_dicts = [{
            #'name': result.__doc__.lower(),
            'score': result.get_score(),
            'content': result.get_content(),
        } for result in results]

        return results_as_dicts

    return rag