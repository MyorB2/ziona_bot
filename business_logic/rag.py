import ast
import logging
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional
from ollama import Client

# RAG imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)

HF_TOKEN = "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"

# Suppress Windows symlink warning for Hugging Face cache
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


@dataclass
class RetrievalResult:
    """Data class for retrieval results"""
    paragraph: str
    source: str
    url: str
    score: float
    category_id: int


class RAGRetriever:
    def __init__(
            self,
            knowledge_base: pd.DataFrame,
            top_k: Optional[int] = 5
    ):
        self.knowledge_base = knowledge_base
        self.top_k = top_k
        self.documents: Optional[List[Document]] = None  # All original documents with metadata
        self.vectorstore = None
        self.llm = None
        self.comment: Optional[str] = None
        self.category_name: Optional[str] = None
        self.category_id: Optional[int] = None
        self.query: Optional[str] = None
        self.embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.bm25_weight: float = 0.6
        self.faiss_weight: float = 0.4

        self._set_documents()
        logger.info("RAG Retriever initialized successfully")

    def _set_documents(self):
        self.documents = [Document(
            page_content=row["paragraph"],
            metadata={
                "source": row["source"],
                "url": row["url"],
                "category_id": row["category_id"]
            }
        )
            for _, row in self.knowledge_base.iterrows()
        ]

    def _set_vectorstore(self):
        try:
            logger.info("Setting vectorstore")
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(self.documents, embedding_model)
            vectorstore.save_local("faiss_index")
            self.vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
        except Exception as e:
            logger.error(f"Error while set_vectorstore: {e}")
            raise Exception(f"Error while set_vectorstore: {e}")

    def _initialize_llm(self):
        """Initialize the Ollama LLaMA 3.2 model"""
        try:
            self.client = Client(host='http://localhost:11434')
            self.model_name = "llama3"
            logger.info("Successfully initialized Ollama LLM client")
        except Exception as e:
            logger.error(f"Error initializing Ollama LLM: {e}")
            raise

    def _filter_documents_by_category(self) -> List[Document]:
        if self.category_id is None:
            return self.documents
        return [
            doc for doc in self.documents
            if "category_id" in doc.metadata and self.category_id in doc.metadata["category_id"]
        ]

    def _bm25_scores(self, filtered_docs) -> Dict[int, float]:
        if not filtered_docs or len(filtered_docs) == 0:
            return {}
        tokenized = [doc.page_content.split() for doc in filtered_docs]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(self.query.split())
        return {i: float(score) for i, score in enumerate(scores)}

    def _faiss_scores(self, filtered_docs) -> Dict[int, float]:
        if not filtered_docs or len(filtered_docs) == 0:
            return {}
        query_embedding = self.embedding_function.embed_query(self.query)
        doc_embeddings = self.embedding_function.embed_documents([doc.page_content for doc in filtered_docs])

        scores = [np.dot(query_embedding, doc_emb) for doc_emb in doc_embeddings]
        return {i: float(score) for i, score in enumerate(scores)}

    def _get_combined_results(self, filtered_docs) -> List[Document]:
        bm25_scores = self._bm25_scores(filtered_docs)
        faiss_scores = self._faiss_scores(filtered_docs)

        combined_scores = {
            i: self.bm25_weight * bm25_scores.get(i, 0.0) + self.faiss_weight * faiss_scores.get(i, 0.0)
            for i in range(len(self.documents))
        }

        top_indices = sorted(combined_scores, key=lambda i: combined_scores[i], reverse=True)[:self.top_k]
        return [self.documents[i] for i in top_indices]

    def retrieve(self, comment: str, category_name: str, category_id: int) -> List[Document]:
        self.comment = comment
        self.category_id = category_id
        self.category_name = category_name
        self.query = comment  # This is not a mistake, do not change it

        self._set_vectorstore()
        filtered_docs = self._filter_documents_by_category()
        return self._get_combined_results(filtered_docs)

