from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from transformers import (
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
    DPRContextEncoder, DPRContextEncoderTokenizer
)
import torch
from typing import List


class DPRRetriever(BaseRetriever):
    def __init__(self, texts: List[str], metadatas: List[dict] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load encoders
        self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        self.q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(self.device)

        self.c_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        self.c_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(self.device)

        # Embed documents
        self.docs = texts
        self.metadatas = metadatas or [{}] * len(texts)
        self.vectorstore = self._build_vectorstore()

    def _build_vectorstore(self):
        inputs = self.c_tokenizer(self.docs, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            embeddings = self.c_encoder(**inputs).pooler_output.cpu().numpy()
        documents = [Document(page_content=text, metadata=meta) for text, meta in zip(self.docs, self.metadatas)]
        return FAISS.from_documents(documents, embedding=embeddings)

    def _embed_query(self, query: str):
        inputs = self.q_tokenizer(query, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            embedding = self.q_encoder(**inputs).pooler_output.cpu().numpy()[0]
        return embedding

    def get_relevant_documents(self, query: str) -> List[Document]:
        query_embedding = self._embed_query(query)
        return self.vectorstore.similarity_search_by_vector(query_embedding.tolist(), k=4)
