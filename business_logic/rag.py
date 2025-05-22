from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from transformers import (
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
    DPRContextEncoder, DPRContextEncoderTokenizer
)
import torch
from typing import List

from models.dpr_retriever import DPRRetriever

# Load documents (e.g. from Excel, PDF, etc.)
texts = ["Solar energy is renewable...", "Wind turbines convert wind into electricity..."]

# Initialize retriever
retriever = DPRRetriever(texts)

# Load your LLM (e.g., Gemma)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto", torch_dtype=torch.float16)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=pipe)

# Build chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
rag_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# Query
response = rag_chain({"question": "How do solar panels work?"})
print(response["answer"])
