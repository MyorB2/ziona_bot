import ollama
import torch
from typing import List, Dict, Any
from pathlib import Path

# LangChain imports
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings.base import Embeddings

# Transformers imports
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, pipeline,
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
    DPRContextEncoder, DPRContextEncoderTokenizer
)

# Business logic imports
from business_logic.llm_evaluation import LLMEvaluator
from business_logic.multilabel_classifier import AntisemitismClassifier


class DPREmbeddings(Embeddings):
    """Custom embeddings class for DPR context encoder"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        )
        self.encoder = DPRContextEncoder.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        ).to(self.device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            embeddings = self.encoder(**inputs).pooler_output.cpu().numpy()

        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query - use question encoder for queries"""
        # For queries, we should use the question encoder
        q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        q_encoder = DPRQuestionEncoder.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        ).to(self.device)

        inputs = q_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            embedding = q_encoder(**inputs).pooler_output.cpu().numpy()[0]

        return embedding.tolist()


class DocumentLoader:
    """Load documents from various sources (Excel, PDF, etc.)"""

    @staticmethod
    def load_from_excel(file_path: str, text_column: str, metadata_columns: List[str] = None) -> List[Document]:
        """Load documents from Excel file"""
        import pandas as pd

        try:
            df = pd.read_excel(file_path)
            documents = []

            for idx, row in df.iterrows():
                text = str(row[text_column])
                metadata = {"source": f"row_{idx}"}

                if metadata_columns:
                    for col in metadata_columns:
                        if col in row:
                            metadata[col] = str(row[col])

                documents.append(Document(page_content=text, metadata=metadata))

            return documents

        except Exception as e:
            print(f"Error loading Excel file: {e}")
            return []

    @staticmethod
    def load_from_text_list(texts: List[str], metadatas: List[Dict] = None) -> List[Document]:
        """Load documents from list of texts"""
        if metadatas is None:
            metadatas = [{"source": f"text_{i}"} for i in range(len(texts))]

        return [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]


class AntisemitismResponseSystem:
    """Main system for generating responses to antisemitic comments"""

    def __init__(self, documents_source: str = None, excel_config: Dict = None):
        """
        Initialize the system

        Args:
            documents_source: Path to Excel file or None for sample data
            excel_config: Dict with 'text_column' and optional 'metadata_columns'
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Initialize classifier
        self.classifier = AntisemitismClassifier()

        # Initialize evaluator
        self.evaluator = LLMEvaluator(evaluator_model="llama3.2")

        # Load documents and setup RAG
        self._setup_rag(documents_source, excel_config)

        # Setup LLM pipeline
        self._setup_llm()

        # Setup conversation chain
        self._setup_conversation_chain()

    def _setup_rag(self, documents_source: str = None, excel_config: Dict = None):
        """Setup RAG retrieval system"""
        print("Setting up RAG system...")

        # Load documents
        if documents_source and Path(documents_source).exists():
            if excel_config:
                documents = DocumentLoader.load_from_excel(
                    documents_source,
                    excel_config.get('text_column', 'text'),
                    excel_config.get('metadata_columns', [])
                )
            else:
                raise ValueError("Excel config required when using Excel file")
        else:
            # Sample documents - replace with your actual data
            sample_texts = [
                "Antisemitism is prejudice against or hatred of Jewish people. It has manifested throughout history in various forms including religious persecution, racial theories, and conspiracy theories.",
                "The International Holocaust Remembrance Alliance (IHRA) working definition of antisemitism provides a comprehensive framework for understanding modern antisemitism.",
                "Combating antisemitism requires education, awareness, and understanding of its various manifestations including classical antisemitism, Holocaust denial, and anti-Zionism that crosses into antisemitism.",
                "Jewish communities have contributed significantly to society in areas of science, arts, philosophy, and social justice throughout history.",
                "Resources for combating antisemitism include educational materials from the Anti-Defamation League, the Simon Wiesenthal Center, and various academic institutions."
            ]
            documents = DocumentLoader.load_from_text_list(sample_texts)

        print(f"Loaded {len(documents)} documents")

        # Create embeddings and vector store
        embeddings = DPREmbeddings()
        self.vectorstore = FAISS.from_documents(documents, embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

    def _setup_llm(self):
        """Setup the language model pipeline"""
        print("Setting up language model...")

        try:
            # Use a more suitable model for text generation
            model_name = "google/gemma-2b-it"

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )

            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            self.llm = HuggingFacePipeline(pipeline=self.pipe)

        except Exception as e:
            print(f"Error setting up LLM: {e}")
            print("Falling back to Ollama...")
            self.llm = None

    def _setup_conversation_chain(self):
        """Setup the conversational retrieval chain"""
        if self.llm:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            self.rag_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=self.memory
            )
        else:
            self.rag_chain = None

    def classify_comment(self, comment: str) -> str:
        """Classify the antisemitic comment"""
        try:
            return self.classifier.predict(comment)
        except Exception as e:
            print(f"Error classifying comment: {e}")
            return "general antisemitism"

    def get_relevant_documents(self, query: str) -> str:
        """Get relevant documents for the query"""
        try:
            if self.rag_chain:
                response = self.rag_chain({"question": query})
                return response.get("answer", "No relevant documents found.")
            else:
                # Fallback: direct retrieval
                docs = self.retriever.get_relevant_documents(query)
                return "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return "Error retrieving relevant information."

    @staticmethod
    def generate_ollama_response(model: str, prompt: str) -> Dict[str, Any]:
        """Generate response using Ollama"""
        try:
            print(f"Generating response with {model}...")
            client = ollama.Client()
            response = client.generate(model=model, prompt=prompt)
            return {
                'success': True,
                'response': response.response,
                'model': model
            }
        except Exception as e:
            print(f"Error with Ollama: {e}")
            return {
                'success': False,
                'response': f"Error generating response: {e}",
                'model': model
            }

    def evaluate_response(self, comment: str, label: str, response: str) -> Dict[str, Any]:
        """Evaluate the generated response"""
        try:
            return self.evaluator.evaluate_response(comment, label, response)
        except Exception as e:
            print(f"Error evaluating response: {e}")
            return {"error": f"Evaluation failed: {e}"}

    def process_comment(self, comment: str, ollama_model: str = "llama3.2") -> Dict[str, Any]:
        """Process a comment end-to-end"""
        print(f"Processing comment: {comment[:100]}...")

        # Step 1: Classify the comment
        label = self.classify_comment(comment)
        print(f"Classification: {label}")

        # Step 2: Create document retrieval prompt
        documents_prompt = (
            f"I need information about {label} antisemitism to respond to this comment: "
            f"'{comment}'. Please provide educational resources, references, and factual "
            f"information that can help counter this type of antisemitism."
        )

        # Step 3: Get relevant documents
        relevant_info = self.get_relevant_documents(documents_prompt)
        print(f"Retrieved {len(relevant_info)} characters of relevant information")

        # Step 4: Create response generation prompt
        response_prompt = (
            f"I encountered this antisemitic comment: '{comment}'\n\n"
            f"This comment represents {label} antisemitism. I want to respond politely but "
            f"educationally to counter this antisemitism.\n\n"
            f"Based on this information: {relevant_info}\n\n"
            f"Please help me write a respectful response that:\n"
            f"1. Explains why the comment is problematic\n"
            f"2. Educates about {label} antisemitism\n"
            f"3. Provides factual information to counter misconceptions\n"
            f"4. Remains civil and constructive\n\n"
            f"Response:"
        )

        # Step 5: Generate response
        ollama_response = self.generate_ollama_response(ollama_model, response_prompt)

        # Step 6: Evaluate response
        evaluation_results = {}
        if ollama_response['success']:
            evaluation_results = self.evaluate_response(
                comment, label, ollama_response['response']
            )

        return {
            'comment': comment,
            'classification': label,
            'relevant_documents': relevant_info,
            'generated_response': ollama_response,
            'evaluation': evaluation_results
        }


def main():
    """Main function to demonstrate the system"""

    # Initialize system (you can specify Excel file path and config here)
    system = AntisemitismResponseSystem()

    # Example comment
    comment = "why do those penguins always have to not fit to the society, they interfere the marathon runners, they should respect them even if it is on their neighborhood and on sabat"

    # Process the comment
    results = system.process_comment(comment)

    # Display results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Original Comment: {results['comment']}")
    print(f"Classification: {results['classification']}")
    print(f"\nGenerated Response:")
    if results['generated_response']['success']:
        print(results['generated_response']['response'])
    else:
        print(f"Error: {results['generated_response']['response']}")

    print(f"\nEvaluation Results:")
    evaluation = results['evaluation']
    if evaluation:
        if "scores" in evaluation:
            print("Scores:")
            for metric, score in evaluation["scores"].items():
                print(f"  - {metric}: {score:.2f}")

        if "feedback" in evaluation:
            print("Feedback:")
            for metric, feedback in evaluation["feedback"].items():
                print(f"  - {metric}: {feedback}")

        if "improvement_suggestions" in evaluation:
            print("Improvement Suggestions:")
            for suggestion in evaluation["improvement_suggestions"]:
                print(f"  - {suggestion}")


if __name__ == "__main__":
    main()
