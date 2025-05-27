import logging
import os
import pandas as pd
from typing import Dict, Any
from ollama import Client

# RAG imports
from langchain.schema import Document
from business_logic.chatbot.documents_retriever import DocumentsRetriever

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)

HF_TOKEN = "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"

# Suppress Windows symlink warning for Hugging Face cache
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class ReActAgent:
    """ReAct agent for generating educational responses to antisemitic comments"""

    def __init__(self, knowledge_base: pd.DataFrame):
        self.knowledge_base = knowledge_base
        self.retriever = DocumentsRetriever(knowledge_base)
        self.pipe = None
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the Ollama LLaMA 3.2 model"""
        try:
            self.client = Client(host='http://localhost:11434')
            self.model_name = "llama3"
            logger.info("Successfully initialized Ollama LLM client")
        except Exception as e:
            logger.error(f"Error initializing Ollama LLM: {e}")
            raise

    @staticmethod
    def _build_category_prompt(comment: str, retrieval_result: Document, category_id: int) -> str:
        """Build category-specific prompts for different types of antisemitism"""

        category_instructions = {
            1: "Explain why this ideological claim is historically inaccurate and harmful, providing factual context.",
            2: "Address this harmful stereotype by explaining its origins and why it's false and damaging.",
            3: "Clarify why this statement delegitimizes Israel in ways defined as antisemitic by the IHRA definition.",
            4: "Counter this Holocaust denial with clear historical evidence and explain why denial is harmful.",
            5: "Address the misleading framing and explain the implicit bias in this statement."
        }

        instruction = category_instructions.get(category_id,
                                                "Provide a factual, educational response.")

        return (f"You are an educational assistant helping counter antisemitism with facts and respectful dialogue."
                f"\nUser's problematic statement: '{comment}'."
                f"\nFactual information from trusted source:\n{retrieval_result.page_content}."
                f"\nTask: {instruction}"
                f"\nRequirements: Write exactly 4 sentences, Be calm, respectful, and educational, "
                f"Use facts from the provided source, Include the source URL: {retrieval_result.metadata["url"]}, "
                f"Focus on education, not confrontation."
                f"\nResponse:")

    def _check_relevance(self, comment: str, paragraph: str) -> Dict[str, Any]:
        """Check if retrieved paragraph is relevant to the comment"""
        relevance_prompt = (
            "### Task:\n"
            "Does the following paragraph provide relevant information to address this statement?\n"
            f"Statement: '{comment}'\n"
            f"Paragraph: '{paragraph}'\n"
            "Answer with YES or NO and explain briefly why.\n\n"
            "### Answer:"
        )

        try:
            logger.info("Connecting llama to check paragraph's relevance")
            result = self.client.generate(model=self.model_name, prompt=relevance_prompt)['response']

            lines = result.splitlines()
            relevancy = lines[0]
            response = " ".join(lines[2:]).lower()

            if "YES" == relevancy:
                is_relevant = True
            elif "NO" == relevancy:
                is_relevant = False
            else:
                is_relevant = False  # Default fallback
            logger.info(f"The current paragraph is {"" if is_relevant else "not"} relevant")

            return {
                "is_relevant": is_relevant,
                "explanation": response
            }
        except Exception as e:
            logger.error(f"Error checking relevance: {e}")
            return {"is_relevant": True, "explanation": "Error in relevance check"}

    def generate_response(self, comment: str, category_id: int, category_name: str) -> Dict[str, Any]:
        """Generate educational response using ReAct methodology"""
        logger.info("Start generating ReAct response")

        # Thought: Plan the approach
        thought_1 = (f"I need to retrieve relevant factual information for antisemitism category {category_name}"
                     f"to counter the problematic statement: '{comment}'. I'll search for factual content "
                     f"that directly addresses this type of antisemitic claim.")

        # Action 1: Retrieve relevant information using the original comment as query
        action_1 = f"Retrieving documents related to antisemitic comment about category {category_name}"
        retrieval_results = self.retriever.retrieve(comment, category_id)

        # Observation 1: Evaluate retrieval results
        if not retrieval_results:
            observation_1 = "No relevant documents found in knowledge base"
            return {
                "thought": thought_1,
                "action_1": action_1,
                "observation_1": observation_1,
                "final_response": "I apologize, but I couldn't find specific information to address your comment. Please consult educational resources about antisemitism and historical facts.",
                "source": "None",
                "url": "None"
            }

        observation_1 = f"Retrieved {len(retrieval_results)} potentially relevant documents"
        logger.info(f"observation_1: {observation_1}")

        # Thought: Analyze retrieved results
        thought_2 = "Now I need to identify the most relevant document that directly addresses this antisemitic claim"

        # Action 2: Select most relevant result through relevance checking
        action_2 = "Evaluating relevance of retrieved documents to the antisemitic comment"
        best_result = None
        for result in retrieval_results:
            relevance_check = self._check_relevance(comment, result.page_content)
            if relevance_check["is_relevant"]:
                best_result = result
                break

        if not best_result:
            best_result = retrieval_results[0]  # Fallback to highest scored result

        # Observation 2: Document selection result
        observation_2 = f"Selected document from {best_result.metadata["source"]}"
        logger.info(f"observation_2: {observation_2}")

        # Thought: Generate educational response
        thought_3 = "Now I'll generate a 4-sentence educational response using the factual information"

        # Action 3: Generate response
        action_3 = "Generating educational response using selected factual information"
        prompt = self._build_category_prompt(comment, best_result, category_id)

        try:
            logger.info("Connecting llama to generate response")
            generated = self.client.generate(model=self.model_name, prompt=prompt)['response']

            # Extract only the response part
            response_text = generated.split("Response:")[-1].strip()

            # Ensure response ends with source
            if best_result.metadata["url"] and best_result.metadata["url"] not in response_text:
                response_text += f" URL: {best_result.metadata["url"]}"

            # Final Observation: Response generated successfully
            observation_3 = "Educational response generated successfully with factual backing"
            logger.info(f"observation_3: {observation_3}")

            return {
                "thought_1": thought_1,
                "action_1": action_1,
                "observation_1": observation_1,
                "thought_2": thought_2,
                "action_2": action_2,
                "observation_2": observation_2,
                "thought_3": thought_3,
                "action_3": action_3,
                "observation_3": observation_3,
                "final_response": response_text,
                "source": best_result.source,
                "url": best_result.url,
                "retrieval_score": best_result.score
            }

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "thought_1": thought_1,
                "action_1": action_1,
                "observation_1": observation_1,
                "error": "Generation failed",
                "final_response": f"Based on factual sources, this statement contains inaccuracies. {best_result.page_content[:200]}... Please refer to: {best_result.url}",
                "source": best_result.source,
                "url": best_result.url
            }

