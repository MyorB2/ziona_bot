import logging
import os
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
from ollama import Client
from langchain.schema import Document
from dataclasses import dataclass
from enum import Enum
import json
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)

# Suppress Windows symlink warning for Hugging Face cache
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)

HF_TOKEN = "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"

# Suppress Windows symlink warning for Hugging Face cache
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class AntisemitismCategory(Enum):
    """Enum for antisemitism categories with clear definitions"""
    ANTISEMITIC_IDEOLOGY = (1, "Antisemitic Ideology",
                            "Calling for annihilation, conspiracy theories, identifying Jewish people as evil")
    STEREOTYPES_DEHUMANIZATION = (2, "Stereotypes and Dehumanization",
                                  "Economic stereotypes, biological racism, humiliating external imaging")
    ANTI_ISRAEL_ZIONISM = (3, "Antisemitism against Israel or Zionism",
                           "Demonizing Israel, comparing Israel/Zionism to Nazis, anti-Zionism, denial of right to exist")
    HOLOCAUST_DENIAL = (4, "Holocaust or Zionism Denial",
                        "Being cynical about the Holocaust, Holocaust denial")
    INDIRECT_ANTISEMITISM = (5, "Indirect Antisemitism",
                             "Mentioning Jewish/Israeli public figures with antisemitic connotations, implying Jewish people")

    def __init__(self, id: int, name: str, description: str):
        self.id = id
        self.category_name = name
        self.description = description

    @classmethod
    def get_all_categories(cls) -> Dict[int, 'AntisemitismCategory']:
        """Get all categories as a dictionary"""
        return {cat.id: cat for cat in cls}

    @classmethod
    def get_category_descriptions(cls) -> str:
        """Get formatted category descriptions for prompts"""
        descriptions = []
        for cat in cls:
            descriptions.append(f"{cat.id}. {cat.category_name}: {cat.description}")
        return "\n".join(descriptions)


@dataclass
class RelevanceResult:
    """Data class for relevance check results"""
    is_relevant: bool
    explanation: str
    confidence_score: Optional[float] = None
    category: Optional[AntisemitismCategory] = None


@dataclass
class ClassificationResult:
    """Data class for document classification results"""
    document_id: Optional[str]
    categories: List[AntisemitismCategory]
    relevance_results: List[RelevanceResult]
    success: bool
    error_message: Optional[str] = None


class DocumentClassifier:
    """Enhanced ReAct agent for categorizing educational documents against antisemitic content"""

    def __init__(self, document: Document, model_name: str = "llama3"):
        """
        Initialize the DocumentClassifier

        Args:
            document: The document to classify
            model_name: Name of the Ollama model to use
            ollama_host: Ollama server host URL
        """
        self.document = document
        self.model_name = model_name
        self.ollama_host: str = 'http://localhost:11434'
        self.client = None
        self.categories = AntisemitismCategory.get_all_categories()
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the Ollama LLaMA model with error handling"""
        try:
            self.client = Client(host=self.ollama_host)
            # Test connection
            self.client.list()
            logger.info(f"Successfully initialized Ollama LLM client with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing Ollama LLM: {e}")
            raise ConnectionError(f"Failed to connect to Ollama at {self.ollama_host}: {e}")

    def _create_relevance_prompt(self, category: AntisemitismCategory, document_content: str) -> str:
        """Create a well-structured prompt for relevance checking"""
        return f"""### TASK: Document Relevance Analysis for Antisemitism Education

You are analyzing whether an educational document is relevant for addressing a specific type of antisemitism.

### ANTISEMITISM CATEGORIES:
{AntisemitismCategory.get_category_descriptions()}

### TARGET CATEGORY: 
{category.id}. {category.category_name}: {category.description}

### DOCUMENT TO ANALYZE:
{document_content[:2000]}{"..." if len(document_content) > 2000 else ""}

### INSTRUCTIONS:
Determine if this document provides relevant educational content to address or counter the target antisemitism category.

Consider:
- Does the document provide factual information that counters this type of antisemitism?
- Does it offer educational content about this specific category?
- Would this document be useful in responding to comments of this antisemitism type?

### RESPONSE FORMAT:
Answer: [YES/NO]
Confidence: [HIGH (confidence>=0.9) /MEDIUM (0.5<confidence<0.9) /LOW (confidence<=0.5)]
Explanation: [Brief justification for your decision]

### RESPONSE:"""

    @staticmethod
    def _parse_relevance_response(response: str, category: AntisemitismCategory) -> RelevanceResult:
        """Parse the LLM response for relevance checking"""
        try:
            lines = [line.strip() for line in response.strip().split('\n') if line.strip()]

            # Extract answer
            answer_line = next((line for line in lines if line.startswith('Answer:')), '')
            is_relevant = 'YES' in answer_line.upper()

            # Extract confidence
            confidence_line = next((line for line in lines if line.startswith('Confidence:')), '')
            confidence_map = {'HIGH': 0.9, 'MEDIUM': 0.7, 'LOW': 0.5}
            confidence_score = confidence_map.get(
                confidence_line.replace('Confidence:', '').strip().upper(), 0.5
            )

            # Extract explanation
            explanation_line = next((line for line in lines if line.startswith('Explanation:')), '')
            explanation = explanation_line.replace('Explanation:', '').strip()

            if not explanation:
                explanation = ' '.join(lines[2:]) if len(lines) > 2 else "No explanation provided"

            return RelevanceResult(
                is_relevant=is_relevant,
                explanation=explanation,
                confidence_score=confidence_score,
                category=category
            )

        except Exception as e:
            logger.warning(f"Error parsing relevance response: {e}")
            return RelevanceResult(
                is_relevant=False,
                explanation=f"Error parsing response: {str(e)}",
                confidence_score=0.0,
                category=category
            )

    def _check_relevance(self, category: AntisemitismCategory) -> RelevanceResult:
        """Check if document is relevant to the specified antisemitism category"""
        prompt = self._create_relevance_prompt(category, self.document.page_content)

        try:
            logger.info(f"Checking relevance for category: {category.category_name}")

            # Add retry logic for robustness
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = self.client.generate(
                        model=self.model_name,
                        prompt=prompt,
                        options={'temperature': 0.1}  # Lower temperature for more consistent results
                    )

                    relevance_result = self._parse_relevance_response(result['response'], category)

                    logger.info(f"Category {category.category_name}: "
                                f"{'Relevant' if relevance_result.is_relevant else 'Not relevant'} "
                                f"(Confidence: {relevance_result.confidence_score})")

                    return relevance_result

                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(1)

        except Exception as e:
            logger.error(f"Error checking relevance for category {category.category_name}: {e}")
            return RelevanceResult(
                is_relevant=False,
                explanation=f"Error during relevance check: {str(e)}",
                confidence_score=0.0,
                category=category
            )

    def categorize(self) -> ClassificationResult:
        """
        Categorize the document using ReAct methodology

        Returns:
            ClassificationResult with categories and detailed information
        """
        start_time = time.time()
        logger.info("Starting document categorization using ReAct methodology")

        try:
            # Thought: Plan the approach
            logger.info("THOUGHT: Planning categorization approach for antisemitism education document")

            # Action: Check relevance for each category
            logger.info("ACTION: Checking document relevance against all antisemitism categories")

            relevant_categories = []
            all_relevance_results = []

            for category in AntisemitismCategory:
                relevance_result = self._check_relevance(category)
                all_relevance_results.append(relevance_result)

                if relevance_result.is_relevant:
                    relevant_categories.append(category)

            # Observation: Analyze results
            logger.info(f"OBSERVATION: Document categorized into {len(relevant_categories)} "
                        f"out of {len(AntisemitismCategory)} categories")

            if relevant_categories:
                category_names = [cat.category_name for cat in relevant_categories]
                logger.info(f"Relevant categories: {', '.join(category_names)}")
            else:
                logger.info("Document not relevant to any antisemitism categories")

            return ClassificationResult(
                document_id=getattr(self.document, 'metadata', {}).get('id', None),
                categories=relevant_categories,
                relevance_results=all_relevance_results,
                success=True
            )

        except Exception as e:
            logger.error(f"Error during document categorization: {e}")

            return ClassificationResult(
                document_id=getattr(self.document, 'metadata', {}).get('id', None),
                categories=[],
                relevance_results=[],
                success=False,
                error_message=str(e)
            )

    @staticmethod
    def export_results_json(result: ClassificationResult) -> str:
        """Export classification results as JSON"""
        export_data = {
            "document_id": result.document_id,
            "success": result.success,
            "categories": [
                {
                    "id": cat.id,
                    "name": cat.category_name,
                    "description": cat.description
                } for cat in result.categories
            ],
            "relevance_details": [
                {
                    "category_id": res.category.id,
                    "category_name": res.category.category_name,
                    "is_relevant": res.is_relevant,
                    "confidence_score": res.confidence_score,
                    "explanation": res.explanation
                } for res in result.relevance_results
            ],
            "error_message": result.error_message
        }

        return json.dumps(export_data, indent=2, ensure_ascii=False)


# Example usage and testing
if __name__ == "__main__":

    BASE_PATH = r"C:\Users\myor1\PycharmProjects\ziona_bot\resources"
    KNOWLEDGE_BASE_PATH = r"C:\Users\myor1\PycharmProjects\ziona_bot\resources\all_collected_articles.csv"
    knowledge_base = pd.read_csv(KNOWLEDGE_BASE_PATH)
    knowledge_base = knowledge_base[['source', 'url', 'content', 'categories']]
    knowledge_base = knowledge_base.dropna(subset=['source', 'url', 'content'])
    knowledge_base = knowledge_base[knowledge_base['url'].apply(lambda x: x.startswith("http"))]
    knowledge_base.reset_index(drop=True, inplace=True)

    try:
        for index, row in knowledge_base.iterrows():
            doc = Document(
                page_content=row["content"],
                metadata={"id": index, "source": row["source"], "url": row["url"]},
            )
            classifier = DocumentClassifier(doc)
            result = classifier.categorize()
            knowledge_base.at[index, 'categories'] = [cat.id for cat in result.categories]

            print("Classification Results:")
            print(f"Success: {result.success}")
            print(f"Categories found: {len(result.categories)}")
            for cat in result.categories:
                print(f"  - {cat.category_name}")

            if index < 10:
                # Export as JSON
                json_result = classifier.export_results_json(result)
                print("\nJSON Export:")
                with open(BASE_PATH + rf"\results_{index}.json", "w", encoding="utf-8") as f:
                    json.dump(json_result, f, indent=4, ensure_ascii=False)
        # Export to CSV
        knowledge_base.to_csv(BASE_PATH + rf"\knowledge_base.csv", index=False)
    except Exception as e:
        logger.error(f"Error in example usage: {e}")
