import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
import re
import pandas as pd
from ollama import Client
from langchain.schema import Document
from dataclasses import dataclass
from enum import Enum
import json
import time

from src.utils import normalize_categories

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

ROOT_PATH = Path(__file__).parent.parent.parent
RESOURCE_PATH = ROOT_PATH / "resources"
KNOWLEDGE_BASE_PATH = RESOURCE_PATH / "knowledge_base.csv"
CONFIDENCE_THRESHOLD = 0.55


def read_csv_with_encoding(file_path):
    encodings = ['utf-8', 'latin-1', 'windows-1252', 'cp1252', 'iso-8859-1']

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"Successfully read file with encoding: {encoding}")
            return df
        except UnicodeDecodeError:
            print(f"Failed to read with encoding: {encoding}")
            continue

    raise ValueError("Could not read the file with any of the attempted encodings")


class AntisemitismCategory(Enum):
    """Enum for antisemitism categories with clear definitions"""
    GENERAL_HATE = (1, "General Hate/Other",
                    "Educational content addressing general hatred, discrimination, or forms of prejudice that are not specifically antisemitic")
    ANTISEMITIC_IDEOLOGY = (2, "Antisemitic Ideology",
                            "Educational content countering calls for annihilation, conspiracy theories, or portraying Jewish people as evil")
    STEREOTYPES_DEHUMANIZATION = (3, "Stereotypes and Dehumanization",
                                  "Educational content countering economic stereotypes, biological racism, or humiliating external imaging of Jewish people")
    ANTI_ISRAEL_ZIONISM = (4, "Anti-Israel/Anti-Zionism",
                           "Educational content countering demonization of Israel, Nazi comparisons, anti-Zionism, or denial of Israel's right to exist")
    HOLOCAUST_DENIAL = (5, "Holocaust Denial and Sarcasm",
                        "Educational content countering Holocaust denial, minimization, or sarcastic references to the Holocaust")
    INDIRECT_ANTISEMITISM = (6, "Indirect Antisemitism",
                             "Educational content countering mentions of Jewish/Israeli public figures with antisemitic implications")

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
class CategoryScore:
    """Data class for individual category scoring"""
    category: AntisemitismCategory
    score: float  # 0-1 confidence score
    reasoning: str
    addresses_category: bool


@dataclass
class ClassificationResult:
    """Data class for document classification results"""
    document_id: Optional[str]
    source: Optional[str]
    url: Optional[str]
    primary_categories: List[AntisemitismCategory]  # Categories with high confidence
    all_scores: List[CategoryScore]
    document_quality: str  # "HIGH", "MEDIUM", "LOW"
    success: bool
    error_message: Optional[str] = None


class DocumentClassifier:
    """Enhanced classifier for educational documents addressing antisemitic content"""

    def __init__(self, document: Document, model_name: str = "llama3"):
        """
        Initialize the DocumentClassifier

        Args:
            document: The document to classify
            model_name: Name of the Ollama model to use
        """
        self.document = document
        self.model_name = model_name
        self.ollama_host: str = 'http://localhost:11434'
        self.client = None
        self.categories = AntisemitismCategory.get_all_categories()
        self.min_content_length = 100  # Minimum content length for classification
        self.confidence_threshold = CONFIDENCE_THRESHOLD  # Minimum confidence for category assignment
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

    def _assess_document_quality(self, content: str) -> str:
        """Assess the quality and length of document content"""
        content_length = len(content.strip())
        word_count = len(content.split())

        if content_length < self.min_content_length or word_count < 20:
            return "LOW"
        elif content_length > 1000 and word_count > 200:
            return "HIGH"
        else:
            return "MEDIUM"

    def _create_classification_prompt(self, document_content: str) -> str:
        """Create a comprehensive classification prompt"""
        # Truncate content if too long to avoid token limits
        max_content_length = 3000
        if len(document_content) > max_content_length:
            document_content = document_content[:max_content_length] + "...[content truncated]"

        return f"""### TASK: Educational Counter-Antisemitism Document Classification

You are analyzing EDUCATIONAL DOCUMENTS that fight against antisemitism. These documents provide facts, education, and responses to counter antisemitic narratives.

**CRITICAL UNDERSTANDING**: 
- ALL documents are educational resources AGAINST antisemitism
- NO documents contain antisemitic content themselves
- You are categorizing which types of antisemitic narratives each document helps to counter

### CLASSIFICATION CATEGORIES:
{AntisemitismCategory.get_category_descriptions()}

### EDUCATIONAL DOCUMENT TO ANALYZE:
{document_content}

### YOUR TASK:
For each category (1-6), determine if this educational document provides information, facts, or arguments that would be useful for COUNTERING that specific type of antisemitic narrative.

**EXAMPLES OF WHAT TO LOOK FOR:**
- Category 2 (Antisemitic Ideology): Does the document provide facts about Jewish contributions to society, debunk conspiracy theories, or explain Jewish history/culture positively?
- Category 3 (Stereotypes): Does the document counter economic myths about Jewish people, provide factual information about Jewish diversity, or address racist stereotypes?
- Category 4 (Anti-Israel/Anti-Zionism): Does the document provide factual information about Israel's history, explain Zionism accurately, or counter false narratives about Israel?
- Category 5 (Holocaust): Does the document provide Holocaust education, historical facts, or counter denial arguments?
- Category 6 (Indirect): Does the document address how public figures are unfairly targeted, or explain the harm of antisemitic dog whistles?
- Category 1 (General Hate): Does the document address general prejudice, discrimination, or hatred that isn't specifically antisemitic?

### RESPONSE FORMAT:

CATEGORY_1_GENERAL_HATE:
ADDRESSES: [YES/NO]
CONFIDENCE: [0.0-1.0]
REASONING: [Explain what content in the document helps counter general hate/discrimination]

CATEGORY_2_ANTISEMITIC_IDEOLOGY:
ADDRESSES: [YES/NO]
CONFIDENCE: [0.0-1.0]
REASONING: [Explain what content helps counter antisemitic ideologies/conspiracy theories]

CATEGORY_3_STEREOTYPES_DEHUMANIZATION:
ADDRESSES: [YES/NO]
CONFIDENCE: [0.0-1.0]
REASONING: [Explain what content helps counter stereotypes about Jewish people]

CATEGORY_4_ANTI_ISRAEL_ZIONISM:
ADDRESSES: [YES/NO]
CONFIDENCE: [0.0-1.0]
REASONING: [Explain what content helps counter anti-Israel/anti-Zionist narratives]

CATEGORY_5_HOLOCAUST_DENIAL:
ADDRESSES: [YES/NO]
CONFIDENCE: [0.0-1.0]
REASONING: [Explain what content helps counter Holocaust denial/minimization]

CATEGORY_6_INDIRECT_ANTISEMITISM:
ADDRESSES: [YES/NO]
CONFIDENCE: [0.0-1.0]
REASONING: [Explain what content helps counter indirect antisemitic targeting]

### BEGIN CLASSIFICATION:"""

    def _parse_classification_response(self, response: str) -> List[CategoryScore]:
        """Parse the comprehensive LLM response"""
        category_scores = []

        try:
            # Split response into category sections
            category_sections = re.split(r'CATEGORY_\d+_[A-Z_]+:', response)
            category_sections = [section.strip() for section in category_sections if section.strip()]

            # Match categories with their enum values
            category_mapping = {
                'GENERAL_HATE': AntisemitismCategory.GENERAL_HATE,
                'ANTISEMITIC_IDEOLOGY': AntisemitismCategory.ANTISEMITIC_IDEOLOGY,
                'STEREOTYPES_DEHUMANIZATION': AntisemitismCategory.STEREOTYPES_DEHUMANIZATION,
                'ANTI_ISRAEL_ZIONISM': AntisemitismCategory.ANTI_ISRAEL_ZIONISM,
                'HOLOCAUST_DENIAL': AntisemitismCategory.HOLOCAUST_DENIAL,
                'INDIRECT_ANTISEMITISM': AntisemitismCategory.INDIRECT_ANTISEMITISM
            }

            # Find category sections in the response
            for category_key, category_enum in category_mapping.items():
                pattern = rf'CATEGORY_\d+_{category_key}:(.*?)(?=CATEGORY_\d+_|$)'
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)

                if match:
                    section_content = match.group(1).strip()

                    # Extract ADDRESSES
                    addresses_match = re.search(r'ADDRESSES:\s*(YES|NO)', section_content, re.IGNORECASE)
                    addresses_category = addresses_match.group(1).upper() == 'YES' if addresses_match else False

                    # Extract CONFIDENCE
                    confidence_match = re.search(r'CONFIDENCE:\s*([\d.]+)', section_content)
                    confidence_score = float(confidence_match.group(1)) if confidence_match else 0.0
                    confidence_score = max(0.0, min(1.0, confidence_score))  # Clamp between 0 and 1

                    # Extract REASONING
                    reasoning_match = re.search(r'REASONING:\s*(.*)', section_content, re.DOTALL)
                    reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"

                    category_scores.append(CategoryScore(
                        category=category_enum,
                        score=confidence_score,
                        reasoning=reasoning,
                        addresses_category=addresses_category
                    ))
                else:
                    # If category not found in response, add default
                    category_scores.append(CategoryScore(
                        category=category_enum,
                        score=0.0,
                        reasoning="Category not analyzed in response",
                        addresses_category=False
                    ))

            return category_scores

        except Exception as e:
            logger.error(f"Error parsing classification response: {e}")
            # Return default scores for all categories
            return [
                CategoryScore(
                    category=cat,
                    score=0.0,
                    reasoning=f"Error parsing response: {str(e)}",
                    addresses_category=False
                ) for cat in AntisemitismCategory
            ]

    def classify(self) -> ClassificationResult:
        """
        Classify the document comprehensively

        Returns:
            ClassificationResult with detailed analysis
        """
        start_time = time.time()
        logger.info("Starting comprehensive document classification")

        try:
            # Assess document quality first
            document_quality = self._assess_document_quality(self.document.page_content)
            logger.info(f"Document quality assessed as: {document_quality}")

            if document_quality == "LOW":
                logger.warning("Low quality document detected - results may be unreliable")

            # Create and execute classification prompt
            prompt = self._create_classification_prompt(self.document.page_content)

            logger.info("Sending classification request to LLM...")

            # Add retry logic for robustness
            max_retries = 3
            response = None

            for attempt in range(max_retries):
                try:
                    result = self.client.generate(
                        model=self.model_name,
                        prompt=prompt,
                        options={
                            'temperature': 0.2,  # Low temperature for consistency
                            'top_p': 0.9,
                            'num_predict': 2000  # Allow longer responses
                        }
                    )
                    response = result['response']
                    break

                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(2)

            # Parse the response
            all_scores = self._parse_classification_response(response)

            # Determine primary categories (high confidence + addresses category)
            primary_categories = [
                score.category for score in all_scores
                if score.addresses_category and score.score >= self.confidence_threshold
            ]

            # If no primary categories found, check if it's general hate
            if not primary_categories:
                general_hate_score = next(
                    (score for score in all_scores if score.category == AntisemitismCategory.GENERAL_HATE),
                    None
                )
                if general_hate_score and general_hate_score.score >= self.confidence_threshold:
                    primary_categories = [AntisemitismCategory.GENERAL_HATE]

            # Log results
            logger.info(f"Classification completed in {time.time() - start_time:.2f} seconds")
            logger.info(f"Primary categories: {[cat.category_name for cat in primary_categories]}")

            for score in all_scores:
                if score.addresses_category:
                    logger.info(f"  {score.category.category_name}: {score.score:.2f} confidence")

            return ClassificationResult(
                document_id=getattr(self.document, 'metadata', {}).get('id', None),
                source=getattr(self.document, 'metadata', {}).get('source', None),
                url=getattr(self.document, 'metadata', {}).get('url', None),
                primary_categories=primary_categories,
                all_scores=all_scores,
                document_quality=document_quality,
                success=True
            )

        except Exception as e:
            logger.error(f"Error during document classification: {e}")
            return ClassificationResult(
                document_id=getattr(self.document, 'metadata', {}).get('id', None),
                source=getattr(self.document, 'metadata', {}).get('source', None),
                url=getattr(self.document, 'metadata', {}).get('url', None),
                primary_categories=[],
                all_scores=[],
                document_quality="UNKNOWN",
                success=False,
                error_message=str(e)
            )

    @staticmethod
    def export_results_json(result: ClassificationResult) -> str:
        """Export classification results as comprehensive JSON"""
        export_data = {
            "document_id": result.document_id,
            "success": result.success,
            "document_quality": result.document_quality,
            "primary_categories": [
                {
                    "id": cat.id,
                    "name": cat.category_name,
                    "description": cat.description
                } for cat in result.primary_categories
            ],
            "detailed_scores": [
                {
                    "category_id": score.category.id,
                    "category_name": score.category.category_name,
                    "addresses_category": score.addresses_category,
                    "confidence_score": score.score,
                    "reasoning": score.reasoning
                } for score in result.all_scores
            ],
            "error_message": result.error_message,
            "classification_summary": {
                "total_categories_addressed": len(result.primary_categories),
                "high_confidence_scores": len([s for s in result.all_scores if s.score >= 0.8]),
                "avg_confidence": sum(s.score for s in result.all_scores) / len(
                    result.all_scores) if result.all_scores else 0
            }
        }

        return json.dumps(export_data, indent=2, ensure_ascii=False)


# Enhanced example usage
if __name__ == "__main__":
    try:
        # Load and prepare data
        knowledge_base = read_csv_with_encoding(str(KNOWLEDGE_BASE_PATH))
        knowledge_base = knowledge_base[['source', 'url', 'content']]
        knowledge_base = knowledge_base.dropna(subset=['source', 'url', 'content'])
        knowledge_base = knowledge_base[knowledge_base['url'].apply(lambda x: x.startswith("http"))]
        knowledge_base = knowledge_base[knowledge_base['content'].apply(lambda x: len(str(x)) > 200)]
        knowledge_base.reset_index(drop=True, inplace=True)

        # Run only over 200 rows
        knowledge_base = knowledge_base.iloc[234:435]

        logger.info(f"Processing {len(knowledge_base)} documents")

        # Add new columns for enhanced results
        knowledge_base['primary_categories'] = None
        knowledge_base['confidence_scores'] = None
        knowledge_base['document_quality'] = None

        successful_classifications = 0

        for index, row in knowledge_base.iterrows():
            logger.info(f"Processing document {index + 1}/{len(knowledge_base)}")

            doc = Document(
                page_content=str(row["content"]),
                metadata={"id": index, "source": row["source"], "url": row["url"]},
            )

            classifier = DocumentClassifier(doc, model_name="llama3")
            result = classifier.classify()
            if result.success:
                successful_classifications += 1

                # Store results in dataframe
                knowledge_base.at[index, 'primary_categories'] = [cat.id for cat in result.primary_categories]
                knowledge_base.at[index, 'document_quality'] = result.document_quality
                knowledge_base.at[index, 'confidence_scores'] = {
                    score.category.id: score.score for score in result.all_scores
                }
                knowledge_base.to_csv(str(RESOURCE_PATH / "knowledge_base_categorized.csv"), index=False)
                # # Print summary for first 5 documents
                # if index < 5:
                #     print(f"\n=== Document {index} Classification Results ===")
                #     print(f"Source: {result.source}")
                #     print(f"URL: {result.url}")
                #     print(f"Success: {result.success}")
                #     print(f"Quality: {result.document_quality}")
                #     print(f"Primary Categories ({len(result.primary_categories)}):")
                #     for cat in result.primary_categories:
                #         print(f"  - {cat.category_name}")
                #     print(f"High-confidence scores:")
                #     for score in result.all_scores:
                #         if score.score >= CONFIDENCE_THRESHOLD:
                #             print(f"  - {score.category.category_name}: {score.score:.2f}")

                # Export detailed JSON for first 10 documents
                if index < 10:
                    json_result = classifier.export_results_json(result)
                    with open(str(RESOURCE_PATH / f"enhanced_results_{index}.json"), "w", encoding="utf-8") as f:
                        f.write(json_result)

            else:
                logger.error(f"Failed to classify document {index}: {result.error_message}")

        # Export enhanced CSV
        knowledge_base["primary_categories"] = knowledge_base["primary_categories"].apply(
            lambda x: normalize_categories(x))
        knowledge_base = knowledge_base.dropna(subset=['primary_categories'])
        knowledge_base.to_csv(str(RESOURCE_PATH / "knowledge_base_categorized.csv"), index=False)

        # Print final statistics
        print(f"\n=== Classification Statistics ===")
        print(f"Total documents: {len(knowledge_base)}")
        print(f"Successful classifications: {successful_classifications}")
        print(f"Success rate: {successful_classifications / len(knowledge_base) * 100:.1f}%")

    except Exception as e:
        logger.error(f"Error in enhanced example usage: {e}")
        raise
