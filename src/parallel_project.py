import logging
import os
from typing import Dict, Any, List, Optional, Tuple
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from ollama import Client
from langchain.schema import Document
from dataclasses import dataclass
from enum import Enum
import json
import time
import asyncio
import aiohttp
from functools import lru_cache

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

HF_TOKEN = "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"


class AntisemitismCategory(Enum):
    """Enum for antisemitism categories with clear definitions"""
    NON_ANTISEMITIC = (1, "Non Anti-Semitic", "Content that does not contain antisemitic elements")
    ANTISEMITIC_IDEOLOGY = (2, "Antisemitic Ideology",
                            "Calling for annihilation, conspiracy theories, identifying Jewish people as evil")
    STEREOTYPES_DEHUMANIZATION = (3, "Stereotypes and Dehumanization",
                                  "Economic stereotypes, biological racism, humiliating external imaging")
    ANTI_ISRAEL_ZIONISM = (4, "Antisemitism against Israel or Zionism",
                           "Demonizing Israel, comparing Israel/Zionism to Nazis, anti-Zionism, denial of right to exist")
    HOLOCAUST_DENIAL = (5, "Holocaust Denial and Sarcasm",
                        "Direct Holocaust denial and implied denial, sarcasm about the Holocaust")
    INDIRECT_ANTISEMITISM = (6, "Indirect Antisemitism",
                             "Mentioning Jewish/Israeli public figures with antisemitic connotations, implying Jewish people")

    def __init__(self, id: int, name: str, description: str):
        self.id = id
        self.category_name = name
        self.description = description

    @classmethod
    @lru_cache(maxsize=1)
    def get_all_categories(cls) -> Dict[int, 'AntisemitismCategory']:
        """Get all categories as a dictionary (cached)"""
        return {cat.id: cat for cat in cls}

    @classmethod
    @lru_cache(maxsize=1)
    def get_category_descriptions(cls) -> str:
        """Get formatted category descriptions for prompts (cached)"""
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


class OptimizedDocumentClassifier:
    """Optimized ReAct agent for categorizing educational documents against antisemitic content"""

    def __init__(self, model_name: str = "llama3", max_workers: int = 4, batch_size: int = 10):
        """
        Initialize the OptimizedDocumentClassifier

        Args:
            model_name: Name of the Ollama model to use
            max_workers: Number of parallel workers for processing
            batch_size: Batch size for processing documents
        """
        self.model_name = model_name
        self.ollama_host: str = 'http://localhost:11434'
        self.client_pool = []
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.categories = AntisemitismCategory.get_all_categories()
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize multiple Ollama clients for parallel processing"""
        try:
            for i in range(self.max_workers):
                client = Client(host=self.ollama_host)
                # Test connection
                client.list()
                self.client_pool.append(client)
            logger.info(f"Successfully initialized {self.max_workers} Ollama clients with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing Ollama clients: {e}")
            raise ConnectionError(f"Failed to connect to Ollama at {self.ollama_host}: {e}")

    @lru_cache(maxsize=100)
    def _create_batch_prompt(self, document_content: str) -> str:
        """Create an optimized prompt for batch classification of all categories"""
        # Truncate content more aggressively for speed
        truncated_content = document_content[:1500] + "..." if len(document_content) > 1500 else document_content

        return f"""### TASK: Multi-Category Document Analysis for Antisemitism Education

Analyze this educational document for relevance to ALL antisemitism categories simultaneously.

### ANTISEMITISM CATEGORIES:
{AntisemitismCategory.get_category_descriptions()}

### DOCUMENT:
{truncated_content}

### INSTRUCTIONS:
For EACH category (1-6), determine if this document provides educational content to address that type of antisemitism.

### RESPONSE FORMAT (be concise):
Category 1: [YES/NO] - [Brief reason]
Category 2: [YES/NO] - [Brief reason]  
Category 3: [YES/NO] - [Brief reason]
Category 4: [YES/NO] - [Brief reason]
Category 5: [YES/NO] - [Brief reason]
Category 6: [YES/NO] - [Brief reason]

### RESPONSE:"""

    def _parse_batch_response(self, response: str) -> List[RelevanceResult]:
        """Parse the batch LLM response for all categories"""
        try:
            lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
            results = []

            categories_list = list(AntisemitismCategory)

            for i, category in enumerate(categories_list, 1):
                # Find the line for this category
                category_line = next((line for line in lines if line.startswith(f'Category {i}:')), '')

                if category_line:
                    # Extract YES/NO and explanation
                    content = category_line.replace(f'Category {i}:', '').strip()
                    is_relevant = content.upper().startswith('YES')

                    # Extract explanation after the dash
                    parts = content.split(' - ', 1)
                    explanation = parts[1] if len(parts) > 1 else content

                    # Simple confidence scoring based on response clarity
                    confidence = 0.8 if ('YES' in content.upper() or 'NO' in content.upper()) else 0.5

                    results.append(RelevanceResult(
                        is_relevant=is_relevant,
                        explanation=explanation,
                        confidence_score=confidence,
                        category=category
                    ))
                else:
                    # Default response if category not found
                    results.append(RelevanceResult(
                        is_relevant=False,
                        explanation="No response found for this category",
                        confidence_score=0.3,
                        category=category
                    ))

            return results

        except Exception as e:
            logger.warning(f"Error parsing batch response: {e}")
            # Return default results for all categories
            return [RelevanceResult(
                is_relevant=False,
                explanation=f"Error parsing response: {str(e)}",
                confidence_score=0.0,
                category=category
            ) for category in AntisemitismCategory]

    def _classify_single_document(self, document: Document, client_index: int = 0) -> ClassificationResult:
        """Classify a single document using batch processing"""
        client = self.client_pool[client_index % len(self.client_pool)]
        prompt = self._create_batch_prompt(document.page_content)

        try:
            # Single API call for all categories
            result = client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.1,
                    'num_predict': 300,  # Limit response length for speed
                    'top_k': 10,  # Reduce sampling space
                    'top_p': 0.9  # Focus on high-probability tokens
                }
            )

            relevance_results = self._parse_batch_response(result['response'])
            relevant_categories = [res.category for res in relevance_results if res.is_relevant]

            return ClassificationResult(
                document_id=getattr(document, 'metadata', {}).get('id', None),
                categories=relevant_categories,
                relevance_results=relevance_results,
                success=True
            )

        except Exception as e:
            logger.error(f"Error classifying document: {e}")
            return ClassificationResult(
                document_id=getattr(document, 'metadata', {}).get('id', None),
                categories=[],
                relevance_results=[],
                success=False,
                error_message=str(e)
            )

    def classify_documents_batch(self, documents: List[Document]) -> List[ClassificationResult]:
        """Classify multiple documents in parallel"""
        logger.info(f"Starting batch classification of {len(documents)} documents")
        start_time = time.time()

        results = []

        # Process documents in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create futures for all documents
            futures = []
            for i, doc in enumerate(documents):
                future = executor.submit(self._classify_single_document, doc, i)
                futures.append(future)

            # Collect results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    result = future.result(timeout=30)  # 30 second timeout per document
                    results.append(result)

                    if (i + 1) % 10 == 0:  # Progress logging
                        logger.info(f"Processed {i + 1}/{len(documents)} documents")

                except concurrent.futures.TimeoutError:
                    logger.warning(f"Document classification timed out")
                    results.append(ClassificationResult(
                        document_id=None,
                        categories=[],
                        relevance_results=[],
                        success=False,
                        error_message="Classification timeout"
                    ))
                except Exception as e:
                    logger.error(f"Error in document classification: {e}")
                    results.append(ClassificationResult(
                        document_id=None,
                        categories=[],
                        relevance_results=[],
                        success=False,
                        error_message=str(e)
                    ))

        end_time = time.time()
        logger.info(f"Batch classification completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Average time per document: {(end_time - start_time) / len(documents):.2f} seconds")

        return results

    def process_dataframe_optimized(self, df: pd.DataFrame, content_col: str = 'content',
                                    source_col: str = 'source', url_col: str = 'url') -> pd.DataFrame:
        """Process entire dataframe with optimizations"""
        logger.info(f"Processing dataframe with {len(df)} rows")

        # Create Document objects
        documents = []
        for index, row in df.iterrows():
            doc = Document(
                page_content=str(row[content_col])[:2000],  # Truncate for speed
                metadata={"id": index, "source": row[source_col], "url": row[url_col]},
            )
            documents.append(doc)

        # Process in batches
        all_results = []
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            batch_results = self.classify_documents_batch(batch)
            all_results.extend(batch_results)

        # Update dataframe with results
        df_copy = df.copy()
        df_copy['primary_categories'] = None
        df_copy['document_quality'] = None
        df_copy['confidence_scores'] = None
        df_copy['classification_success'] = False
        df_copy['error_message'] = None

        for i, result in enumerate(all_results):
            if i < len(df_copy):
                df_copy.at[i, 'primary_categories'] = [cat.id for cat in result.categories] if result.success else []
                df_copy.at[i, 'classification_success'] = result.success
                df_copy.at[i, 'error_message'] = result.error_message

        return df_copy

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


# Optimized main execution
if __name__ == "__main__":
    BASE_PATH = r"/resources"
    KNOWLEDGE_BASE_PATH = r"/resources/all_collected_articles.csv"

    try:
        # Load and preprocess data more efficiently
        logger.info("Loading knowledge base...")
        knowledge_base = pd.read_csv(KNOWLEDGE_BASE_PATH)

        # Efficient data cleaning
        required_cols = ['source', 'url', 'content', 'categories']
        knowledge_base = knowledge_base[required_cols].copy()

        # Vectorized operations for faster filtering
        knowledge_base = knowledge_base.dropna(subset=['source', 'url', 'content'])
        knowledge_base = knowledge_base[knowledge_base['url'].str.startswith("http", na=False)]
        knowledge_base.reset_index(drop=True, inplace=True)

        logger.info(f"Processing {len(knowledge_base)} documents")

        # Initialize optimized classifier
        classifier = OptimizedDocumentClassifier(
            model_name="llama3",
            max_workers=4,  # Adjust based on your system
            batch_size=20  # Process 20 documents at a time
        )

        # Process all documents efficiently
        processed_df = classifier.process_dataframe_optimized(knowledge_base)

        # Save results
        output_path = os.path.join(BASE_PATH, "knowledge_base_categorized_optimized.csv")
        processed_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

        # Print summary statistics
        successful_classifications = processed_df['classification_success'].sum()
        logger.info(f"Successfully classified {successful_classifications}/{len(processed_df)} documents")

        # Show category distribution
        category_counts = {}
        for categories_list in processed_df['categories'].dropna():
            if isinstance(categories_list, list):
                for cat_id in categories_list:
                    category_counts[cat_id] = category_counts.get(cat_id, 0) + 1

        logger.info("Category distribution:")
        for cat_id, count in category_counts.items():
            cat_name = AntisemitismCategory.get_all_categories()[cat_id].category_name
            logger.info(f"  {cat_name}: {count} documents")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

