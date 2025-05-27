import numpy as np
import re
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from typing import Dict, List, Optional
import requests
from urllib.parse import urlparse


class ResponseEvaluator:
    def __init__(self):
        # Load models with error handling
        try:
            self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.sentiment_pipeline = pipeline("sentiment-analysis")
            self.toxicity_pipeline = pipeline("text-classification", model="unitary/toxic-bert")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise

        # Define comprehensive keyword sets for different evaluation aspects
        self.educational_keywords = [
            "history", "historical", "fact", "facts", "evidence", "research",
            "study", "studies", "learn", "understand", "education", "truth"
        ]

        self.empathy_keywords = [
            "understand", "respect", "perspective", "feelings", "experience",
            "consider", "appreciate", "recognize", "acknowledge"
        ]

        self.antisemitism_countering_keywords = [
            "stereotype", "myth", "misconception", "prejudice", "discrimination",
            "jewish", "judaism", "holocaust", "antisemitism", "hate"
        ]

        # Quality indicators for thought process
        self.thought_quality_indicators = [
            "analyze", "consider", "evaluate", "assess", "determine",
            "relevant", "appropriate", "factual", "educational", "respectful"
        ]

    @staticmethod
    def cosine_similarity_numpy(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity with improved error handling."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    @staticmethod
    def validate_url(url: str, timeout: int = 5) -> Dict[str, any]:
        """Check if URL is accessible and valid."""
        if not url or not url.strip():
            return {"valid": False, "status_code": None, "error": "Empty URL"}

        try:
            # Basic URL format validation
            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.netloc]):
                return {"valid": False, "status_code": None, "error": "Invalid URL format"}

            # Check accessibility (with timeout)
            response = requests.head(url, timeout=timeout, allow_redirects=True)
            return {
                "valid": response.status_code < 400,
                "status_code": response.status_code,
                "error": None if response.status_code < 400 else f"HTTP {response.status_code}"
            }
        except requests.RequestException as e:
            return {"valid": False, "status_code": None, "error": str(e)}
        except Exception as e:
            return {"valid": False, "status_code": None, "error": f"Unexpected error: {str(e)}"}

    def evaluate_educational_quality(self, response: str, paragraph: str) -> Dict[str, float]:
        """Evaluate how educational and informative the response is."""
        try:
            combined_text = f"{response} {paragraph}".lower()

            educational_score = sum(1 for kw in self.educational_keywords if kw in combined_text)
            educational_score = min(educational_score / len(self.educational_keywords), 1.0)

            # Check for specific educational patterns
            has_citations = bool(re.search(r'\[[^]]+]|\([^)]+\)|according to|research shows', combined_text))
            has_statistics = bool(re.search(r'\d+%|\d+\s*(percent|million|billion|thousand)', combined_text))
            provides_context = len(combined_text.split()) > 50  # Reasonable length for context

            return {
                "educational_keyword_score": educational_score,
                "has_citations": has_citations,
                "has_statistics": has_statistics,
                "provides_context": provides_context,
                "overall_educational_score": np.mean([
                    educational_score,
                    float(has_citations),
                    float(has_statistics),
                    float(provides_context)
                ])
            }
        except Exception as e:
            raise Exception(f"Error evaluate_educational_quality: {e}")

    def evaluate_empathy_and_tone(self, response: str) -> Dict[str, any]:
        try:
            """Evaluate empathetic language and appropriate tone."""
            response_lower = response.lower()

            empathy_score = sum(1 for kw in self.empathy_keywords if kw in response_lower)
            empathy_score = min(empathy_score / len(self.empathy_keywords), 1.0)

            # Check for confrontational language
            confrontational_patterns = [
                r'\byou\'re wrong\b', r'\bthat\'s false\b', r'\bstupid\b',
                r'\bidiot\b', r'\bfool\b', r'\bignor', r'\bhateful\b'
            ]
            is_confrontational = any(re.search(pattern, response_lower) for pattern in confrontational_patterns)

            # Check for respectful language patterns
            respectful_patterns = [
                r'\bi understand\b', r'\blets?\s+consider\b', r'\bmight\s+be\s+helpful\b',
                r'\bperhaps\b', r'\bmay\s+want\s+to\b', r'\bconsider\b'
            ]
            is_respectful = any(re.search(pattern, response_lower) for pattern in respectful_patterns)

            return {
                "empathy_keyword_score": empathy_score,
                "is_confrontational": is_confrontational,
                "is_respectful": is_respectful,
                "tone_appropriateness": float(is_respectful and not is_confrontational)
            }
        except Exception as e:
            raise Exception(f"Error evaluate_empathy_and_tone: {e}")

    def evaluate_antisemitism_addressing(self, comment: str, response: str, label: str) -> Dict[str, any]:
        try:
            """Evaluate how well the response addresses the specific type of antisemitism."""
            response_lower = response.lower()
            comment_lower = comment.lower()

            # Check if response addresses antisemitic content
            addresses_antisemitism = sum(1 for kw in self.antisemitism_countering_keywords if kw in response_lower)
            addresses_antisemitism = min(addresses_antisemitism / len(self.antisemitism_countering_keywords), 1.0)

            # Check if response is contextually relevant to the comment
            comment_words = set(comment_lower.split())
            response_words = set(response_lower.split())
            word_overlap = len(comment_words.intersection(response_words)) / len(comment_words.union(response_words))

            # Label-specific evaluation
            label_relevance = self._evaluate_label_specific_response(label.lower(), response_lower)

            return {
                "addresses_antisemitism_score": addresses_antisemitism,
                "contextual_word_overlap": word_overlap,
                "label_specific_relevance": label_relevance,
                "overall_addressing_score": np.mean([addresses_antisemitism, word_overlap, label_relevance])
            }
        except Exception as e:
            raise Exception(f"Error evaluate_antisemitism_addressing: {e}")

    @staticmethod
    def _evaluate_label_specific_response(label: str, response: str) -> float:
        """Evaluate response relevance to specific antisemitism category."""
        try:
            label_keywords = {
                "conspiracy": ["conspiracy", "theory", "plot", "control", "manipulation"],
                "stereotype": ["stereotype", "generalization", "assumption", "prejudice"],
                "holocaust": ["holocaust", "genocide", "history", "historical", "evidence"],
                "economic": ["money", "wealth", "banking", "financial", "economic"],
                "religious": ["religion", "belief", "practice", "tradition", "jewish"]
            }

            relevant_keywords = []
            for category, keywords in label_keywords.items():
                if category in label:
                    relevant_keywords.extend(keywords)

            if not relevant_keywords:
                return 0.5  # Neutral score if no specific category matches

            matches = sum(1 for kw in relevant_keywords if kw in response)
            return min(matches / len(relevant_keywords), 1.0)
        except Exception as e:
            raise Exception(f"Error evaluate_label_specific_response: {e}")

    def evaluate_thought_quality(self, thought: str) -> Dict[str, any]:
        """Enhanced evaluation of the agent's reasoning process."""
        try:
            if not thought.strip():
                return {
                    "thought_present": False,
                    "thought_quality_score": 0.0,
                    "shows_reasoning": False,
                    "mentions_sources": False
                }

            thought_lower = thought.lower()

            # Check for quality indicators
            quality_score = sum(1 for indicator in self.thought_quality_indicators if indicator in thought_lower)
            quality_score = min(quality_score / len(self.thought_quality_indicators), 1.0)

            # Check for reasoning patterns
            reasoning_patterns = [
                r'because\b', r'therefore\b', r'since\b', r'given that\b',
                r'considering\b', r'based on\b', r'due to\b'
            ]
            shows_reasoning = any(re.search(pattern, thought_lower) for pattern in reasoning_patterns)

            # Check if mentions source evaluation
            mentions_sources = any(term in thought_lower for term in ["source", "reference", "citation", "evidence"])

            return {
                "thought_present": True,
                "thought_quality_score": quality_score,
                "shows_reasoning": shows_reasoning,
                "mentions_sources": mentions_sources
            }
        except Exception as e:
            raise Exception(f"Error evaluate_thought_quality: {e}")

    def evaluate_comprehensive(self, comment: str, label: str, agent_output: Dict,
                               expected_keywords: Optional[List[str]] = None) -> Dict[str, any]:
        """
        Comprehensive evaluation of the agent's response.

        Args:
            comment: Original antisemitic comment
            label: Classification label
            agent_output: Agent's response dict
            expected_keywords: Optional keywords to check for

        Returns:
            Dict with comprehensive evaluation metrics
        """
        response = agent_output.get("response", "")
        thought = agent_output.get("thought", "")
        paragraph = agent_output.get("paragraph", "")
        source = agent_output.get("source", "")
        url = agent_output.get("url", "")

        metrics = {}

        # Original metrics (improved)
        metrics.update(self.evaluate_thought_quality(thought))

        # Reference quality
        metrics['paragraph_present'] = bool(paragraph.strip())
        metrics['source_present'] = bool(source.strip())
        metrics['url_present'] = bool(url.strip())

        # URL validation
        if url:
            url_validation = self.validate_url(url)
            metrics['url_valid'] = url_validation["valid"]
            metrics['url_status_code'] = url_validation["status_code"]
            metrics['url_error'] = url_validation["error"]
        else:
            metrics.update({"url_valid": False, "url_status_code": None, "url_error": "No URL provided"})

        # Enhanced URL analysis
        found_urls = re.findall(r'https?://\S+', response)
        metrics['num_urls_found'] = len(found_urls)
        metrics['includes_agent_url'] = url in response if url else False

        # Semantic analysis
        if response and comment:
            comment_emb = self.semantic_model.encode(comment, convert_to_numpy=True)
            response_emb = self.semantic_model.encode(response, convert_to_numpy=True)
            metrics['semantic_relevance'] = self.cosine_similarity_numpy(comment_emb, response_emb)
        else:
            metrics['semantic_relevance'] = 0.0

        # Educational quality
        educational_metrics = self.evaluate_educational_quality(response, paragraph)
        metrics.update({f"educational_{k}": v for k, v in educational_metrics.items()})

        # Empathy and tone
        empathy_metrics = self.evaluate_empathy_and_tone(response)
        metrics.update({f"empathy_{k}": v for k, v in empathy_metrics.items()})

        # Antisemitism addressing
        antisemitism_metrics = self.evaluate_antisemitism_addressing(comment, response, label)
        metrics.update({f"antisemitism_{k}": v for k, v in antisemitism_metrics.items()})

        # Keyword coverage
        if expected_keywords:
            found_keywords = [kw for kw in expected_keywords if kw.lower() in response.lower()]
            metrics['keyword_coverage'] = len(found_keywords) / len(expected_keywords)
            metrics['found_keywords'] = found_keywords
        else:
            metrics['keyword_coverage'] = None
            metrics['found_keywords'] = []

        # Sentiment and toxicity (with error handling)
        try:
            sentiment = self.sentiment_pipeline(response[:512])[0]
            metrics['sentiment_label'] = sentiment['label']
            metrics['sentiment_score'] = sentiment['score']
        except Exception as e:
            metrics.update({"sentiment_label": "ERROR", "sentiment_score": 0.0, "sentiment_error": str(e)})

        try:
            toxicity = self.toxicity_pipeline(response[:512])[0]
            metrics['toxicity_label'] = toxicity['label']
            metrics['toxicity_score'] = toxicity['score']
        except Exception as e:
            metrics.update({"toxicity_label": "ERROR", "toxicity_score": 0.0, "toxicity_error": str(e)})

        # Length and richness
        metrics['response_length_tokens'] = len(response.split())
        metrics['response_length_chars'] = len(response)
        metrics['paragraph_length_tokens'] = len(paragraph.split())
        metrics['avg_sentence_length'] = np.mean([len(sent.split()) for sent in response.split('.') if sent.strip()])

        # Overall quality score
        quality_components = [
            metrics.get('thought_quality_score', 0),
            metrics.get('educational_overall_educational_score', 0),
            metrics.get('empathy_tone_appropriateness', 0),
            metrics.get('antisemitism_overall_addressing_score', 0),
            metrics.get('semantic_relevance', 0),
            1.0 if metrics.get('url_valid', False) else 0.0
        ]

        metrics['overall_quality_score'] = np.mean([score for score in quality_components if score is not None])

        return metrics

    @staticmethod
    def evaluate_agent_response(comment, label, agent_output, expected_keywords=None):
        """Convenience function for backward compatibility."""
        evaluator = ResponseEvaluator()
        return evaluator.evaluate_comprehensive(comment, label, agent_output, expected_keywords)
