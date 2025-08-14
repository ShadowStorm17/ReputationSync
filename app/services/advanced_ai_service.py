"""
Advanced AI and NLP service.
Provides enhanced analysis capabilities for reputation management.
"""

from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from transformers import pipeline

from app.core.error_handling import (
    ErrorCategory,
    ErrorSeverity,
    ReputationError,
)
from app.core.metrics import track_performance


class AdvancedAIService:
    """Service for advanced AI and NLP features."""

    def __init__(self):
        """Initialize advanced AI service."""
        # Initialize multilingual sentiment analyzer
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
        )

        # Initialize emotion detector
        self.emotion_analyzer = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
        )

        # Initialize NER model
        self.ner_model = pipeline(
            "ner", model="xlm-roberta-large-finetuned-conll03-english"
        )

        # Initialize sentence transformer for topic clustering
        self.sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize fake review detector
        self.fake_review_detector = pipeline(
            "text-classification", model="microsoft/deberta-v3-base"
        )

        # Initialize language detector
        self.language_detector = pipeline(
            "text-classification",
            model="papluca/xlm-roberta-base-language-detection",
        )

    @track_performance
    async def analyze_multilingual_sentiment(
        self, text: str, language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze sentiment in multiple languages."""
        try:
            # Detect language if not provided
            if not language:
                lang_result = self.language_detector(text)[0]
                language = lang_result["label"]

            # Analyze sentiment
            sentiment = self.sentiment_analyzer(text)[0]

            return {
                "language": language,
                "sentiment": sentiment["label"],
                "score": float(sentiment["score"]),
                "confidence": float(sentiment["score"]),
            }
        except Exception as e:
            raise ReputationError(
                message=f"Error in multilingual sentiment analysis: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS,
            )

    @track_performance
    async def detect_emotions(self, text: str) -> Dict[str, float]:
        """Detect emotions in text."""
        try:
            emotions = self.emotion_analyzer(text)[0]
            return {
                "emotion": emotions["label"],
                "score": float(emotions["score"]),
            }
        except Exception as e:
            raise ReputationError(
                message=f"Error in emotion detection: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS,
            )

    @track_performance
    async def cluster_topics(
        self, texts: List[str], min_samples: int = 2, eps: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Cluster texts by topic."""
        try:
            # Generate embeddings
            embeddings = self.sentence_transformer.encode(texts)

            # Perform clustering
            clustering = DBSCAN(
                min_samples=min_samples, eps=eps, metric="cosine"
            ).fit(embeddings)

            # Group texts by cluster
            clusters = {}
            for idx, label in enumerate(clustering.labels_):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(texts[idx])

            # Format results
            results = []
            for label, texts in clusters.items():
                results.append(
                    {
                        "cluster_id": int(label),
                        "texts": texts,
                        "size": len(texts),
                    }
                )

            return results
        except Exception as e:
            raise ReputationError(
                message=f"Error in topic clustering: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS,
            )

    @track_performance
    async def detect_fake_reviews(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Detect fake reviews using pattern recognition and NLP."""
        try:
            # Analyze text
            fake_score = self.fake_review_detector(text)[0]

            # Additional pattern analysis
            patterns = {
                "excessive_punctuation": sum(1 for c in text if c in "!?") > 3,
                "all_caps": text.isupper(),
                "repetitive_words": len(set(text.split()))
                < len(text.split()) * 0.5,
                "short_length": len(text.split()) < 5,
            }

            # Calculate pattern score
            pattern_score = sum(patterns.values()) / len(patterns)

            # Combine scores
            final_score = (float(fake_score["score"]) + pattern_score) / 2

            return {
                "is_fake": final_score > 0.7,
                "fake_score": final_score,
                "patterns_detected": patterns,
                "confidence": float(fake_score["score"]),
            }
        except Exception as e:
            raise ReputationError(
                message=f"Error in fake review detection: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS,
            )

    @track_performance
    async def extract_entities(
        self, text: str, entity_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        try:
            # Extract entities
            entities = self.ner_model(text)

            # Filter by entity types if specified
            if entity_types:
                entities = [
                    entity
                    for entity in entities
                    if entity["entity"] in entity_types
                ]

            # Group consecutive entities
            grouped_entities = []
            current_entity = None

            for entity in entities:
                if (
                    current_entity
                    and entity["entity"] == current_entity["entity"]
                ):
                    current_entity["word"] += " " + entity["word"]
                    current_entity["end"] = entity["end"]
                else:
                    if current_entity:
                        grouped_entities.append(current_entity)
                    current_entity = entity.copy()

            if current_entity:
                grouped_entities.append(current_entity)

            return grouped_entities
        except Exception as e:
            raise ReputationError(
                message=f"Error in entity extraction: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS,
            )
