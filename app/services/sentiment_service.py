import logging
from datetime import datetime, timezone
from typing import Dict, List, Union

import numpy as np
import torch
from prometheus_client import Counter
from tenacity import retry, stop_after_attempt, wait_exponential
from textblob import TextBlob
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import asyncio
import hashlib

from app.core.cache import cache
from app.core.config import get_settings
from app.core.metrics import SENTIMENT_ANALYSIS_LATENCY, SENTIMENT_ERRORS

logger = logging.getLogger(__name__)
settings = get_settings()


class SentimentService:
    """Advanced sentiment analysis service with emotion detection and contextual understanding."""

    def __init__(self):
        """Initialize advanced sentiment analysis models and tools."""
        # Initialize transformer models
        self.tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.emotion_model = (
            AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased-finetuned-sst-2-english"
            )
        )
        self.context_model = AutoModelForTokenClassification.from_pretrained(
            "dslim/bert-base-NER"
        )

        # Initialize additional sentiment analyzers
        self.vader = SentimentIntensityAnalyzer()

        # Initialize sentiment pipeline
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            tokenizer=self.tokenizer,
        )

        # Performance monitoring
        self.analysis_counter = None  # No longer needed, use shared metrics if required
        self.error_counter = SENTIMENT_ERRORS

        # Cache configuration
        self.cache_ttl = settings.cache["sentiment_analysis"] if hasattr(settings.cache, '__getitem__') and "sentiment_analysis" in settings.cache else 3600

        # Analysis thresholds
        self.thresholds = {
            "emotion_confidence": 0.75,
            "context_relevance": 0.6,
            "sentiment_agreement": 0.7,
        }

        # Emotion categories
        self.emotion_categories = ["positive", "negative"]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def analyze_text(self, text: str) -> Dict:
        """Perform comprehensive sentiment analysis on text."""
        try:
            start_time = datetime.now(timezone.utc)
            self.analysis_counter.inc()

            # Check cache
            cache_key = "sentiment:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
            cached_result = await cache.get(cache_key)
            if cached_result:
                return cached_result

            # Perform multi-model analysis
            results = await asyncio.gather(
                self._analyze_emotions(text),
                self._analyze_context(text),
                self._analyze_vader(text),
            )

            # Combine and reconcile results
            analysis = self._reconcile_analyses(
                emotion_analysis=results[0],
                context_analysis=results[1],
                vader_analysis=results[2],
            )

            # Add metadata
            analysis["metadata"] = {
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
                "processing_time": (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds(),
                "models_used": ["distilbert", "bert", "vader"],
                "confidence": self._calculate_confidence(results),
            }

            # Cache results
            await cache.set(cache_key, analysis, ttl=self.cache_ttl)

            # Record latency
            SENTIMENT_ANALYSIS_LATENCY.observe(
                (datetime.now(timezone.utc) - start_time).total_seconds()
            )

            return analysis

        except Exception as e:
            self.error_counter.inc()
            logger.error(f"Error in sentiment analysis: {str(e)}")
            raise

    async def _analyze_emotions(self, text: str) -> Dict:
        """Analyze emotions using transformer model."""
        try:
            # Tokenize and encode text
            encoded = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            # Get emotion predictions
            with torch.no_grad():
                outputs = self.emotion_model(**encoded)
                predictions = torch.softmax(outputs.logits, dim=1)

            # Process results
            emotions = {}
            for idx, score in enumerate(predictions[0]):
                emotions[self.emotion_categories[idx]] = score.item()

            return {
                "emotions": emotions,
                "dominant_emotion": max(emotions.items(), key=lambda x: x[1])[
                    0
                ],
                "confidence": max(emotions.values()),
            }

        except Exception as e:
            logger.error(f"Error in emotion analysis: {str(e)}")
            return {}

    async def _analyze_context(self, text: str) -> Dict:
        """Analyze contextual elements affecting sentiment."""
        try:
            # Tokenize and encode text
            encoded = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            # Get context predictions
            with torch.no_grad():
                outputs = self.context_model(**encoded)
                predictions = torch.softmax(outputs.logits, dim=2)

            # Extract contextual elements
            context_elements = self._extract_context_elements(
                text, predictions[0]
            )

            return {
                "context": context_elements,
                "modifiers": self._identify_sentiment_modifiers(text),
                "targets": self._identify_sentiment_targets(text),
            }

        except Exception as e:
            logger.error(f"Error in context analysis: {str(e)}")
            return {}

    def _analyze_vader(self, text: str) -> Dict:
        """Analyze sentiment using VADER."""
        try:
            scores = self.vader.polarity_scores(text)

            return {
                "compound": scores["compound"],
                "positive": scores["pos"],
                "negative": scores["neg"],
                "neutral": scores["neu"],
                "intensity": abs(scores["compound"]),
            }

        except Exception as e:
            logger.error(f"Error in VADER analysis: {str(e)}")
            return {}

    def _reconcile_analyses(
        self,
        emotion_analysis: Dict,
        context_analysis: Dict,
        vader_analysis: Dict,
    ) -> Dict:
        """Reconcile results from different analysis methods."""
        try:
            # Determine overall sentiment
            sentiment = self._determine_overall_sentiment(
                emotion_analysis, vader_analysis
            )

            # Combine all analyses
            return {
                "sentiment": sentiment,
                "emotions": emotion_analysis.get("emotions", {}),
                "context": {
                    "elements": context_analysis.get("context", {}),
                    "modifiers": context_analysis.get("modifiers", []),
                    "targets": context_analysis.get("targets", []),
                },
                "intensity": {
                    "score": vader_analysis.get("intensity", 0),
                    "label": self._get_intensity_label(
                        vader_analysis.get("intensity", 0)
                    ),
                },
                "agreement": self._calculate_model_agreement(
                    emotion_analysis, vader_analysis
                ),
            }

        except Exception as e:
            logger.error(f"Error reconciling analyses: {str(e)}")
            return {}

    def _determine_overall_sentiment(
        self, emotion_analysis: Dict, vader_analysis: Dict
    ) -> Dict:
        """Determine overall sentiment from multiple analyses."""
        try:
            # Get emotion-based sentiment
            emotion = emotion_analysis.get("dominant_emotion", "neutral")
            emotion_sentiment = self._emotion_to_sentiment(emotion)

            # Get VADER sentiment
            vader_compound = vader_analysis.get("compound", 0)
            vader_sentiment = self._compound_to_sentiment(vader_compound)

            # Calculate confidence-weighted sentiment
            emotion_confidence = emotion_analysis.get("confidence", 0)
            vader_confidence = abs(vader_compound)

            if emotion_confidence > self.thresholds["emotion_confidence"]:
                primary_sentiment = emotion_sentiment
                confidence = emotion_confidence
            elif vader_confidence > 0.5:
                primary_sentiment = vader_sentiment
                confidence = vader_confidence
            else:
                primary_sentiment = "neutral"
                confidence = max(emotion_confidence, vader_confidence)

            return {
                "label": primary_sentiment,
                "confidence": confidence,
                "sources": {
                    "emotion": emotion_sentiment,
                    "vader": vader_sentiment,
                },
            }

        except Exception as e:
            logger.error(f"Error determining overall sentiment: {str(e)}")
            return {"label": "neutral", "confidence": 0}

    def _emotion_to_sentiment(self, emotion: str) -> str:
        """Convert emotion to sentiment label."""
        positive_emotions = {"joy", "surprise"}
        negative_emotions = {"sadness", "anger", "fear", "disgust"}

        if emotion in positive_emotions:
            return "positive"
        elif emotion in negative_emotions:
            return "negative"
        return "neutral"

    def _compound_to_sentiment(self, compound: float) -> str:
        """Convert VADER compound score to sentiment label."""
        if compound > 0.05:
            return "positive"
        elif compound < -0.05:
            return "negative"
        return "neutral"

    def _calculate_model_agreement(
        self, emotion_analysis: Dict, vader_analysis: Dict
    ) -> float:
        """Calculate agreement between different analysis methods."""
        try:
            emotion_sentiment = self._emotion_to_sentiment(
                emotion_analysis.get("dominant_emotion", "neutral")
            )
            vader_sentiment = self._compound_to_sentiment(
                vader_analysis.get("compound", 0)
            )

            return 1.0 if emotion_sentiment == vader_sentiment else 0.0

        except Exception:
            return 0.0

    def _get_intensity_label(self, intensity: float) -> str:
        """Convert intensity score to label."""
        if intensity > 0.75:
            return "very_strong"
        elif intensity > 0.5:
            return "strong"
        elif intensity > 0.25:
            return "moderate"
        return "weak"

    def _extract_context_elements(
        self, text: str, predictions: torch.Tensor
    ) -> Dict:
        """Extract contextual elements affecting sentiment."""
        try:
            # Implementation would identify contextual elements
            return {}
        except Exception:
            return {}

    def _identify_sentiment_modifiers(self, text: str) -> List[Dict]:
        """Identify words/phrases that modify sentiment."""
        try:
            # Implementation would identify sentiment modifiers
            return []
        except Exception:
            return []

    def _identify_sentiment_targets(self, text: str) -> List[Dict]:
        """Identify targets of sentiment expressions."""
        try:
            # Implementation would identify sentiment targets
            return []
        except Exception:
            return []

    def _calculate_confidence(self, results: List[Dict]) -> float:
        """Calculate overall confidence in analysis results."""
        try:
            confidences = [
                results[0].get("confidence", 0),  # Emotion confidence
                abs(results[2].get("compound", 0)),  # VADER confidence
            ]
            return np.mean(confidences)
        except Exception:
            return 0.0

    async def analyze_bulk(self, texts: List[str]) -> List[Dict]:
        """Analyze sentiment of multiple texts."""
        results = []
        for text in texts:
            result = await self.analyze_text(text)
            results.append(result)
        return results

    async def get_sentiment_trends(
        self, texts: List[str], timeframe: str = "1d"
    ) -> Dict:
        """Get sentiment trends over time."""
        try:
            # Analyze all texts
            results = await self.analyze_bulk(texts)

            # Calculate trends
            sentiments = {"positive": 0, "neutral": 0, "negative": 0}
            total_score = 0.0

            for result in results:
                sentiment = result.get("sentiment", "neutral")
                sentiments[sentiment] += 1
                total_score += result.get("score", 0.0)

            total = len(results)
            avg_score = total_score / total if total > 0 else 0.0

            return {
                "distribution": {k: v / total for k, v in sentiments.items()},
                "average_score": avg_score,
                "total_analyzed": total,
                "timeframe": timeframe,
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Error analyzing sentiment trends: {str(e)}")
            return {
                "error": str(e),
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
            }

    async def analyze_trend(
        self,
        texts: List[Dict[str, Union[str, datetime]]],
        window_size: int = 24,
    ) -> Dict:
        """Analyze sentiment trend over time."""
        try:
            # Group texts by time windows
            now = datetime.now(timezone.utc)
            windows = {}

            for text_data in texts:
                text = text_data["text"]
                timestamp = text_data.get("timestamp", now)

                window_key = timestamp.replace(
                    minute=0, second=0, microsecond=0
                ).isoformat()

                if window_key not in windows:
                    windows[window_key] = []
                windows[window_key].append(text)

            # Analyze sentiment for each window
            trend_data = []
            for window_key, window_texts in windows.items():
                sentiment = await self.analyze_text(
                    window_texts, cache_key=f"sentiment:window:{window_key}"
                )

                trend_data.append(
                    {
                        "timestamp": window_key,
                        "average": sentiment["average"],
                        "count": len(window_texts),
                        "distribution": sentiment["distribution"],
                    }
                )

            # Calculate trend metrics
            if trend_data:
                latest = trend_data[-1]["average"]["polarity"]
                earliest = trend_data[0]["average"]["polarity"]
                change = latest - earliest
            else:
                change = 0

            return {
                "windows": trend_data,
                "metrics": {
                    "change": change,
                    "volatility": self._calculate_volatility(trend_data),
                },
            }

        except Exception as e:
            logger.error(f"Error analyzing sentiment trend: {str(e)}")
            raise ValueError(f"Error analyzing sentiment trend: {str(e)}")

    def _classify_sentiment(self, polarity: float) -> str:
        """Classify sentiment based on polarity score."""
        if polarity > 0.1:
            return "positive"
        elif polarity < -0.1:
            return "negative"
        else:
            return "neutral"

    def _get_sentiment_distribution(
        self, sentiments: List[Dict]
    ) -> Dict[str, int]:
        """Calculate distribution of sentiment classifications."""
        distribution = {"positive": 0, "neutral": 0, "negative": 0}

        for sentiment in sentiments:
            classification = sentiment["classification"]
            distribution[classification] += 1

        return distribution

    def _calculate_volatility(self, trend_data: List[Dict]) -> float:
        """Calculate sentiment volatility from trend data."""
        if not trend_data or len(trend_data) < 2:
            return 0.0

        polarities = [w["average"]["polarity"] for w in trend_data]
        changes = [
            abs(polarities[i] - polarities[i - 1])
            for i in range(1, len(polarities))
        ]

        return sum(changes) / len(changes) if changes else 0.0

    async def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text using TextBlob."""
        try:
            # Check cache first
            cache_key = "sentiment:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
            cached = await cache.get(cache_key)
            if cached:
                return cached

            # Perform sentiment analysis
            analysis = TextBlob(text)
            sentiment = {
                "polarity": float(analysis.sentiment.polarity),
                "subjectivity": float(analysis.sentiment.subjectivity),
                "sentiment": self._get_sentiment_label(
                    analysis.sentiment.polarity
                ),
            }

            # Cache the result
            await cache.set(cache_key, sentiment, self.cache_ttl)
            return sentiment

        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {
                "polarity": 0.0,
                "subjectivity": 0.0,
                "sentiment": "neutral",
            }

    async def analyze_bulk_sentiment(
        self, texts: List[str]
    ) -> List[Dict[str, float]]:
        """Analyze sentiment for multiple texts in parallel."""
        try:
            results = []
            for text in texts:
                sentiment = await self.analyze_sentiment(text)
                results.append(sentiment)
            return results
        except Exception as e:
            logger.error(f"Error in bulk sentiment analysis: {str(e)}")
            return [{"error": str(e)}] * len(texts)

    async def get_reputation_score(
        self, platform_data: Dict, timeframe: str = "7d"
    ) -> Dict:
        """Calculate overall reputation score based on multiple factors."""
        try:
            scores = {}

            # Analyze comments sentiment
            if "comments" in platform_data:
                comment_sentiments = await self.analyze_bulk_sentiment(
                    [c["text"] for c in platform_data["comments"]]
                )
                scores["comment_score"] = self._calculate_avg_sentiment(
                    comment_sentiments
                )

            # Analyze mentions sentiment
            if "mentions" in platform_data:
                mention_sentiments = await self.analyze_bulk_sentiment(
                    [m["text"] for m in platform_data["mentions"]]
                )
                scores["mention_score"] = self._calculate_avg_sentiment(
                    mention_sentiments
                )

            # Calculate engagement score
            if "engagement" in platform_data:
                scores["engagement_score"] = self._calculate_engagement_score(
                    platform_data["engagement"]
                )

            # Calculate growth score
            if "growth" in platform_data:
                scores["growth_score"] = self._calculate_growth_score(
                    platform_data["growth"]
                )

            # Calculate overall score
            overall_score = self._calculate_overall_score(scores)

            return {
                "overall_score": overall_score,
                "component_scores": scores,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "timeframe": timeframe,
            }

        except Exception as e:
            logger.error(f"Error calculating reputation score: {str(e)}")
            raise ValueError(f"Error calculating reputation score: {str(e)}")

    def _get_sentiment_label(self, polarity: float) -> str:
        """Convert sentiment polarity to label."""
        if polarity > 0.3:
            return "positive"
        elif polarity < -0.3:
            return "negative"
        return "neutral"

    def _calculate_avg_sentiment(
        self, sentiments: List[Dict[str, float]]
    ) -> float:
        """Calculate average sentiment score."""
        if not sentiments:
            return 0.0
        return np.mean([s["polarity"] for s in sentiments])

    def _calculate_engagement_score(self, engagement_data: Dict) -> float:
        """Calculate engagement score based on various metrics."""
        try:
            metrics = {
                "likes": 1.0,
                "comments": 2.0,
                "shares": 3.0,
                "saves": 2.5,
            }

            total_score = 0
            total_weight = 0

            for metric, weight in metrics.items():
                if metric in engagement_data:
                    total_score += engagement_data[metric] * weight
                    total_weight += weight

            return total_score / total_weight if total_weight > 0 else 0

        except Exception:
            return 0.0

    def _calculate_growth_score(self, growth_data: Dict) -> float:
        """Calculate growth score based on follower/engagement growth."""
        try:
            metrics = {"follower_growth": 0.6, "engagement_growth": 0.4}

            total_score = 0
            total_weight = 0

            for metric, weight in metrics.items():
                if metric in growth_data:
                    total_score += growth_data[metric] * weight
                    total_weight += weight

            return total_score / total_weight if total_weight > 0 else 0

        except Exception:
            return 0.0

    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calculate overall reputation score."""
        try:
            total_score = 0
            total_weight = 0

            for component, score in scores.items():
                weight = self.sentiment_weights.get(
                    component.replace("_score", ""), 0.1
                )
                total_score += score * weight
                total_weight += weight

            return round(
                (total_score / total_weight if total_weight > 0 else 0) * 100,
                2,
            )

        except Exception:
            return 0.0
