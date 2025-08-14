"""
Content analysis service.
Provides comprehensive analysis of various content types for reputation management.
"""

from typing import Any, Dict, List

import cv2
import pytesseract
from PIL import Image
from transformers import pipeline

from app.core.error_handling import (
    ErrorCategory,
    ErrorSeverity,
    ReputationError,
)
from app.core.metrics import track_performance
from app.models.premium_features import (
    ContentAnalysis,
    ContentRecommendation,
    ContentType,
)


class ContentAnalysisService:
    """Service for analyzing various types of content."""

    def __init__(self):
        """Initialize content analysis service."""
        # Initialize ML models
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.toxicity_analyzer = pipeline(
            "text-classification", model="unitary/toxic-bert"
        )
        self.spam_detector = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.entity_recognizer = pipeline("ner")
        self.topic_classifier = pipeline("zero-shot-classification")
        self.image_analyzer = pipeline("image-classification")
        self.object_detector = pipeline("object-detection")
        # self.face_analyzer = pipeline("face-detection")
        # self.text_recognizer = pipeline("text-recognition")
        self.language_detector = pipeline("language-detection")

    @track_performance
    async def analyze_content(
        self,
        content_type: ContentType,
        content_data: Dict[str, Any],
        platform: str,
    ) -> ContentAnalysis:
        """Analyze content comprehensively."""
        try:
            # Initialize analysis
            analysis = ContentAnalysis(
                content_id=content_data.get("id"),
                content_type=content_type,
                platform=platform,
                analysis_version="1.0",
            )

            # Analyze text content
            if content_data.get("text"):
                await self._analyze_text(content_data["text"], analysis)

            # Analyze image content
            if content_data.get("image_url"):
                await self._analyze_image(content_data["image_url"], analysis)

            # Analyze video content
            if content_data.get("video_url"):
                await self._analyze_video(content_data["video_url"], analysis)

            # Analyze thumbnail
            if content_data.get("thumbnail_url"):
                await self._analyze_thumbnail(
                    content_data["thumbnail_url"], analysis
                )

            # Calculate overall scores
            analysis.reputation_impact_score = (
                self._calculate_reputation_impact(analysis)
            )
            analysis.confidence_score = self._calculate_confidence_score(
                analysis
            )

            return analysis

        except Exception as e:
            raise ReputationError(
                message=f"Error analyzing content: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS,
            )

    async def _analyze_text(
        self, text: str, analysis: ContentAnalysis
    ) -> None:
        """Analyze text content."""
        # Sentiment analysis
        sentiment = self.sentiment_analyzer(text)[0]
        analysis.sentiment_score = (
            float(sentiment["score"])
            if sentiment["label"] == "POSITIVE"
            else -float(sentiment["score"])
        )

        # Toxicity analysis
        toxicity = self.toxicity_analyzer(text)[0]
        analysis.toxicity_score = float(toxicity["score"])

        # Spam detection
        spam = self.spam_detector(text)[0]
        analysis.spam_score = float(spam["score"])

        # Entity recognition
        entities = self.entity_recognizer(text)
        analysis.entities = entities

        # Topic classification
        topics = self.topic_classifier(
            text,
            candidate_labels=[
                "business",
                "technology",
                "entertainment",
                "sports",
                "politics",
            ],
        )
        analysis.topics = [
            label
            for label, score in zip(topics["labels"], topics["scores"])
            if score > 0.5
        ]

        # Language detection
        language = self.language_detector(text)[0]
        analysis.language = language["label"]
        analysis.language_confidence = float(language["score"])

    async def _analyze_image(
        self, image_url: str, analysis: ContentAnalysis
    ) -> None:
        """Analyze image content."""
        # Download and process image
        image = Image.open(image_url)

        # Image classification
        classification = self.image_analyzer(image)[0]
        analysis.categories.append(classification["label"])

        # Object detection
        objects = self.object_detector(image)
        analysis.entities.extend(
            [{"type": "object", "label": obj["label"]} for obj in objects]
        )

        # Face detection
        # faces = self.face_analyzer(image)
        # analysis.metadata["face_count"] = len(faces)

        # Text recognition
        text = pytesseract.image_to_string(image)
        if text.strip():
            analysis.text = text
            await self._analyze_text(text, analysis)

    async def _analyze_video(
        self, video_url: str, analysis: ContentAnalysis
    ) -> None:
        """Analyze video content."""
        # Download video
        cap = cv2.VideoCapture(video_url)

        # Extract frames
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        # Analyze key frames
        for frame in frames[::30]:  # Analyze every 30th frame
            # Convert frame to PIL Image
            frame_image = Image.fromarray(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            # Analyze frame
            await self._analyze_image(frame_image, analysis)

        cap.release()

    async def _analyze_thumbnail(
        self, thumbnail_url: str, analysis: ContentAnalysis
    ) -> None:
        """Analyze video thumbnail."""
        await self._analyze_image(thumbnail_url, analysis)

    def _calculate_reputation_impact(self, analysis: ContentAnalysis) -> float:
        """Calculate overall reputation impact score."""
        weights = {
            "sentiment": 0.3,
            "toxicity": -0.2,
            "spam": -0.1,
            "engagement": 0.2,
            "virality": 0.1,
            "controversy": -0.1,
        }

        impact = (
            analysis.sentiment_score * weights["sentiment"]
            + analysis.toxicity_score * weights["toxicity"]
            + analysis.spam_score * weights["spam"]
            + analysis.engagement_score * weights["engagement"]
            + analysis.virality_score * weights["virality"]
            + analysis.controversy_score * weights["controversy"]
        )

        return max(0, min(1, impact))

    def _calculate_confidence_score(self, analysis: ContentAnalysis) -> float:
        """Calculate confidence score for the analysis."""
        factors = [
            analysis.language_confidence,
            1 - analysis.toxicity_score,
            1 - analysis.spam_score,
        ]

        return sum(factors) / len(factors)

    @track_performance
    async def generate_recommendations(
        self, analysis: ContentAnalysis
    ) -> List[ContentRecommendation]:
        """Generate recommendations based on content analysis."""
        recommendations = []

        # Check for high toxicity
        if analysis.toxicity_score > 0.7:
            recommendations.append(
                ContentRecommendation(
                    content_id=analysis.content_id,
                    content_type=analysis.content_type,
                    platform=analysis.platform,
                    recommendation_type="moderation",
                    priority=1,
                    action="remove",
                    reason="High toxicity detected",
                    impact_score=0.9,
                    effort_score=0.1,
                    urgency_score=0.9,
                    status="pending",
                )
            )

        # Check for spam
        if analysis.spam_score > 0.8:
            recommendations.append(
                ContentRecommendation(
                    content_id=analysis.content_id,
                    content_type=analysis.content_type,
                    platform=analysis.platform,
                    recommendation_type="moderation",
                    priority=1,
                    action="remove",
                    reason="Spam detected",
                    impact_score=0.8,
                    effort_score=0.1,
                    urgency_score=0.8,
                    status="pending",
                )
            )

        # Check for negative sentiment
        if analysis.sentiment_score < -0.5:
            recommendations.append(
                ContentRecommendation(
                    content_id=analysis.content_id,
                    content_type=analysis.content_type,
                    platform=analysis.platform,
                    recommendation_type="engagement",
                    priority=2,
                    action="respond",
                    reason="Negative sentiment detected",
                    impact_score=0.7,
                    effort_score=0.5,
                    urgency_score=0.6,
                    status="pending",
                )
            )

        # Check for low engagement
        if analysis.engagement_score < 0.3:
            recommendations.append(
                ContentRecommendation(
                    content_id=analysis.content_id,
                    content_type=analysis.content_type,
                    platform=analysis.platform,
                    recommendation_type="optimization",
                    priority=3,
                    action="improve",
                    reason="Low engagement detected",
                    impact_score=0.6,
                    effort_score=0.7,
                    urgency_score=0.4,
                    status="pending",
                )
            )

        return recommendations
