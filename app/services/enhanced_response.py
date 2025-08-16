"""
Enhanced response service.
Provides advanced response generation with context awareness and personalization.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import torch
from prometheus_client import Counter
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)

from app.core.config import get_settings
from app.core.metrics import RESPONSE_GENERATION_LATENCY, RESPONSE_ERRORS
from app.services.sentiment_service import SentimentService

logger = logging.getLogger(__name__)
settings = get_settings()


class EnhancedResponse:
    """Enhanced response generation service with context awareness."""

    def __init__(self):
        """Initialize enhanced response models and tools."""
        # Initialize models
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
        self.response_model = AutoModelForCausalLM.from_pretrained(
            "gpt2-medium")
        self.context_model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-base-mnli")

        # Initialize sentiment service
        self.sentiment_service = SentimentService()

        # Initialize response pipeline
        self.response_pipeline = pipeline(
            "text-generation",
            model="gpt2-medium",
            tokenizer=self.tokenizer
        )

        # Cache configuration
        self.cache_ttl = settings.cache["response_generation"] if hasattr(settings.cache, '__getitem__') and "response_generation" in settings.cache else 3600

        # Response parameters
        self.max_length = 150
        self.temperature = 0.7
        self.top_p = 0.9
        self.top_k = 50

        # Context weights
        self.context_weights = {
            "sentiment": 0.3,
            "user_history": 0.2,
            "platform": 0.2,
            "urgency": 0.3
        }

        self.generation_counter = None  # Use shared metrics only
        self.error_counter = RESPONSE_ERRORS

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_response(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate enhanced response with context awareness."""
        try:
            start_time = datetime.now(timezone.utc)
            self.generation_counter.inc()

            # Analyze sentiment
            sentiment = await self.sentiment_service.analyze_sentiment(text)

            # Prepare context
            enhanced_context = await self._enhance_context(
                text,
                sentiment,
                context
            )

            # Generate response
            response = await self._generate_contextual_response(
                text,
                enhanced_context
            )

            # Validate response
            validated_response = await self._validate_response(
                response,
                text,
                enhanced_context
            )

            # Record latency
            RESPONSE_GENERATION_LATENCY.observe(
                (datetime.now(timezone.utc) - start_time).total_seconds()
            )

            return {
                "response": validated_response,
                "sentiment": sentiment,
                "context_used": enhanced_context,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            self.error_counter.inc()
            logger.error(f"Error generating enhanced response: {str(e)}")
            raise

    async def _enhance_context(
        self,
        text: str,
        sentiment: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Enhance context with additional information."""
        try:
            enhanced = {
                "sentiment": sentiment,
                "length": len(text),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            if context:
                enhanced.update({
                    "user_history": context.get("user_history", []),
                    "platform": context.get("platform", "unknown"),
                    "urgency": context.get("urgency", "normal")
                })

            return enhanced

        except Exception as e:
            logger.error(f"Error enhancing context: {str(e)}")
            return {"error": str(e)}

    async def _generate_contextual_response(
        self,
        text: str,
        context: Dict[str, Any]
    ) -> str:
        """Generate response using context-aware model."""
        try:
            # Prepare prompt
            prompt = self._prepare_prompt(text, context)

            # Generate response
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )

            with torch.no_grad():
                outputs = self.response_model.generate(
                    **inputs,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    num_return_sequences=1,
                    do_sample=True
                )

            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            return response

        except Exception as e:
            logger.error(f"Error generating contextual response: {str(e)}")
            return self._get_fallback_response(context)

    async def _validate_response(
        self,
        response: str,
        original_text: str,
        context: Dict[str, Any]
    ) -> str:
        """Validate and ensure response quality."""
        try:
            # Check response length
            if len(response) < 10:
                return self._get_fallback_response(context)

            # Check response relevance
            relevance = await self._check_response_relevance(
                response,
                original_text
            )
            if relevance < 0.5:
                return self._get_fallback_response(context)

            # Check sentiment alignment
            if not await self._check_sentiment_alignment(
                response,
                context.get("sentiment", {})
            ):
                return self._get_fallback_response(context)

            return response

        except Exception as e:
            logger.error(f"Error validating response: {str(e)}")
            return self._get_fallback_response(context)

    def _prepare_prompt(
        self,
        text: str,
        context: Dict[str, Any]
    ) -> str:
        """Prepare context-aware prompt for response generation."""
        try:
            # Build prompt parts
            parts = [
                "Generate a professional response to the following message:",
                f"Message: {text}",
                f"Sentiment: {context.get('sentiment', {}).get('label', 'neutral')}",
                f"Platform: {context.get('platform', 'unknown')}",
                f"Urgency: {context.get('urgency', 'normal')}"
            ]

            # Add user history if available
            history = context.get("user_history", [])
            if history:
                parts.append("Previous interactions:")
                for interaction in history[-3:]:  # Last 3 interactions
                    parts.append(f"- {interaction}")

            return "\n".join(parts)

        except Exception:
            return f"Generate a professional response to: {text}"

    async def _check_response_relevance(
        self,
        response: str,
        original_text: str
    ) -> float:
        """Check relevance of response to original text."""
        try:
            # Encode texts
            inputs = self.tokenizer(
                [original_text, response],
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            # Get relevance score
            with torch.no_grad():
                outputs = self.context_model(**inputs)
                scores = torch.softmax(outputs.logits, dim=1)
                relevance = scores[:, 1].item()  # Entailment score

            return relevance

        except Exception:
            return 0.5

    async def _check_sentiment_alignment(
        self,
        response: str,
        sentiment: Dict[str, Any]
    ) -> bool:
        """Check if response sentiment aligns with context."""
        try:
            response_sentiment = await self.sentiment_service.analyze_sentiment(
                response
            )

            # Check if sentiments match or are appropriately aligned
            original_label = sentiment.get("label", "neutral")
            response_label = response_sentiment.get("label", "neutral")

            if original_label == "negative":
                return response_label in ["neutral", "positive"]
            elif original_label == "positive":
                return response_label in ["neutral", "positive"]
            return True

        except Exception:
            return True

    def _get_fallback_response(self, context: Dict[str, Any]) -> str:
        """Get appropriate fallback response."""
        sentiment = context.get("sentiment", {}).get("label", "neutral")
        urgency = context.get("urgency", "normal")

        if urgency == "high":
            return "I understand this is urgent. Our team will address this right away."
        elif sentiment == "negative":
            return "I apologize for any inconvenience. We're here to help resolve this situation."
        elif sentiment == "positive":
            return "Thank you for your positive feedback! We appreciate your support."
        return "Thank you for reaching out. How can we assist you further?"
