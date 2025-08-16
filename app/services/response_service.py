"""
Automated response service.
Provides intelligent responses to reputation-related interactions.
"""

import json
import logging
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import redis.asyncio as redis
import torch
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer, pipeline)

from app.core.config import get_settings
from app.core.metrics import RESPONSE_GENERATION_LATENCY
from app.core.optimizations import CircuitBreaker, cache_warmer
from app.services.sentiment_service import SentimentService

logger = logging.getLogger(__name__)
settings = get_settings()


class ResponseTemplate:
    """Response template management."""

    def __init__(self):
        """Initialize response template."""
        self.redis = redis.Redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )

    async def get_template(
        self,
        template_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get response template."""
        template = await self.redis.get(f"template:{template_id}")
        return json.loads(template) if template else None

    async def save_template(
        self,
        template: Dict[str, Any]
    ) -> str:
        """Save response template."""
        template_id = str(uuid.uuid4())
        await self.redis.set(
            f"template:{template_id}",
            json.dumps(template)
        )
        return template_id

    async def get_templates_by_category(
        self,
        category: str
    ) -> List[Dict[str, Any]]:
        """Get templates by category."""
        template_ids = await self.redis.smembers(f"templates:{category}")
        templates = []

        for template_id in template_ids:
            template = await self.get_template(template_id)
            if template:
                templates.append(template)

        return templates


class ResponseGenerator:
    """Intelligent response generation."""

    def __init__(self):
        """Initialize response generator."""
        self.sentiment_service = SentimentService()
        self.template_manager = ResponseTemplate()

    async def generate_response(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate appropriate response."""
        # Analyze message sentiment
        sentiment = await self.sentiment_service.analyze_text(message)

        # Get appropriate template category
        category = self._determine_template_category(
            sentiment,
            context
        )

        # Get matching templates
        templates = await self.template_manager.get_templates_by_category(
            category
        )

        if not templates:
            return self._generate_fallback_response()

        # Select best template
        template = self._select_template(
            templates,
            sentiment,
            context
        )

        # Personalize response
        response = self._personalize_response(
            template,
            context
        )

        return {
            'response_text': response,
            'sentiment': sentiment,
            'category': category,
            'template_id': template['id']
        }

    def _determine_template_category(
        self,
        sentiment: Dict[str, float],
        context: Dict[str, Any]
    ) -> str:
        """Determine appropriate template category."""
        if sentiment['score'] <= -0.5:
            return 'negative'
        elif sentiment['score'] >= 0.5:
            return 'positive'

        # Consider context
        if context.get('priority') == 'high':
            return 'urgent'
        elif context.get('is_complaint'):
            return 'complaint'

        return 'neutral'

    def _select_template(
        self,
        templates: List[Dict[str, Any]],
        sentiment: Dict[str, float],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Select most appropriate template."""
        scored_templates = []

        for template in templates:
            score = self._calculate_template_score(
                template,
                sentiment,
                context
            )
            scored_templates.append((score, template))

        return max(scored_templates, key=lambda x: x[0])[1]

    def _calculate_template_score(
        self,
        template: Dict[str, Any],
        sentiment: Dict[str, float],
        context: Dict[str, Any]
    ) -> float:
        """Calculate template appropriateness score."""
        score = 0.0

        # Match sentiment
        if abs(template['sentiment_score'] - sentiment['score']) < 0.3:
            score += 0.3

        # Match urgency
        if template.get('priority') == context.get('priority'):
            score += 0.2

        # Match topic
        if template.get('topic') == context.get('topic'):
            score += 0.3

        # Consider success rate
        success_rate = template.get('success_rate', 0.5)
        score += 0.2 * success_rate

        return score

    def _personalize_response(
        self,
        template: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Personalize template with context."""
        response = template['text']

        # Replace placeholders
        placeholders = {
            '{name}': context.get('user_name', 'valued customer'),
            '{company}': context.get('company_name', 'our company'),
            '{product}': context.get('product_name', 'our product'),
            '{issue}': context.get('issue_description', 'the issue'),
            '{resolution}': context.get('resolution_steps', 'resolution'),
            '{timeframe}': context.get('timeframe', 'as soon as possible')
        }

        for placeholder, value in placeholders.items():
            response = response.replace(placeholder, value)

        return response

    def _generate_fallback_response(self) -> Dict[str, Any]:
        """Generate fallback response."""
        return {
            'response_text': "Thank you for your message. We'll review it and respond shortly.",
            'sentiment': {
                'score': 0,
                'magnitude': 0},
            'category': 'fallback',
            'template_id': None}


class ResponseOptimizer:
    """Response optimization system."""

    def __init__(self):
        """Initialize response optimizer."""
        self.redis = redis.Redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )

    async def track_response_success(
        self,
        template_id: str,
        success: bool
    ):
        """Track response success rate."""
        key = f"template:stats:{template_id}"

        async with self.redis.pipeline() as pipe:
            await pipe.hincrby(key, 'total', 1)
            if success:
                await pipe.hincrby(key, 'successful', 1)
            await pipe.execute()

    async def get_template_stats(
        self,
        template_id: str
    ) -> Dict[str, float]:
        """Get template success statistics."""
        key = f"template:stats:{template_id}"
        stats = await self.redis.hgetall(key)

        total = int(stats.get('total', 0))
        successful = int(stats.get('successful', 0))

        return {
            'total_uses': total,
            'successful_uses': successful,
            'success_rate': successful / total if total > 0 else 0
        }

    async def optimize_templates(self):
        """Optimize response templates."""
        # Get all templates
        template_ids = await self.redis.smembers('templates:all')

        for template_id in template_ids:
            stats = await self.get_template_stats(template_id)

            # Remove or flag poorly performing templates
            if stats['total_uses'] >= 10 and stats['success_rate'] < 0.3:
                await self.redis.sadd('templates:review', template_id)


class ResponsePersonalizer:
    """Advanced response personalization system."""

    def __init__(self):
        """Initialize response personalizer."""
        self.model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
        self.redis = redis.Redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )

    async def personalize_response(
        self,
        template: str,
        user_data: Dict[str, Any],
        interaction_history: List[Dict[str, Any]]
    ) -> str:
        """Personalize response based on user data and history."""
        try:
            # Build context
            context = self._build_personalization_context(
                user_data,
                interaction_history
            )

            # Generate personalized response
            response = await self._generate_personalized_response(
                template,
                context
            )

            return response

        except Exception as e:
            logger.error("Error personalizing response: %s", e)
            return template

    def _build_personalization_context(
        self,
        user_data: Dict[str, Any],
        interaction_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build context for personalization."""
        context = {
            'user_preferences': self._extract_preferences(interaction_history),
            'communication_style': self._analyze_communication_style(interaction_history),
            'sentiment_history': self._analyze_sentiment_history(interaction_history),
            'user_profile': user_data}

        return context

    def _extract_preferences(
        self,
        history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract user preferences from interaction history."""
        preferences = {
            'response_length': 'medium',
            'formality_level': 'professional',
            'detail_level': 'balanced'
        }

        if history:
            # Analyze successful interactions
            successful = [
                h for h in history
                if h.get('success', False)
            ]

            if successful:
                # Update preferences based on successful interactions
                avg_length = sum(len(h['response'])
                                 for h in successful) / len(successful)
                if avg_length > 200:
                    preferences['response_length'] = 'long'
                elif avg_length < 100:
                    preferences['response_length'] = 'short'

        return preferences

    def _analyze_communication_style(
        self,
        history: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Analyze user's communication style."""
        if not history:
            return {
                'formality': 0.7,
                'directness': 0.5,
                'emotionality': 0.3
            }

        # Analyze message characteristics
        styles = []
        for interaction in history:
            if 'message' in interaction:
                style = self._analyze_message_style(interaction['message'])
                styles.append(style)

        if not styles:
            return {
                'formality': 0.7,
                'directness': 0.5,
                'emotionality': 0.3
            }

        # Average style metrics
        return {
            key: sum(s[key] for s in styles) / len(styles)
            for key in styles[0].keys()
        }

    def _analyze_message_style(
        self,
        message: str
    ) -> Dict[str, float]:
        """Analyze style of individual message."""
        # Implement style analysis logic
        return {
            'formality': 0.7,
            'directness': 0.5,
            'emotionality': 0.3
        }

    def _analyze_sentiment_history(
        self,
        history: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Analyze sentiment patterns in interaction history."""
        if not history:
            return {
                'average_sentiment': 0.0,
                'sentiment_stability': 1.0,
                'trend': 0.0
            }

        sentiments = [
            h.get('sentiment', {}).get('score', 0)
            for h in history
            if 'sentiment' in h
        ]

        if not sentiments:
            return {
                'average_sentiment': 0.0,
                'sentiment_stability': 1.0,
                'trend': 0.0
            }

        return {
            'average_sentiment': sum(sentiments) / len(sentiments),
            'sentiment_stability': np.std(sentiments),
            'trend': sentiments[-1] - sentiments[0] if len(sentiments) > 1 else 0
        }

    async def _generate_personalized_response(
        self,
        template: str,
        context: Dict[str, Any]
    ) -> str:
        """Generate personalized response using AI model."""
        try:
            # Prepare prompt
            prompt = self._prepare_personalization_prompt(template, context)

            # Generate response
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=200,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )

            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            return self._post_process_response(response, context)

        except Exception as e:
            logger.error("Error generating personalized response: %s", e)
            return template

    def _prepare_personalization_prompt(
        self,
        template: str,
        context: Dict[str, Any]
    ) -> str:
        """Prepare prompt for response personalization."""
        style = context['communication_style']
        preferences = context['user_preferences']

        prompt = f"""
        Personalize the following response:
        {template}

        Style guidelines:
        - Formality: {style['formality']:.1f}
        - Directness: {style['directness']:.1f}
        - Emotionality: {style['emotionality']:.1f}

        Preferences:
        - Length: {preferences['response_length']}
        - Detail level: {preferences['detail_level']}

        Generate a personalized version that matches these characteristics.
        """

        return prompt

    def _post_process_response(
        self,
        response: str,
        context: Dict[str, Any]
    ) -> str:
        """Post-process generated response."""
        # Clean up response
        response = response.strip()

        # Adjust length if needed
        target_length = {
            'short': 100,
            'medium': 150,
            'long': 250
        }.get(context['user_preferences']['response_length'], 150)

        if len(response) > target_length:
            response = self._truncate_response(response, target_length)

        return response

    def _truncate_response(
        self,
        response: str,
        target_length: int
    ) -> str:
        """Intelligently truncate response to target length."""
        if len(response) <= target_length:
            return response

        # Try to truncate at sentence boundary
        sentences = response.split('.')
        truncated = ''

        for sentence in sentences:
            if len(truncated) + len(sentence) + 1 > target_length:
                break
            truncated += sentence + '.'

        return truncated.strip()


class ABTester:
    """A/B testing system for response templates."""

    def __init__(self):
        """Initialize A/B tester."""
        self.redis = redis.Redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )

    async def get_template_variant(
        self,
        template_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Get template variant for A/B testing."""
        variants = await self._get_template_variants(template_id)

        if not variants:
            return None

        # Select variant based on user_id
        user_digest = int(hashlib.sha256(user_id.encode("utf-8")).hexdigest(), 16)
        variant_index = user_digest % len(variants)
        variant = variants[variant_index]

        # Record exposure
        await self._record_exposure(template_id, variant['id'], user_id)

        return variant

    async def record_outcome(
        self,
        template_id: str,
        variant_id: str,
        user_id: str,
        success: bool
    ):
        """Record outcome of template variant."""
        key = f"ab:outcomes:{template_id}:{variant_id}"

        async with self.redis.pipeline() as pipe:
            await pipe.hincrby(key, 'total', 1)
            if success:
                await pipe.hincrby(key, 'successful', 1)
            await pipe.execute()

    async def get_test_results(
        self,
        template_id: str
    ) -> Dict[str, Any]:
        """Get A/B test results for template."""
        variants = await self._get_template_variants(template_id)
        results = {}

        for variant in variants:
            variant_id = variant['id']
            key = f"ab:outcomes:{template_id}:{variant_id}"
            stats = await self.redis.hgetall(key)

            total = int(stats.get('total', 0))
            successful = int(stats.get('successful', 0))

            results[variant_id] = {
                'total_uses': total,
                'successful_uses': successful,
                'success_rate': successful / total if total > 0 else 0
            }

        return results

    async def _get_template_variants(
        self,
        template_id: str
    ) -> List[Dict[str, Any]]:
        """Get variants for template."""
        variants_json = await self.redis.get(f"ab:variants:{template_id}")
        return json.loads(variants_json) if variants_json else []

    async def _record_exposure(
        self,
        template_id: str,
        variant_id: str,
        user_id: str
    ):
        """Record template variant exposure."""
        key = f"ab:exposures:{template_id}:{variant_id}"
        await self.redis.sadd(key, user_id)


class ResponseService:
    """Automated response service."""

    def __init__(self):
        """Initialize response service."""
        self.generator = ResponseGenerator()
        self.optimizer = ResponseOptimizer()
        self.personalizer = ResponsePersonalizer()
        self.ab_tester = ABTester()

    @CircuitBreaker(failure_threshold=3, reset_timeout=30)
    async def handle_interaction(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle incoming interaction."""
        start_time = datetime.now(timezone.utc)

        # Generate base response
        response = await self.generator.generate_response(
            message,
            context
        )

        # Get A/B test variant if available
        if response.get('template_id'):
            variant = await self.ab_tester.get_template_variant(
                response['template_id'],
                context['user_id']
            )

            if variant:
                response['template'] = variant

        # Personalize response
        if 'user_data' in context:
            history = await self._get_interaction_history(context['user_id'])
            response['response_text'] = await self.personalizer.personalize_response(
                response['response_text'],
                context['user_data'],
                history
            )

        # Record interaction
        await self._record_interaction(
            message,
            response,
            context
        )

        # Record latency
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        RESPONSE_GENERATION_LATENCY.observe(duration)

        return response

    async def _record_interaction(
        self,
        message: str,
        response: Dict[str, Any],
        context: Dict[str, Any]
    ):
        """Record interaction details."""
        record = {
            'message': message,
            'response': response,
            'context': context,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        await self.redis.lpush(
            'interactions:history',
            json.dumps(record)
        )
        await self.redis.ltrim(
            'interactions:history',
            0,
            9999  # Keep last 10000 interactions
        )

    async def feedback_loop(
        self,
        interaction_id: str,
        success: bool,
        feedback: Optional[str] = None
    ):
        """Process interaction feedback."""
        interaction = await self.redis.get(f"interaction:{interaction_id}")

        if interaction:
            data = json.loads(interaction)
            template_id = data['response'].get('template_id')
            variant_id = data['response'].get('template', {}).get('id')

            if template_id:
                # Track template success
                await self.optimizer.track_response_success(
                    template_id,
                    success
                )

                # Track A/B test outcome if applicable
                if variant_id:
                    await self.ab_tester.record_outcome(
                        template_id,
                        variant_id,
                        data['context']['user_id'],
                        success
                    )

            if feedback:
                await self.redis.lpush(
                    f"interaction:feedback:{interaction_id}",
                    feedback
                )

    @cache_warmer(['templates:stats'])
    async def optimize_responses(self):
        """Run response optimization."""
        await self.optimizer.optimize_templates()

    def _extract_response_info(self, message: str) -> Dict:
        """Extract key information for response generation."""
        try:
            # Use NLP pipeline for information extraction
            nlp = pipeline("ner")
            entities = nlp(message)

            # Extract key information
            info = {
                "entities": entities,
                "keywords": self._extract_keywords(message),
                "topics": self._identify_topics(message),
                "urgency": self._assess_urgency(message)
            }

            return info

        except Exception as e:
            logger.error("Error extracting response info: %s", e)
            return {}

    async def _generate_ai_response(
        self,
        message: str,
        sentiment: Dict,
        extracted_info: Dict,
        context: Optional[Dict]
    ) -> str:
        """Generate response using transformer model."""
        try:
            # Prepare input for model
            prompt = self._prepare_response_prompt(
                message,
                sentiment,
                extracted_info,
                context
            )

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
                    max_length=150,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )

            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            return response

        except Exception as e:
            logger.error("Error generating AI response: %s", e)
            return self._get_fallback_response({"sentiment": "neutral"})

    def _prepare_response_prompt(
        self,
        message: str,
        sentiment: Dict,
        extracted_info: Dict,
        context: Optional[Dict]
    ) -> str:
        """Prepare prompt for response generation."""
        try:
            # Build context-aware prompt
            prompt_parts = [
                "Generate a professional and empathetic response to:",
                f"Message: {message}",
                f"Sentiment: {sentiment.get('sentiment', {}).get('label', 'neutral')}"
            ]

            if context:
                prompt_parts.extend([
                    f"User: {context.get('user_name', 'Customer')}",
                    f"History: {context.get('interaction_history', 'First interaction')}"
                ])

            if extracted_info:
                prompt_parts.extend([
                    f"Topics: {', '.join(extracted_info.get('topics', []))}",
                    f"Urgency: {extracted_info.get('urgency', 'normal')}"
                ])

            return "\n".join(prompt_parts)

        except Exception:
            return f"Generate a professional response to: {message}"

    def _enhance_response(
        self,
        response: str,
        context: Optional[Dict],
        extracted_info: Dict
    ) -> str:
        """Enhance response with context and personalization."""
        try:
            enhanced = response

            # Add personalization
            if context and context.get("user_name"):
                enhanced = enhanced.replace(
                    "{name}",
                    context["user_name"]
                )
            else:
                enhanced = enhanced.replace("{name}", "valued customer")

            # Add issue context
            if extracted_info.get("topics"):
                enhanced = enhanced.replace(
                    "{issue}",
                    extracted_info["topics"][0]
                )
            else:
                enhanced = enhanced.replace("{issue}", "this matter")

            # Add additional context
            if context and context.get("additional_info"):
                enhanced = enhanced.replace(
                    "{context}",
                    context["additional_info"]
                )
            else:
                enhanced = enhanced.replace("{context}", "")

            return enhanced.strip()

        except Exception as e:
            logger.error("Error enhancing response: %s", e)
            return response

    def _validate_response(
        self,
        response: str,
        original_message: str,
        context: Optional[Dict]
    ) -> str:
        """Validate and ensure response quality."""
        try:
            # Check response length
            if len(response) < 10:
                return self._get_fallback_response({"sentiment": "neutral"})

            # Check response relevance
            relevance = self._check_response_relevance(
                response,
                original_message
            )
            if relevance < 0.5:
                return self._get_fallback_response({"sentiment": "neutral"})

            # Check tone appropriateness
            if not self._check_tone_appropriateness(response, context):
                return self._get_fallback_response({"sentiment": "neutral"})

            return response

        except Exception as e:
            logger.error("Error validating response: %s", e)
            return self._get_fallback_response({"sentiment": "neutral"})

    def _check_response_relevance(
        self,
        response: str,
        original_message: str
    ) -> float:
        """Check relevance of response to original message."""
        try:
            # Encode texts
            inputs = self.tokenizer(
                [original_message, response],
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            # Get relevance score
            with torch.no_grad():
                outputs = self.classifier(**inputs)
                scores = torch.softmax(outputs.logits, dim=1)
                relevance = scores[:, 1].item()  # Entailment score

            return relevance

        except Exception:
            return 0.5

    def _check_tone_appropriateness(
        self,
        response: str,
        context: Optional[Dict]
    ) -> bool:
        """Check if response tone is appropriate."""
        try:
            # Implementation would check tone appropriateness
            return True
        except Exception:
            return True

    async def _record_response(
        self,
        response: str,
        sentiment: Dict,
        context: Optional[Dict]
    ):
        """Record response for analysis and improvement."""
        try:
            record = {
                "response": response,
                "sentiment": sentiment,
                "context": context,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            await self.redis.lpush(
                "response_history",
                json.dumps(record)
            )
            await self.redis.ltrim("response_history", 0, 999)

        except Exception as e:
            logger.error("Error recording response: %s", e)

    def _get_fallback_response(self, sentiment: Dict) -> str:
        """Get appropriate fallback response."""
        sentiment_label = sentiment.get(
            "sentiment", {}).get(
            "label", "neutral")
        templates = self.response_templates.get(
            sentiment_label, self.response_templates["neutral"])
        return templates[0].format(
            name="valued customer",
            issue="this matter",
            context="")

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        try:
            # Implementation would extract keywords
            return []
        except Exception:
            return []

    def _identify_topics(self, text: str) -> List[str]:
        """Identify topics in text."""
        try:
            # Implementation would identify topics
            return []
        except Exception:
            return []

    def _assess_urgency(self, text: str) -> str:
        """Assess message urgency."""
        try:
            # Implementation would assess urgency
            return "normal"
        except Exception:
            return "normal"

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=4, max=10))
    async def check_crisis(
        self,
        metrics: Dict,
        timeframe: str = "1h"
    ) -> Dict:
        """Check for potential crisis situations with advanced detection."""
        try:
            start_time = datetime.now(timezone.utc)

            # Analyze metrics
            crisis_indicators = await self._analyze_crisis_indicators(metrics)

            # Determine crisis level
            crisis_level = self._determine_crisis_level(crisis_indicators)

            if crisis_level != "none":
                self.crisis_counter.inc()

            # Generate response plan
            response_plan = await self._generate_crisis_response_plan(
                crisis_level,
                crisis_indicators
            )

            return {
                "is_crisis": crisis_level != "none",
                "level": crisis_level,
                "indicators": crisis_indicators,
                "response_plan": response_plan,
                "metadata": {
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                    "processing_time": (
                        datetime.now(timezone.utc) - start_time
                    ).total_seconds(),
                    "timeframe": timeframe
                }
            }

        except Exception as e:
            logger.error("Error checking crisis status: %s", e)
            raise

    async def _analyze_crisis_indicators(self, metrics: Dict) -> List[Dict]:
        """Analyze metrics for crisis indicators."""
        try:
            indicators = []

            # Check mention spike
            mention_ratio = metrics.get("mention_ratio", 1.0)
            if mention_ratio > self.crisis_thresholds["mention_spike"]["moderate"]:
                indicators.append({
                    "type": "mention_spike",
                    "value": mention_ratio,
                    "threshold": self.crisis_thresholds["mention_spike"],
                    "severity": self._get_severity_level(
                        mention_ratio,
                        self.crisis_thresholds["mention_spike"]
                    )
                })

            # Check negative sentiment ratio
            negative_ratio = metrics.get("negative_ratio", 0.0)
            if negative_ratio > self.crisis_thresholds["negative_ratio"]["moderate"]:
                indicators.append({
                    "type": "negative_sentiment",
                    "value": negative_ratio,
                    "threshold": self.crisis_thresholds["negative_ratio"],
                    "severity": self._get_severity_level(
                        negative_ratio,
                        self.crisis_thresholds["negative_ratio"]
                    )
                })

            # Check engagement drop
            engagement_drop = metrics.get("engagement_drop", 0.0)
            if engagement_drop > self.crisis_thresholds["engagement_drop"]["moderate"]:
                indicators.append({
                    "type": "engagement_drop",
                    "value": engagement_drop,
                    "threshold": self.crisis_thresholds["engagement_drop"],
                    "severity": self._get_severity_level(
                        engagement_drop,
                        self.crisis_thresholds["engagement_drop"]
                    )
                })

            return indicators

        except Exception as e:
            logger.error("Error analyzing crisis indicators: %s", e)
            return []

    def _determine_crisis_level(self, indicators: List[Dict]) -> str:
        """Determine overall crisis level."""
        try:
            if not indicators:
                return "none"

            # Get maximum severity
            severities = [i["severity"] for i in indicators]
            max_severity = max(severities)

            # Count severe and critical indicators
            severe_count = len(
                [s for s in severities if s in ["severe", "critical"]])

            if max_severity == "critical" or severe_count >= 2:
                return "critical"
            elif max_severity == "severe" or severe_count >= 1:
                return "severe"
            elif max_severity == "moderate":
                return "moderate"
            else:
                return "none"

        except Exception:
            return "none"

    def _get_severity_level(
        self,
        value: float,
        thresholds: Dict
    ) -> str:
        """Get severity level based on thresholds."""
        if value >= thresholds["critical"]:
            return "critical"
        elif value >= thresholds["severe"]:
            return "severe"
        elif value >= thresholds["moderate"]:
            return "moderate"
        return "normal"

    async def _generate_crisis_response_plan(
        self,
        level: str,
        indicators: List[Dict]
    ) -> Dict:
        """Generate comprehensive crisis response plan."""
        try:
            if level == "none":
                return {}

            # Get base recommendations
            recommendations = self._get_crisis_recommendations(
                level, indicators)

            # Add immediate actions
            immediate_actions = await self._get_immediate_actions(
                level,
                indicators
            )

            # Generate communication strategy
            communication_strategy = self._generate_communication_strategy(
                level,
                indicators
            )

            # Create monitoring plan
            monitoring_plan = self._create_monitoring_plan(level)

            return {
                "recommendations": recommendations,
                "immediate_actions": immediate_actions,
                "communication_strategy": communication_strategy,
                "monitoring_plan": monitoring_plan,
                "escalation_path": self._get_escalation_path(level)
            }

        except Exception as e:
            logger.error("Error generating crisis response plan: %s", e)
            return {}

    def _get_crisis_recommendations(
        self,
        level: str,
        indicators: List[Dict]
    ) -> List[Dict]:
        """Get recommended actions for crisis management."""
        recommendations = []

        if level == "high":
            recommendations.extend([{"action": "alert_management",
                                     "priority": "immediate",
                                     "description": "Alert senior management immediately"},
                                    {"action": "prepare_statement",
                                     "priority": "high",
                                     "description": "Prepare official statement addressing concerns"},
                                    {"action": "monitor_channels",
                                     "priority": "high",
                                     "description": "Increase monitoring frequency across all channels"}])
        elif level == "medium":
            recommendations.extend([{"action": "increase_monitoring",
                                     "priority": "high",
                                     "description": "Increase monitoring frequency"},
                                    {"action": "prepare_responses",
                                     "priority": "medium",
                                     "description": "Prepare response templates for common issues"}])

        # Add specific recommendations based on indicators
        for indicator in indicators:
            if indicator["type"] == "mention_spike":
                recommendations.append({
                    "action": "analyze_mentions",
                    "priority": "high",
                    "description": "Analyze mention patterns and common themes"
                })
            elif indicator["type"] == "high_negative_sentiment":
                recommendations.append({
                    "action": "sentiment_analysis",
                    "priority": "high",
                    "description": "Conduct detailed sentiment analysis"
                })

        return recommendations

    async def auto_respond(
        self,
        comment_id: str,
        platform: str,
        text: str,
        user_id: str
    ) -> Dict:
        """Generate and post automated response."""
        try:
            # Analyze sentiment
            sentiment = await self.sentiment_service.analyze_sentiment(text)

            # Get appropriate response template
            response = await self._get_response_template(
                sentiment['sentiment'],
                platform,
                user_id
            )

            # Store response record
            response_data = {
                'comment_id': comment_id,
                'platform': platform,
                'original_text': text,
                'sentiment': sentiment,
                'response': response,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'user_id': user_id
            }

            await self.redis.hset(
                'responses',
                comment_id,
                json.dumps(response_data)
            )

            return response_data

        except Exception as e:
            logger.error("Error generating response: %s", e)
            raise ValueError("Error generating response: %s", e)

    async def escalate_issue(
        self,
        issue_data: Dict,
        priority: str = "medium"
    ) -> Dict:
        """Escalate an issue for human intervention."""
        try:
            issue_id = f"issue_{datetime.now(timezone.utc).timestamp()}"

            # Enrich issue data
            issue_data.update({
                'id': issue_id,
                'status': 'open',
                'priority': priority,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat()
            })

            # Store issue
            await self.redis.hset(
                'issues',
                issue_id,
                json.dumps(issue_data)
            )

            # Add to priority queue
            await self.redis.zadd(
                'issue_queue',
                {issue_id: self._get_priority_score(priority)}
            )

            return issue_data

        except Exception as e:
            logger.error("Error escalating issue: %s", e)
            raise ValueError("Error escalating issue: %s", e)

    async def check_crisis_conditions(
        self,
        platform: str,
        timeframe: int = 3600
    ) -> Dict:
        """Check for potential crisis conditions."""
        try:
            # Get recent mentions
            mentions = await self._get_recent_mentions(platform, timeframe)

            # Analyze mentions
            sentiments = await self.sentiment_service.analyze_bulk_sentiment(
                [m['text'] for m in mentions]
            )

            # Calculate metrics
            negative_count = sum(
                1 for s in sentiments
                if s['sentiment'] == 'negative'
            )
            avg_sentiment = sum(
                s['polarity'] for s in sentiments
            ) / len(sentiments) if sentiments else 0

            is_crisis = (
                negative_count >= self.crisis_thresholds['negative_mentions']
                or avg_sentiment <= self.crisis_thresholds['sentiment_threshold']
            )

            crisis_data = {
                'is_crisis': is_crisis,
                'platform': platform,
                'negative_mentions': negative_count,
                'total_mentions': len(mentions),
                'average_sentiment': avg_sentiment,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

            if is_crisis:
                await self._handle_crisis(crisis_data)

            return crisis_data

        except Exception as e:
            logger.error("Error checking crisis conditions: %s", e)
            raise ValueError("Error checking crisis conditions: %s", e)

    async def update_issue(
        self,
        issue_id: str,
        updates: Dict
    ) -> Dict:
        """Update an existing issue."""
        try:
            issue_data = await self.redis.hget('issues', issue_id)
            if not issue_data:
                raise ValueError("Issue %s not found", issue_id)

            issue = json.loads(issue_data)
            issue.update(updates)
            issue['updated_at'] = datetime.now(timezone.utc).isoformat()

            await self.redis.hset(
                'issues',
                issue_id,
                json.dumps(issue)
            )

            return issue

        except Exception as e:
            logger.error("Error updating issue: %s", e)
            raise ValueError("Error updating issue: %s", e)

    async def _get_response_template(
        self,
        sentiment: str,
        platform: str,
        user_id: str
    ) -> str:
        """Get appropriate response template based on context."""
        try:
            templates = self.response_templates.get(sentiment, [])
            if not templates:
                return self.response_templates['neutral'][0]

            # Get user interaction history
            await self._get_user_history(platform, user_id)

            # Select template based on history and context
            # For now, just return first template
            return templates[0]

        except Exception:
            return self.response_templates['neutral'][0]

    def _get_priority_score(self, priority: str) -> float:
        """Convert priority string to numeric score."""
        scores = {
            'low': 1.0,
            'medium': 2.0,
            'high': 3.0,
            'critical': 4.0
        }
        return scores.get(priority.lower(), 1.0)

    async def _get_recent_mentions(
        self,
        platform: str,
        timeframe: int
    ) -> List[Dict]:
        """Get recent mentions from platform."""
        # Implementation would depend on platform-specific services
        return []

    async def _get_user_history(
        self,
        platform: str,
        user_id: str
    ) -> List[Dict]:
        """Get user interaction history."""
        try:
            history = await self.redis.lrange(
                f'user_history:{platform}:{user_id}',
                0,
                -1
            )
            return [json.loads(h) for h in history]
        except Exception:
            return []

    async def _handle_crisis(self, crisis_data: Dict):
        """Handle detected crisis situation."""
        try:
            # Create high-priority issue
            await self.escalate_issue(
                {
                    'type': 'crisis',
                    'data': crisis_data,
                    'requires_immediate_action': True
                },
                priority='critical'
            )

            # Store crisis record
            crisis_id = f"crisis_{datetime.now(timezone.utc).timestamp()}"
            await self.redis.hset(
                'crises',
                crisis_id,
                json.dumps(crisis_data)
            )

            # Additional crisis handling logic here

        except Exception as e:
            logger.error("Error handling crisis: %s", e)

    async def _get_interaction_history(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get user's recent interaction history."""
        history = await self.redis.lrange(
            f"user:interactions:{user_id}",
            0,
            limit - 1
        )

        return [
            json.loads(interaction)
            for interaction in history
        ]
