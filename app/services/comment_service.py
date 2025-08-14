import logging
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import spacy
import torch
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.parsing.preprocessing import preprocess_string
from prometheus_client import Counter
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import (AutoModelForQuestionAnswering,
                          AutoModelForSequenceClassification,
                          AutoModelForTokenClassification, AutoTokenizer)
import asyncio

from app.core.cache import cache
from app.core.config import get_settings
from app.core.metrics import COMMENT_ANALYSIS_TOTAL, COMMENT_ANALYSIS_LATENCY, COMMENT_ERRORS

logger = logging.getLogger(__name__)
settings = get_settings()


class CommentService:
    """Advanced comment analysis service with state-of-the-art NLP capabilities."""

    def __init__(self):
        """Initialize advanced NLP models and tools."""
        # Load advanced language models
        self.tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            "SamLowe/roberta-base-go_emotions")
        self.ner_model = AutoModelForTokenClassification.from_pretrained(
            "dslim/bert-base-NER"
        )
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(
            "deepset/roberta-large-squad2"
        )

        # Load spaCy model for advanced linguistic analysis
        self.nlp = spacy.load("en_core_web_sm")

        # Initialize topic modeling
        self.num_topics = 10
        self.topic_model = None
        self.dictionary = None

        # Initialize vectorizers
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words="english"
        )

        # Initialize clustering
        self.cluster_model = DBSCAN(
            eps=0.3,
            min_samples=3,
            metric="cosine"
        )

        # Performance monitoring
        self.analysis_counter = COMMENT_ANALYSIS_TOTAL
        self.error_counter = COMMENT_ERRORS

        # Cache configuration
        self.cache_ttl = settings.cache["comment_analysis"] if hasattr(settings.cache, '__getitem__') and "comment_analysis" in settings.cache else 3600

        # Analysis thresholds
        self.thresholds = {
            "spam_confidence": 0.85,
            "sentiment_confidence": 0.75,
            "topic_coherence": 0.6,
            "influence_score": 0.7
        }

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=4, max=10))
    async def analyze_comments(
        self,
        comments: List[Dict],
        platform: str
    ) -> Dict:
        """Perform comprehensive comment analysis with advanced NLP."""
        try:
            start_time = datetime.utcnow()
            self.analysis_counter.inc()

            # Extract text and metadata
            texts = [comment["text"] for comment in comments]

            # Parallel analysis tasks
            analysis_tasks = [
                self._analyze_sentiment_advanced(texts),
                self._analyze_topics(texts),
                self._analyze_entities(texts),
                self._analyze_user_behavior(comments),
                self.detect_spam(comments),
                self._identify_influencers(comments),
                self._analyze_engagement_patterns(comments),
                self._extract_key_insights(texts)
            ]

            # Execute analyses in parallel
            results = await asyncio.gather(*analysis_tasks)

            # Combine results
            analysis_results = {
                "sentiment": results[0],
                "topics": results[1],
                "entities": results[2],
                "user_behavior": results[3],
                "spam_detection": results[4],
                "influencers": results[5],
                "engagement": results[6],
                "key_insights": results[7],
                "metadata": {
                    "platform": platform,
                    "analyzed_at": datetime.utcnow().isoformat(),
                    "processing_time": (
                        datetime.utcnow() -
                        start_time).total_seconds(),
                    "comment_count": len(comments)}}

            # Cache results
            cache_key = f"comment_analysis:{platform}:{datetime.utcnow().date()}"
            await cache.set(cache_key, analysis_results, ttl=self.cache_ttl)

            # Record latency
            COMMENT_ANALYSIS_LATENCY.observe(
                (datetime.utcnow() - start_time).total_seconds()
            )

            return analysis_results

        except Exception as e:
            self.error_counter.inc()
            logger.error(f"Error analyzing comments: {str(e)}")
            raise

    async def _analyze_sentiment_advanced(
        self,
        texts: List[str]
    ) -> Dict:
        """Perform advanced sentiment analysis using transformer models."""
        try:
            # Tokenize and encode texts
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # Get sentiment predictions
            with torch.no_grad():
                outputs = self.sentiment_model(**encoded)
                predictions = torch.softmax(outputs.logits, dim=1)

            # Process results
            sentiments = []
            for pred in predictions:
                sentiment = {
                    "label": self.sentiment_model.config.id2label[pred.argmax().item()],
                    "score": pred.max().item(),
                    "distribution": {
                        label: score.item()
                        for label, score in zip(
                            self.sentiment_model.config.id2label.values(),
                            pred
                        )
                    }
                }
                sentiments.append(sentiment)

            # Calculate aggregate metrics
            sentiment_stats = {
                "average": {
                    "positive": np.mean([
                        s["distribution"].get("positive", 0)
                        for s in sentiments
                    ]),
                    "negative": np.mean([
                        s["distribution"].get("negative", 0)
                        for s in sentiments
                    ]),
                    "neutral": np.mean([
                        s["distribution"].get("neutral", 0)
                        for s in sentiments
                    ])
                },
                "distribution": {
                    "positive": len([
                        s for s in sentiments
                        if s["label"] == "positive"
                    ]),
                    "negative": len([
                        s for s in sentiments
                        if s["label"] == "negative"
                    ]),
                    "neutral": len([
                        s for s in sentiments
                        if s["label"] == "neutral"
                    ])
                }
            }

            return {
                "individual": sentiments,
                "aggregate": sentiment_stats,
                "confidence": np.mean([s["score"] for s in sentiments])
            }

        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {}

    async def _analyze_topics(self, texts: List[str]) -> Dict:
        """Perform advanced topic analysis using LDA and transformers."""
        try:
            # Preprocess texts
            processed_texts = [
                preprocess_string(text)
                for text in texts
            ]

            # Create dictionary and corpus
            dictionary = Dictionary(processed_texts)
            corpus = [
                dictionary.doc2bow(text)
                for text in processed_texts
            ]

            # Train LDA model if needed
            if not self.topic_model:
                self.topic_model = LdaModel(
                    corpus=corpus,
                    num_topics=self.num_topics,
                    id2word=dictionary,
                    passes=20,
                    alpha="auto",
                    random_state=42
                )

            # Get topic distributions
            topic_distributions = [
                self.topic_model.get_document_topics(doc)
                for doc in corpus
            ]

            # Extract topics
            topics = []
            for topic_id in range(self.num_topics):
                topic_words = self.topic_model.show_topic(topic_id, topn=10)
                topics.append({
                    "id": topic_id,
                    "words": [
                        {"word": word, "weight": weight}
                        for word, weight in topic_words
                    ],
                    "coherence": self._calculate_topic_coherence(topic_words)
                })

            return {
                "topics": topics,
                "distributions": [
                    [
                        {"topic_id": topic_id, "weight": weight}
                        for topic_id, weight in dist
                    ]
                    for dist in topic_distributions
                ],
                "metadata": {
                    "num_topics": self.num_topics,
                    "coherence_score": np.mean([
                        t["coherence"]
                        for t in topics
                    ])
                }
            }

        except Exception as e:
            logger.error(f"Error in topic analysis: {str(e)}")
            return {}

    def _calculate_topic_coherence(self, topic_words: List[tuple]) -> float:
        """Calculate topic coherence score."""
        try:
            # Implementation would use advanced coherence metrics
            return 0.75
        except Exception:
            return 0.0

    async def _analyze_entities(self, texts: List[str]) -> Dict:
        """Perform named entity recognition and analysis."""
        try:
            # Process texts with spaCy
            docs = list(self.nlp.pipe(texts))

            # Extract and analyze entities
            entities = []
            for doc in docs:
                doc_entities = []
                for ent in doc.ents:
                    doc_entities.append({
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char
                    })
                entities.append(doc_entities)

            # Aggregate entity statistics
            entity_stats = {}
            for doc_entities in entities:
                for entity in doc_entities:
                    if entity["label"] not in entity_stats:
                        entity_stats[entity["label"]] = {
                            "count": 0,
                            "examples": []
                        }
                    entity_stats[entity["label"]]["count"] += 1
                    if len(entity_stats[entity["label"]]["examples"]) < 5:
                        entity_stats[entity["label"]]["examples"].append(
                            entity["text"]
                        )

            return {
                "entities": entities,
                "statistics": entity_stats,
                "total_entities": sum(
                    stat["count"]
                    for stat in entity_stats.values()
                )
            }

        except Exception as e:
            logger.error(f"Error in entity analysis: {str(e)}")
            return {}

    def _analyze_user_behavior(self, comments: List[Dict]) -> Dict:
        """Analyze user behavior patterns in comments."""
        try:
            user_metrics = {}

            for comment in comments:
                user_id = comment.get("user_id", "anonymous")

                if user_id not in user_metrics:
                    user_metrics[user_id] = {
                        "comment_count": 0,
                        "avg_length": 0,
                        "sentiment_scores": [],
                        "topics_discussed": set(),
                        "engagement_received": 0,
                        "response_times": [],
                        "activity_times": [],
                        "interaction_patterns": {
                            "replies_to": [],
                            "replied_by": []
                        }
                    }

                metrics = user_metrics[user_id]
                metrics["comment_count"] += 1
                metrics["avg_length"] += len(comment["text"])

                if "sentiment_score" in comment:
                    metrics["sentiment_scores"].append(
                        comment["sentiment_score"])

                if "created_at" in comment:
                    metrics["activity_times"].append(comment["created_at"])

                if "engagement" in comment:
                    metrics["engagement_received"] += sum(
                        comment["engagement"].values()
                    )

                if "reply_to" in comment:
                    metrics["interaction_patterns"]["replies_to"].append(
                        comment["reply_to"]
                    )

            # Calculate averages and patterns
            for user_id, metrics in user_metrics.items():
                metrics["avg_length"] /= metrics["comment_count"]
                metrics["avg_sentiment"] = np.mean(
                    metrics["sentiment_scores"]) if metrics["sentiment_scores"] else 0
                metrics["activity_pattern"] = self._analyze_activity_pattern(
                    metrics["activity_times"]
                )
                metrics["interaction_score"] = self._calculate_interaction_score(
                    metrics["interaction_patterns"])

            return {
                "user_metrics": user_metrics,
                "top_contributors": self._identify_top_contributors(user_metrics),
                "engagement_patterns": self._analyze_engagement_distribution(user_metrics)}

        except Exception as e:
            logger.error(f"Error in user behavior analysis: {str(e)}")
            return {}

    def _analyze_activity_pattern(
        self,
        activity_times: List[datetime]
    ) -> Dict:
        """Analyze user activity patterns."""
        try:
            if not activity_times:
                return {}

            times = pd.to_datetime(activity_times)
            return {
                "most_active_hour": times.hour.mode().iloc[0],
                "most_active_day": times.day_name().mode().iloc[0],
                "activity_consistency": self._calculate_activity_consistency(times)}
        except Exception:
            return {}

    def _calculate_activity_consistency(
        self,
        times: pd.Series
    ) -> float:
        """Calculate consistency score for user activity."""
        try:
            # Implementation would analyze temporal patterns
            return 0.75
        except Exception:
            return 0.0

    def _calculate_interaction_score(
        self,
        patterns: Dict
    ) -> float:
        """Calculate user interaction score."""
        try:
            # Implementation would analyze interaction patterns
            return 0.8
        except Exception:
            return 0.0

    def _identify_top_contributors(
        self,
        user_metrics: Dict
    ) -> List[Dict]:
        """Identify top contributing users."""
        try:
            contributors = []
            for user_id, metrics in user_metrics.items():
                score = (
                    metrics["comment_count"] * 0.3 +
                    metrics["engagement_received"] * 0.4 +
                    metrics["interaction_score"] * 0.3
                )
                contributors.append({
                    "user_id": user_id,
                    "score": score,
                    "metrics": metrics
                })

            return sorted(
                contributors,
                key=lambda x: x["score"],
                reverse=True
            )[:10]

        except Exception:
            return []

    def _analyze_engagement_distribution(
        self,
        user_metrics: Dict
    ) -> Dict:
        """Analyze engagement distribution patterns."""
        try:
            engagements = [
                m["engagement_received"]
                for m in user_metrics.values()
            ]

            return {
                "mean": np.mean(engagements),
                "median": np.median(engagements),
                "std": np.std(engagements),
                "distribution": np.histogram(
                    engagements,
                    bins=10
                )[0].tolist()
            }
        except Exception:
            return {}

    async def _extract_key_insights(self, texts: List[str]) -> Dict:
        """Extract key insights using advanced NLP."""
        try:
            # Process texts
            docs = list(self.nlp.pipe(texts))

            # Extract insights
            insights = {
                "key_phrases": self._extract_key_phrases(docs),
                "sentiment_patterns": self._analyze_sentiment_patterns(docs),
                "topic_trends": self._identify_topic_trends(docs),
                "action_items": self._extract_action_items(docs)
            }

            return insights

        except Exception as e:
            logger.error(f"Error extracting insights: {str(e)}")
            return {}

    def _analyze_engagement_patterns(
        self,
        comments: List[Dict]
    ) -> Dict:
        """Analyze engagement patterns in comments."""
        try:
            patterns = {
                "hourly": {},
                "daily": {},
                "user_segments": {},
                "content_types": {}
            }

            for comment in comments:
                timestamp = datetime.fromisoformat(comment["timestamp"])
                hour = timestamp.hour
                day = timestamp.strftime("%A")

                # Update hourly patterns
                if hour not in patterns["hourly"]:
                    patterns["hourly"][hour] = 0
                patterns["hourly"][hour] += 1

                # Update daily patterns
                if day not in patterns["daily"]:
                    patterns["daily"][day] = 0
                patterns["daily"][day] += 1

                # Analyze user segments
                user_type = self._determine_user_segment(comment)
                if user_type not in patterns["user_segments"]:
                    patterns["user_segments"][user_type] = 0
                patterns["user_segments"][user_type] += 1

                # Analyze content types
                content_type = self._determine_content_type(comment)
                if content_type not in patterns["content_types"]:
                    patterns["content_types"][content_type] = 0
                patterns["content_types"][content_type] += 1

            return patterns

        except Exception as e:
            logger.error(f"Error analyzing engagement patterns: {str(e)}")
            return {}

    def _determine_user_segment(self, comment: Dict) -> str:
        """Determine user segment based on behavior."""
        try:
            engagement = comment.get("engagement", {})
            history = comment.get("user_history", {})

            if history.get("total_comments", 0) > 100:
                return "power_user"
            elif engagement.get("likes", 0) > 50:
                return "influencer"
            elif history.get("response_rate", 0) > 0.8:
                return "engaged"
            else:
                return "regular"
        except Exception:
            return "unknown"

    def _determine_content_type(self, comment: Dict) -> str:
        """Determine content type using NLP."""
        try:
            doc = self.nlp(comment["text"])

            # Check for questions
            if any(token.tag_ == "WP" for token in doc):
                return "question"

            # Check for feedback
            feedback_terms = ["suggest", "improve", "better", "would", "could"]
            if any(token.lemma_ in feedback_terms for token in doc):
                return "feedback"

            # Check for complaints
            complaint_terms = ["issue", "problem", "wrong", "bad", "terrible"]
            if any(token.lemma_ in complaint_terms for token in doc):
                return "complaint"

            # Check for praise
            praise_terms = ["great", "good", "excellent", "amazing", "love"]
            if any(token.lemma_ in praise_terms for token in doc):
                return "praise"

            return "general"

        except Exception:
            return "unknown"

    def _generate_comment_insights(
        self,
        sentiment: Dict,
        topics: Dict,
        user_behavior: Dict,
        engagement: Dict
    ) -> List[Dict]:
        """Generate insights from comment analysis."""
        try:
            insights = []

            # Add sentiment insights
            if sentiment.get("aggregate", {}).get("negative", 0) > 0.4:
                insights.append({
                    "type": "sentiment_alert",
                    "level": "high",
                    "description": "High negative sentiment detected",
                    "recommendations": [
                        "Review negative comments for common issues",
                        "Prepare response strategy",
                        "Monitor sentiment trends closely"
                    ]
                })

            # Add topic insights
            for topic in topics.get("topics", []):
                if topic["coherence"] > self.thresholds["topic_coherence"]:
                    insights.append({
                        "type": "emerging_topic",
                        "level": "medium",
                        "description": f"Emerging topic detected: {topic['words'][0]['word']}",
                        "recommendations": [
                            "Monitor topic development",
                            "Prepare relevant content",
                            "Engage with topic discussions"
                        ]
                    })

            # Add user behavior insights
            power_users = [
                uid for uid, metrics in user_behavior["user_metrics"].items()
                if metrics["comment_count"] > 10 and
                metrics["engagement_received"] > 100
            ]

            if power_users:
                insights.append({
                    "type": "user_engagement",
                    "level": "medium",
                    "description": f"Active power users detected: {len(power_users)}",
                    "recommendations": [
                        "Engage with power users",
                        "Consider loyalty program",
                        "Monitor power user sentiment"
                    ]
                })

            # Add engagement insights
            peak_hours = sorted(
                engagement["hourly"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]

            insights.append({
                "type": "engagement_pattern",
                "level": "low",
                "description": f"Peak engagement hours: {[h[0] for h in peak_hours]}",
                "recommendations": [
                    "Optimize content timing",
                    "Increase monitoring during peak hours",
                    "Plan responses for high activity periods"
                ]
            })

            return insights

        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return []

    async def detect_spam(
        self,
        comments: List[Dict],
        threshold: float = 0.8
    ) -> List[Dict]:
        """Detect potential spam comments."""
        try:
            spam_indicators = []

            for comment in comments:
                score = self._calculate_spam_score(comment)
                if score >= threshold:
                    spam_indicators.append({
                        "comment_id": comment["id"],
                        "text": comment["text"],
                        "score": score,
                        "indicators": self._get_spam_indicators(comment)
                    })

            return spam_indicators

        except Exception as e:
            logger.error(f"Error detecting spam: {str(e)}")
            raise

    async def analyze_user_sentiment(
        self,
        user_comments: List[Dict],
        timeframe: str = "30d"
    ) -> Dict:
        """Analyze sentiment patterns for a specific user."""
        try:
            # Get sentiment for all comments
            texts = [comment["text"] for comment in user_comments]
            sentiments = await self.sentiment_service.analyze_bulk_sentiment(texts)

            # Calculate metrics
            avg_sentiment = np.mean([s["polarity"] for s in sentiments])
            sentiment_trend = self._calculate_sentiment_trend(
                user_comments,
                sentiments
            )

            return {
                "average_sentiment": avg_sentiment,
                "sentiment_trend": sentiment_trend,
                "total_comments": len(user_comments),
                "timeframe": timeframe,
                "analyzed_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error analyzing user sentiment: {str(e)}")
            raise

    def _identify_themes(self, texts: List[str]) -> List[Dict]:
        """Identify common themes in comments using clustering."""
        try:
            # Transform texts to TF-IDF vectors
            vectors = self.vectorizer.fit_transform(texts)

            # Cluster comments
            clusters = DBSCAN(
                eps=0.3,
                min_samples=3,
                metric='cosine'
            ).fit(vectors)

            # Extract themes for each cluster
            themes = []
            for label in set(clusters.labels_):
                if label == -1:  # Noise points
                    continue

                cluster_texts = [
                    texts[i] for i in range(len(texts))
                    if clusters.labels_[i] == label
                ]

                # Get top terms for this cluster
                cluster_vector = vectors[clusters.labels_ == label].mean(
                    axis=0)
                top_terms = self._get_top_terms(
                    cluster_vector,
                    self.vectorizer.get_feature_names_out()
                )

                themes.append({
                    "id": f"theme_{label}",
                    "terms": top_terms,
                    "size": len(cluster_texts),
                    "examples": cluster_texts[:3]
                })

            return themes

        except Exception:
            return []

    def _calculate_engagement(self, comments: List[Dict]) -> Dict:
        """Calculate engagement metrics for comments."""
        total_likes = sum(c.get("like_count", 0) for c in comments)
        total_replies = sum(c.get("reply_count", 0) for c in comments)

        return {
            "total_engagement": total_likes +
            total_replies,
            "avg_likes_per_comment": total_likes /
            len(comments) if comments else 0,
            "avg_replies_per_comment": total_replies /
            len(comments) if comments else 0,
            "engagement_rate": (
                total_likes +
                total_replies) /
            len(comments) if comments else 0}

    def _identify_influencers(self, comments: List[Dict]) -> List[Dict]:
        """Identify key influencers in comments."""
        user_stats = {}

        for comment in comments:
            user = comment.get("username") or comment.get("author")
            if not user:
                continue

            if user not in user_stats:
                user_stats[user] = {
                    "comment_count": 0,
                    "total_likes": 0,
                    "total_replies": 0
                }

            stats = user_stats[user]
            stats["comment_count"] += 1
            stats["total_likes"] += comment.get("like_count", 0)
            stats["total_replies"] += comment.get("reply_count", 0)

        # Calculate influence scores
        influencers = []
        for user, stats in user_stats.items():
            if stats["comment_count"] >= 3:  # Minimum threshold
                influence_score = (
                    stats["total_likes"] +
                    stats["total_replies"] * 2
                ) / stats["comment_count"]

                influencers.append({
                    "username": user,
                    "influence_score": influence_score,
                    "stats": stats
                })

        return sorted(
            influencers,
            key=lambda x: x["influence_score"],
            reverse=True
        )[:10]  # Top 10 influencers

    def _calculate_spam_score(self, comment: Dict) -> float:
        """Calculate spam probability score."""
        indicators = []

        # Check for spam indicators
        text = comment["text"].lower()

        # Common spam phrases
        spam_phrases = ["click here", "buy now", "earn money", "win prize"]
        phrase_matches = sum(1 for phrase in spam_phrases if phrase in text)
        indicators.append(phrase_matches / len(spam_phrases))

        # Excessive URLs
        url_count = text.count("http")
        indicators.append(min(url_count / 2, 1.0))

        # Excessive capitalization
        caps_ratio = sum(1 for c in text if c.isupper()) / \
            len(text) if text else 0
        indicators.append(1.0 if caps_ratio > 0.5 else caps_ratio)

        # Account age/history if available
        if "author_age_days" in comment:
            age_score = 1.0 - min(comment["author_age_days"] / 30, 1.0)
            indicators.append(age_score)

        return sum(indicators) / len(indicators)

    def _get_spam_indicators(self, comment: Dict) -> List[str]:
        """Get specific spam indicators for a comment."""
        indicators = []
        text = comment["text"].lower()

        if any(
            phrase in text for phrase in [
                "click here",
                "buy now",
                "earn money"]):
            indicators.append("suspicious_phrases")

        if text.count("http") > 1:
            indicators.append("multiple_urls")

        caps_ratio = sum(1 for c in text if c.isupper()) / \
            len(text) if text else 0
        if caps_ratio > 0.5:
            indicators.append("excessive_caps")

        if "author_age_days" in comment and comment["author_age_days"] < 7:
            indicators.append("new_account")

        return indicators

    def _get_sentiment_distribution(
        self,
        sentiments: List[Dict]
    ) -> Dict[str, float]:
        """Calculate distribution of sentiments."""
        total = len(sentiments)
        if not total:
            return {"positive": 0, "neutral": 0, "negative": 0}

        counts = {
            "positive": sum(
                1 for s in sentiments if s["sentiment"] == "positive"), "neutral": sum(
                1 for s in sentiments if s["sentiment"] == "neutral"), "negative": sum(
                1 for s in sentiments if s["sentiment"] == "negative")}

        return {
            k: v / total for k, v in counts.items()
        }

    def _calculate_sentiment_trend(
        self,
        comments: List[Dict],
        sentiments: List[Dict]
    ) -> List[Dict]:
        """Calculate sentiment trend over time."""
        # Sort comments by timestamp
        comment_sentiments = sorted(
            zip(comments, sentiments),
            key=lambda x: x[0].get("timestamp", datetime.min)
        )

        # Group by day
        daily_sentiments = {}
        for comment, sentiment in comment_sentiments:
            date = comment.get("timestamp", datetime.min).date()
            if date not in daily_sentiments:
                daily_sentiments[date] = []
            daily_sentiments[date].append(sentiment["polarity"])

        # Calculate daily averages
        trend = [
            {
                "date": date.isoformat(),
                "average_sentiment": sum(sentiments) / len(sentiments)
            }
            for date, sentiments in daily_sentiments.items()
        ]

        return sorted(trend, key=lambda x: x["date"])

    def _get_top_terms(
        self,
        cluster_vector: np.ndarray,
        feature_names: List[str],
        top_n: int = 5
    ) -> List[str]:
        """Get top terms for a cluster based on TF-IDF scores."""
        # Get indices of top terms
        top_indices = np.argsort(cluster_vector.toarray()[0])[-top_n:]

        # Return terms
        return [feature_names[i] for i in top_indices]
