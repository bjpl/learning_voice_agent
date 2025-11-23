"""
Analysis Agent - NLP-Powered Conversation Analysis

SPECIFICATION:
- Input: Conversation text or exchanges
- Output: Structured analysis (entities, concepts, topics, sentiment, keywords)
- Constraints: < 500ms processing, high accuracy
- Intelligence: Pattern detection, relationship extraction

PSEUDOCODE:
1. Receive text from conversation
2. Extract named entities (people, places, organizations)
3. Identify key concepts using noun phrases
4. Classify topics and themes
5. Analyze sentiment and tone
6. Extract keywords by importance
7. Return structured analysis

ARCHITECTURE:
- NLP Pipeline: spacy for entity recognition
- Topic Modeling: TF-IDF + keyword extraction
- Sentiment: Rule-based + context-aware
- Caching: Frequent patterns cached for speed

REFINEMENT:
- Batch processing for multiple messages
- Incremental learning from user patterns
- Relationship graph between concepts
- Time-aware topic trends

CODE:
"""
from typing import List, Dict, Any, Set, Tuple
from collections import Counter, defaultdict
from datetime import datetime
import re
import asyncio

from app.agents.base import BaseAgent, AgentMessage, MessageType
from app.logger import get_logger

# Optional: spacy for advanced NLP (will use fallback if not available)
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None


class AnalysisAgent(BaseAgent):
    """
    Extracts concepts, entities, and patterns from conversations

    PATTERN: NLP-powered analysis with entity recognition
    WHY: Enables intelligent organization and retrieval

    Features:
    - Named Entity Recognition (NER)
    - Concept extraction from noun phrases
    - Topic classification
    - Sentiment analysis
    - Keyword extraction
    - Relationship detection

    Example:
        agent = AnalysisAgent()
        message = AgentMessage(
            message_type="analyze_text",
            sender="conversation_handler",
            content={"text": "I'm learning about machine learning and neural networks"}
        )
        result = await agent.process(message)
        # result.content contains: entities, concepts, topics, sentiment, keywords
    """

    def __init__(self, agent_id: str = None):
        super().__init__(agent_id)

        # Load NLP model if available
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.info("spacy_model_loaded", model="en_core_web_sm")
            except OSError:
                self.logger.warning(
                    "spacy_model_not_found",
                    message="Run: python -m spacy download en_core_web_sm"
                )
        else:
            self.logger.warning("spacy_not_available", message="Install: pip install spacy")

        # Common stop words for filtering
        self.stop_words = self._load_stop_words()

        # Topic keywords for classification
        self.topic_keywords = self._load_topic_keywords()

        # Sentiment lexicon
        self.sentiment_words = self._load_sentiment_lexicon()

    def _load_stop_words(self) -> Set[str]:
        """Load common stop words to filter"""
        return {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
            'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
            'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
            'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once'
        }

    def _load_topic_keywords(self) -> Dict[str, List[str]]:
        """Load topic classification keywords"""
        return {
            "technology": [
                "programming", "software", "code", "algorithm", "data", "computer",
                "ai", "machine learning", "neural network", "api", "database", "web",
                "app", "development", "tech", "digital", "system", "network"
            ],
            "science": [
                "research", "study", "experiment", "theory", "hypothesis", "analysis",
                "biology", "chemistry", "physics", "mathematics", "scientific",
                "discovery", "evidence", "method"
            ],
            "learning": [
                "study", "learn", "education", "course", "tutorial", "lesson",
                "practice", "skill", "knowledge", "understand", "concept", "idea",
                "teach", "training", "improvement"
            ],
            "business": [
                "company", "market", "sales", "revenue", "customer", "product",
                "strategy", "business", "management", "startup", "enterprise",
                "finance", "growth", "team"
            ],
            "personal": [
                "life", "personal", "feel", "think", "experience", "goal", "plan",
                "habit", "routine", "wellness", "health", "mindset", "growth"
            ]
        }

    def _load_sentiment_lexicon(self) -> Dict[str, float]:
        """Load sentiment words with polarity scores"""
        return {
            # Positive words
            "good": 0.7, "great": 0.9, "excellent": 1.0, "amazing": 1.0,
            "love": 0.8, "like": 0.5, "enjoy": 0.6, "happy": 0.8,
            "excited": 0.9, "wonderful": 0.9, "fantastic": 1.0,
            "interesting": 0.6, "useful": 0.5, "helpful": 0.6,

            # Negative words
            "bad": -0.7, "terrible": -1.0, "awful": -1.0, "hate": -0.9,
            "dislike": -0.5, "sad": -0.6, "difficult": -0.4, "hard": -0.3,
            "frustrating": -0.7, "confused": -0.5, "boring": -0.6,
            "unhelpful": -0.6, "useless": -0.7,

            # Neutral/context-dependent
            "okay": 0.1, "fine": 0.2, "alright": 0.2
        }

    async def process(self, message: AgentMessage) -> AgentMessage:
        """
        Process message and extract structured information

        Supported content action types:
        - analyze_text: Analyze single text
        - analyze_conversation: Analyze full conversation
        - extract_relationships: Find concept relationships
        """
        action = message.content.get("action", "analyze_text")

        if action == "analyze_text":
            analysis = await self._analyze_text(message.content.get("text", ""))
        elif action == "analyze_conversation":
            analysis = await self._analyze_conversation(
                message.content.get("exchanges", [])
            )
        elif action == "extract_relationships":
            analysis = await self._extract_relationships(
                message.content.get("text", "")
            )
        else:
            analysis = {"error": f"Unknown action: {action}"}

        return AgentMessage(
            sender=self.agent_id,
            recipient=message.sender,
            message_type=MessageType.ANALYSIS_COMPLETE,
            content=analysis,
            correlation_id=message.message_id
        )

    async def _analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze single text and extract all features

        Returns:
            Dictionary with entities, concepts, topics, sentiment, keywords
        """
        if not text or not text.strip():
            return self._empty_analysis()

        # Run analysis tasks concurrently
        entities_task = self._extract_entities(text)
        concepts_task = self._extract_concepts(text)
        topics_task = self._classify_topics(text)
        sentiment_task = self._analyze_sentiment(text)
        keywords_task = self._extract_keywords(text)

        # Await all tasks
        entities, concepts, topics, sentiment, keywords = await asyncio.gather(
            entities_task,
            concepts_task,
            topics_task,
            sentiment_task,
            keywords_task
        )

        return {
            "entities": entities,
            "concepts": concepts,
            "topics": topics,
            "sentiment": sentiment,
            "keywords": keywords,
            "analyzed_at": datetime.utcnow().isoformat(),
            "text_length": len(text),
            "word_count": len(text.split())
        }

    async def _analyze_conversation(
        self,
        exchanges: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Analyze full conversation history

        Args:
            exchanges: List of {user: str, agent: str} dictionaries

        Returns:
            Aggregated analysis across conversation
        """
        if not exchanges:
            return self._empty_analysis()

        # Combine all user messages
        all_text = " ".join([ex.get("user", "") for ex in exchanges])

        # Get base analysis
        analysis = await self._analyze_text(all_text)

        # Add conversation-specific metrics
        analysis["exchange_count"] = len(exchanges)
        analysis["conversation_flow"] = await self._analyze_conversation_flow(exchanges)

        return analysis

    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities using spacy or fallback

        Returns:
            List of {text, label, start, end} dictionaries
        """
        if self.nlp:
            # Use spacy NER
            doc = self.nlp(text)
            return [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                }
                for ent in doc.ents
            ]
        else:
            # Fallback: Simple capitalized word detection
            return await self._extract_entities_fallback(text)

    async def _extract_entities_fallback(self, text: str) -> List[Dict[str, Any]]:
        """Fallback entity extraction without spacy"""
        entities = []

        # Find capitalized words (potential proper nouns)
        words = text.split()
        for i, word in enumerate(words):
            # Remove punctuation
            clean_word = re.sub(r'[^\w\s]', '', word)

            if clean_word and clean_word[0].isupper() and len(clean_word) > 1:
                # Skip if it's the start of a sentence
                if i == 0 or words[i-1].endswith(('.', '!', '?')):
                    continue

                entities.append({
                    "text": clean_word,
                    "label": "UNKNOWN",
                    "start": text.find(word),
                    "end": text.find(word) + len(word)
                })

        return entities

    async def _extract_concepts(self, text: str) -> List[str]:
        """
        Extract key concepts using noun phrases

        Returns:
            List of concept strings
        """
        if self.nlp:
            doc = self.nlp(text)
            # Get noun chunks and filter stop words
            concepts = [
                chunk.text.lower()
                for chunk in doc.noun_chunks
                if chunk.text.lower() not in self.stop_words
            ]
            return list(set(concepts))  # Unique concepts
        else:
            # Fallback: Extract multi-word phrases
            return await self._extract_concepts_fallback(text)

    async def _extract_concepts_fallback(self, text: str) -> List[str]:
        """Fallback concept extraction without spacy"""
        # Simple approach: Find noun-like words
        words = text.lower().split()

        # Filter meaningful words (> 4 chars, not stop words)
        concepts = [
            word
            for word in words
            if len(word) > 4 and word not in self.stop_words
        ]

        # Count frequency and return top concepts
        concept_freq = Counter(concepts)
        return [concept for concept, _ in concept_freq.most_common(10)]

    async def _classify_topics(self, text: str) -> List[Dict[str, Any]]:
        """
        Classify text into topics based on keyword matching

        Returns:
            List of {topic, confidence, matched_keywords}
        """
        text_lower = text.lower()
        topic_scores = []

        for topic, keywords in self.topic_keywords.items():
            matched_keywords = [kw for kw in keywords if kw in text_lower]

            if matched_keywords:
                # Calculate confidence based on number of matches
                confidence = min(len(matched_keywords) / 5.0, 1.0)

                topic_scores.append({
                    "topic": topic,
                    "confidence": round(confidence, 2),
                    "matched_keywords": matched_keywords[:3]  # Top 3
                })

        # Sort by confidence
        topic_scores.sort(key=lambda x: x["confidence"], reverse=True)

        return topic_scores[:3]  # Top 3 topics

    async def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using lexicon-based approach

        Returns:
            Dictionary with polarity, label, and contributing words
        """
        words = text.lower().split()
        sentiment_scores = []
        contributing_words = []

        for word in words:
            # Remove punctuation
            clean_word = re.sub(r'[^\w\s]', '', word)

            if clean_word in self.sentiment_words:
                score = self.sentiment_words[clean_word]
                sentiment_scores.append(score)
                contributing_words.append((clean_word, score))

        if not sentiment_scores:
            return {
                "polarity": 0.0,
                "label": "neutral",
                "confidence": 0.5,
                "contributing_words": []
            }

        # Calculate average polarity
        avg_polarity = sum(sentiment_scores) / len(sentiment_scores)

        # Determine label
        if avg_polarity > 0.3:
            label = "positive"
        elif avg_polarity < -0.3:
            label = "negative"
        else:
            label = "neutral"

        return {
            "polarity": round(avg_polarity, 2),
            "label": label,
            "confidence": round(min(abs(avg_polarity), 1.0), 2),
            "contributing_words": contributing_words[:5]
        }

    async def _extract_keywords(self, text: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Extract keywords by frequency and importance

        Args:
            text: Input text
            top_n: Number of top keywords to return

        Returns:
            List of {keyword, frequency, importance}
        """
        words = text.lower().split()

        # Filter words: remove stop words, short words
        filtered_words = [
            re.sub(r'[^\w\s]', '', word)
            for word in words
            if len(word) > 4 and word.lower() not in self.stop_words
        ]

        # Count frequencies
        word_freq = Counter(filtered_words)

        # Calculate importance (frequency * length)
        keywords = []
        for word, freq in word_freq.most_common(top_n):
            importance = freq * (len(word) / 10.0)  # Normalize by length
            keywords.append({
                "keyword": word,
                "frequency": freq,
                "importance": round(importance, 2)
            })

        return keywords

    async def _extract_relationships(self, text: str) -> Dict[str, Any]:
        """
        Extract relationships between concepts

        Returns:
            Dictionary with concept pairs and relationship types
        """
        concepts = await self._extract_concepts(text)

        # Find co-occurrence patterns
        relationships = []

        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                # Check if concepts appear near each other
                if concept1 in text and concept2 in text:
                    # Simple proximity check
                    idx1 = text.find(concept1)
                    idx2 = text.find(concept2)

                    if abs(idx1 - idx2) < 100:  # Within 100 chars
                        relationships.append({
                            "concept1": concept1,
                            "concept2": concept2,
                            "relationship": "co-occurs",
                            "strength": 0.7
                        })

        return {
            "relationships": relationships[:10],  # Top 10
            "concept_count": len(concepts)
        }

    async def _analyze_conversation_flow(
        self,
        exchanges: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Analyze conversation flow and patterns

        Returns:
            Flow metrics and patterns
        """
        if not exchanges:
            return {}

        # Track topic shifts
        topics_over_time = []

        for exchange in exchanges:
            user_text = exchange.get("user", "")
            topics = await self._classify_topics(user_text)
            if topics:
                topics_over_time.append(topics[0]["topic"])

        # Count topic transitions
        topic_changes = sum(
            1 for i in range(1, len(topics_over_time))
            if topics_over_time[i] != topics_over_time[i-1]
        )

        return {
            "topic_progression": topics_over_time,
            "topic_changes": topic_changes,
            "focus_score": 1.0 - (topic_changes / max(len(exchanges), 1))
        }

    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure"""
        return {
            "entities": [],
            "concepts": [],
            "topics": [],
            "sentiment": {
                "polarity": 0.0,
                "label": "neutral",
                "confidence": 0.0,
                "contributing_words": []
            },
            "keywords": [],
            "analyzed_at": datetime.utcnow().isoformat()
        }
