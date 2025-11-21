"""
Synthesis Agent - AI-Powered Insight Generation

SPECIFICATION:
- Input: Analyzed conversation data, patterns, concepts
- Output: Insights, summaries, recommendations, learning schedules
- Constraints: < 800ms processing, actionable outputs
- Intelligence: Pattern recognition, spaced repetition scheduling

PSEUDOCODE:
1. Receive analysis data or conversation history
2. Identify patterns and themes
3. Generate insights from patterns
4. Create comprehensive summaries
5. Recommend related topics
6. Generate spaced repetition schedule
7. Return structured synthesis

ARCHITECTURE:
- Pattern Recognition: Frequency, co-occurrence, temporal
- Summarization: Extractive + abstractive (Claude API)
- Recommendation: Content-based filtering
- Scheduling: SM-2 algorithm for spaced repetition

REFINEMENT:
- Cache common patterns
- Incremental learning from feedback
- Personalized recommendations
- Adaptive scheduling based on recall

CODE:
"""
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import asyncio
import math
import anthropic

from app.agents.base import BaseAgent, AgentMessage, MessageType
from app.config import settings
from app.logger import get_logger


class SynthesisAgent(BaseAgent):
    """
    Generates insights, summaries, and recommendations

    PATTERN: AI-powered synthesis with pattern recognition
    WHY: Transforms raw data into actionable knowledge

    Features:
    - Pattern detection in learning behavior
    - Insight generation from conversations
    - Comprehensive summarization
    - Related topic recommendations
    - Spaced repetition scheduling (SM-2 algorithm)
    - Learning path optimization

    Example:
        agent = SynthesisAgent()
        message = AgentMessage(
            message_type="generate_insights",
            sender="coordinator",
            content={
                "analysis": {...},  # From AnalysisAgent
                "history": [...]    # Conversation exchanges
            }
        )
        result = await agent.process(message)
        # result.content contains: insights, summary, recommendations, schedule
    """

    def __init__(self, agent_id: str = None):
        super().__init__(agent_id)

        # Initialize Claude client for intelligent summarization
        self.claude_client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key
        )

        # Topic knowledge graph (for recommendations)
        self.topic_graph = self._build_topic_graph()

        # Spaced repetition parameters (SM-2 algorithm)
        self.sm2_defaults = {
            "initial_interval": 1,      # 1 day
            "min_ease_factor": 1.3,
            "default_ease_factor": 2.5
        }

    def _build_topic_graph(self) -> Dict[str, List[str]]:
        """
        Build knowledge graph of related topics

        PATTERN: Topic relationship mapping
        WHY: Enable intelligent recommendations
        """
        return {
            "machine learning": [
                "deep learning", "neural networks", "data science",
                "statistics", "algorithms", "python programming"
            ],
            "deep learning": [
                "neural networks", "tensorflow", "pytorch", "computer vision",
                "nlp", "transformers", "machine learning"
            ],
            "python programming": [
                "data structures", "algorithms", "web development",
                "machine learning", "automation", "testing"
            ],
            "web development": [
                "javascript", "react", "api design", "databases",
                "devops", "security", "ux design"
            ],
            "data science": [
                "statistics", "machine learning", "data visualization",
                "sql", "python programming", "analytics"
            ],
            "algorithms": [
                "data structures", "complexity analysis", "dynamic programming",
                "graphs", "sorting", "searching"
            ],
            "databases": [
                "sql", "nosql", "data modeling", "indexing",
                "transactions", "optimization"
            ],
            "distributed systems": [
                "microservices", "consensus", "caching", "messaging",
                "cloud computing", "scalability"
            ]
        }

    async def process(self, message: AgentMessage) -> AgentMessage:
        """
        Process synthesis request

        Supported content action types:
        - generate_insights: Generate insights from analysis
        - create_summary: Create conversation summary
        - recommend_topics: Recommend related topics
        - create_schedule: Generate spaced repetition schedule
        - synthesize_all: Run all synthesis tasks
        """
        action = message.content.get("action", "synthesize_all")
        content = message.content

        if action == "generate_insights":
            result = await self._generate_insights(content)
        elif action == "create_summary":
            result = await self._create_summary(content)
        elif action == "recommend_topics":
            result = await self._recommend_topics(content)
        elif action == "create_schedule":
            result = await self._create_schedule(content)
        elif action == "synthesize_all":
            # Run all synthesis tasks
            result = await self._synthesize_all(content)
        else:
            result = {"error": f"Unknown action: {action}"}

        return AgentMessage(
            sender=self.agent_id,
            recipient=message.sender,
            message_type=MessageType.RESPONSE,
            content=result,
            correlation_id=message.message_id
        )

    async def _generate_insights(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate insights from analyzed conversation data

        Args:
            data: Dictionary with analysis results and conversation history

        Returns:
            Dictionary with insights, patterns, and observations
        """
        analysis = data.get("analysis", {})
        history = data.get("history", [])

        insights = []

        # Insight 1: Learning focus
        focus_insight = await self._analyze_learning_focus(analysis)
        if focus_insight:
            insights.append(focus_insight)

        # Insight 2: Concept connections
        connection_insight = await self._analyze_concept_connections(analysis)
        if connection_insight:
            insights.append(connection_insight)

        # Insight 3: Learning patterns
        pattern_insight = await self._analyze_learning_patterns(history)
        if pattern_insight:
            insights.append(pattern_insight)

        # Insight 4: Sentiment trends
        sentiment_insight = await self._analyze_sentiment_trends(history, analysis)
        if sentiment_insight:
            insights.append(sentiment_insight)

        return {
            "insights": insights,
            "insight_count": len(insights),
            "generated_at": datetime.utcnow().isoformat()
        }

    async def _analyze_learning_focus(
        self,
        analysis: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Identify primary learning focus"""
        topics = analysis.get("topics", [])

        if not topics:
            return None

        primary_topic = topics[0]

        return {
            "type": "learning_focus",
            "insight": f"Your primary focus is on {primary_topic['topic']}",
            "details": {
                "topic": primary_topic["topic"],
                "confidence": primary_topic["confidence"],
                "keywords": primary_topic.get("matched_keywords", [])
            },
            "actionable": f"Consider exploring advanced concepts in {primary_topic['topic']}"
        }

    async def _analyze_concept_connections(
        self,
        analysis: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Identify interesting concept connections"""
        concepts = analysis.get("concepts", [])

        if len(concepts) < 2:
            return None

        # Find concepts that appear together
        top_concepts = concepts[:5]

        return {
            "type": "concept_connections",
            "insight": f"You're connecting {len(top_concepts)} different concepts",
            "details": {
                "concepts": top_concepts,
                "connection_strength": "high" if len(top_concepts) > 3 else "medium"
            },
            "actionable": "Try creating a concept map to visualize these relationships"
        }

    async def _analyze_learning_patterns(
        self,
        history: List[Dict[str, str]]
    ) -> Optional[Dict[str, Any]]:
        """Analyze patterns in learning behavior"""
        if len(history) < 3:
            return None

        # Analyze question frequency
        question_count = sum(
            1 for ex in history
            if '?' in ex.get('user', '')
        )

        question_ratio = question_count / len(history)

        if question_ratio > 0.5:
            pattern_type = "inquiry-driven"
            insight_text = "You learn by asking questions - great for deep understanding"
        elif question_ratio > 0.2:
            pattern_type = "balanced"
            insight_text = "You balance exploration with declarative learning"
        else:
            pattern_type = "reflective"
            insight_text = "You learn by articulating and reflecting on ideas"

        return {
            "type": "learning_pattern",
            "insight": insight_text,
            "details": {
                "pattern": pattern_type,
                "question_ratio": round(question_ratio, 2),
                "exchange_count": len(history)
            },
            "actionable": "This pattern shows active engagement - keep it up!"
        }

    async def _analyze_sentiment_trends(
        self,
        history: List[Dict[str, str]],
        analysis: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Analyze sentiment trends in learning"""
        sentiment = analysis.get("sentiment", {})

        if not sentiment or sentiment.get("label") == "neutral":
            return None

        polarity = sentiment.get("polarity", 0)
        label = sentiment.get("label", "neutral")

        if label == "positive":
            insight_text = "You're showing enthusiasm in your learning - that's powerful!"
            actionable = "Channel this energy into consistent practice"
        else:
            insight_text = "You might be facing some challenges"
            actionable = "Break down complex topics into smaller, manageable pieces"

        return {
            "type": "sentiment_trend",
            "insight": insight_text,
            "details": {
                "sentiment": label,
                "polarity": polarity,
                "contributing_words": sentiment.get("contributing_words", [])
            },
            "actionable": actionable
        }

    async def _create_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create comprehensive conversation summary

        Args:
            data: Dictionary with exchanges and optional analysis

        Returns:
            Dictionary with summary text, key points, and metadata
        """
        exchanges = data.get("exchanges", [])
        analysis = data.get("analysis", {})

        if not exchanges:
            return {
                "summary": "No conversation to summarize",
                "key_points": [],
                "summary_type": "empty"
            }

        # Extract key information
        topics = analysis.get("topics", [])
        concepts = analysis.get("concepts", [])
        keywords = analysis.get("keywords", [])

        # Build summary components
        summary_parts = []

        # Topic summary
        if topics:
            topic_names = [t["topic"] for t in topics[:3]]
            summary_parts.append(
                f"You explored {', '.join(topic_names)}"
            )

        # Concept summary
        if concepts:
            summary_parts.append(
                f"discussing {len(concepts)} key concepts including {', '.join(concepts[:3])}"
            )

        # Generate key points
        key_points = []

        for i, exchange in enumerate(exchanges[:5], 1):
            user_text = exchange.get("user", "")
            if len(user_text) > 20:  # Substantial contribution
                key_points.append({
                    "point": user_text[:100] + ("..." if len(user_text) > 100 else ""),
                    "order": i
                })

        # Use Claude for intelligent summarization if available
        if len(exchanges) > 3 and settings.anthropic_api_key:
            ai_summary = await self._generate_ai_summary(exchanges)
            if ai_summary:
                summary_parts.append(ai_summary)

        return {
            "summary": ". ".join(summary_parts) if summary_parts else "Brief conversation",
            "key_points": key_points[:5],
            "topics_covered": [t["topic"] for t in topics],
            "concepts_discussed": concepts[:10],
            "exchange_count": len(exchanges),
            "summary_type": "comprehensive",
            "generated_at": datetime.utcnow().isoformat()
        }

    async def _generate_ai_summary(
        self,
        exchanges: List[Dict[str, str]]
    ) -> Optional[str]:
        """
        Generate AI-powered summary using Claude

        Args:
            exchanges: Conversation exchanges

        Returns:
            Summary string or None if failed
        """
        try:
            # Build conversation text
            conversation_text = "\n".join([
                f"User: {ex.get('user', '')}\nAgent: {ex.get('agent', '')}"
                for ex in exchanges
            ])

            # Call Claude for summarization
            message = await self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=150,
                temperature=0.3,
                messages=[{
                    "role": "user",
                    "content": f"Summarize the key learning points from this conversation in 2-3 sentences:\n\n{conversation_text}"
                }]
            )

            return message.content[0].text.strip()

        except Exception as e:
            self.logger.warning(
                "ai_summary_failed",
                error=str(e),
                error_type=type(e).__name__
            )
            return None

    async def _recommend_topics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recommend related topics to explore

        Args:
            data: Dictionary with current topics and concepts

        Returns:
            Dictionary with recommended topics and rationale
        """
        current_topics = data.get("topics", [])
        concepts = data.get("concepts", [])

        recommendations = []

        # Find related topics from knowledge graph
        for topic_data in current_topics:
            topic = topic_data.get("topic", "")

            if topic in self.topic_graph:
                related = self.topic_graph[topic]

                for related_topic in related[:3]:  # Top 3 per topic
                    recommendations.append({
                        "topic": related_topic,
                        "reason": f"Related to {topic}",
                        "confidence": 0.8,
                        "source": "knowledge_graph"
                    })

        # Deduplicate and rank
        seen = set()
        unique_recommendations = []

        for rec in recommendations:
            if rec["topic"] not in seen:
                seen.add(rec["topic"])
                unique_recommendations.append(rec)

        # Limit to top 5
        return {
            "recommendations": unique_recommendations[:5],
            "recommendation_count": len(unique_recommendations),
            "based_on": [t.get("topic") for t in current_topics],
            "generated_at": datetime.utcnow().isoformat()
        }

    async def _create_schedule(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create spaced repetition schedule using SM-2 algorithm

        Args:
            data: Dictionary with concepts and learning history

        Returns:
            Dictionary with review schedule
        """
        concepts = data.get("concepts", [])

        if not concepts:
            return {
                "schedule": [],
                "message": "No concepts to schedule"
            }

        schedule = []
        base_date = datetime.utcnow()

        for i, concept in enumerate(concepts[:10]):  # Limit to top 10
            # Calculate review intervals using SM-2
            intervals = self._calculate_sm2_intervals(
                quality=4,  # Assume good initial quality
                repetition=0
            )

            # Create review schedule
            for rep_num, interval_days in enumerate(intervals[:4], 1):
                review_date = base_date + timedelta(days=interval_days)

                schedule.append({
                    "concept": concept,
                    "repetition": rep_num,
                    "review_date": review_date.isoformat(),
                    "days_from_now": interval_days,
                    "priority": self._calculate_priority(i, rep_num)
                })

        # Sort by review date
        schedule.sort(key=lambda x: x["review_date"])

        return {
            "schedule": schedule[:20],  # Next 20 reviews
            "total_items": len(schedule),
            "algorithm": "SM-2",
            "generated_at": datetime.utcnow().isoformat()
        }

    def _calculate_sm2_intervals(
        self,
        quality: int,
        repetition: int,
        ease_factor: float = None
    ) -> List[int]:
        """
        Calculate spaced repetition intervals using SM-2 algorithm

        Args:
            quality: Response quality (0-5, 5 being perfect)
            repetition: Current repetition number
            ease_factor: Current ease factor (default 2.5)

        Returns:
            List of interval days for next reviews
        """
        if ease_factor is None:
            ease_factor = self.sm2_defaults["default_ease_factor"]

        intervals = []
        current_interval = self.sm2_defaults["initial_interval"]

        for i in range(5):  # Generate 5 intervals
            if i == 0:
                intervals.append(1)
            elif i == 1:
                intervals.append(6)
            else:
                current_interval = math.ceil(current_interval * ease_factor)
                intervals.append(current_interval)

            # Update ease factor based on quality
            ease_factor = max(
                self.sm2_defaults["min_ease_factor"],
                ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
            )

        return intervals

    def _calculate_priority(self, concept_index: int, repetition: int) -> str:
        """Calculate review priority"""
        if concept_index < 3 and repetition == 1:
            return "high"
        elif concept_index < 7 or repetition <= 2:
            return "medium"
        else:
            return "low"

    async def _synthesize_all(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run all synthesis tasks in parallel

        Args:
            data: Complete data bundle

        Returns:
            Combined synthesis results
        """
        # Run all synthesis tasks concurrently
        insights_task = self._generate_insights(data)
        summary_task = self._create_summary(data)
        recommendations_task = self._recommend_topics(data)
        schedule_task = self._create_schedule(data)

        insights, summary, recommendations, schedule = await asyncio.gather(
            insights_task,
            summary_task,
            recommendations_task,
            schedule_task,
            return_exceptions=True
        )

        return {
            "insights": insights if not isinstance(insights, Exception) else {},
            "summary": summary if not isinstance(summary, Exception) else {},
            "recommendations": recommendations if not isinstance(recommendations, Exception) else {},
            "schedule": schedule if not isinstance(schedule, Exception) else {},
            "generated_at": datetime.utcnow().isoformat()
        }
