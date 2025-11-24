"""
Tests for SynthesisAgent

PATTERN: Comprehensive unit and integration tests
WHY: Ensure synthesis quality and performance
"""
import pytest
from datetime import datetime, timedelta

from app.agents.synthesis_agent import SynthesisAgent
from app.agents.base import AgentMessage, MessageType


@pytest.fixture
def synthesis_agent():
    """Create SynthesisAgent instance for testing"""
    return SynthesisAgent(agent_id="test_synthesis_agent")


@pytest.fixture
def sample_analysis():
    """Sample analysis data for testing"""
    return {
        "entities": [
            {"text": "Python", "label": "TECH"},
            {"text": "TensorFlow", "label": "TECH"}
        ],
        "concepts": [
            "machine learning",
            "neural networks",
            "deep learning",
            "algorithms"
        ],
        "topics": [
            {
                "topic": "technology",
                "confidence": 0.9,
                "matched_keywords": ["machine learning", "neural networks"]
            },
            {
                "topic": "learning",
                "confidence": 0.7,
                "matched_keywords": ["study", "learn"]
            }
        ],
        "sentiment": {
            "polarity": 0.6,
            "label": "positive",
            "confidence": 0.8,
            "contributing_words": [("great", 0.9), ("love", 0.8)]
        },
        "keywords": [
            {"keyword": "learning", "frequency": 5, "importance": 8.5},
            {"keyword": "neural", "frequency": 3, "importance": 6.0}
        ]
    }


@pytest.fixture
def sample_exchanges():
    """Sample conversation exchanges"""
    return [
        {
            "user": "I'm learning about machine learning",
            "agent": "That's wonderful! What interests you most?"
        },
        {
            "user": "Neural networks and deep learning algorithms",
            "agent": "Great topics! Those are very powerful."
        },
        {
            "user": "Can you explain backpropagation?",
            "agent": "Sure! It's how neural networks learn..."
        },
        {
            "user": "That makes sense. Thank you!",
            "agent": "You're welcome! Keep exploring!"
        }
    ]


@pytest.mark.asyncio
class TestSynthesisAgent:
    """Test suite for SynthesisAgent"""

    async def test_initialization(self, synthesis_agent):
        """Test agent initializes correctly"""
        assert synthesis_agent.agent_id == "test_synthesis_agent"
        assert synthesis_agent.metrics["messages_processed"] == 0
        assert len(synthesis_agent.topic_graph) > 0
        assert "default_ease_factor" in synthesis_agent.sm2_defaults

    async def test_generate_insights(self, synthesis_agent, sample_analysis, sample_exchanges):
        """Test insight generation"""
        message = AgentMessage(
            sender="test",
            recipient=synthesis_agent.agent_id,
            message_type=MessageType.REQUEST,
            content={
                "action": "generate_insights",
                "analysis": sample_analysis,
                "history": sample_exchanges
            }
        )

        result = await synthesis_agent.process(message)

        assert result.message_type == MessageType.RESPONSE
        assert "insights" in result.content
        assert "insight_count" in result.content
        assert len(result.content["insights"]) > 0

        # Check insight structure
        for insight in result.content["insights"]:
            assert "type" in insight
            assert "insight" in insight
            assert "details" in insight
            assert "actionable" in insight

    async def test_learning_focus_insight(self, synthesis_agent, sample_analysis):
        """Test learning focus insight generation"""
        insight = await synthesis_agent._analyze_learning_focus(sample_analysis)

        assert insight is not None
        assert insight["type"] == "learning_focus"
        assert "topic" in insight["details"]
        assert "confidence" in insight["details"]
        assert "actionable" in insight

    async def test_concept_connections_insight(self, synthesis_agent, sample_analysis):
        """Test concept connection insight"""
        insight = await synthesis_agent._analyze_concept_connections(sample_analysis)

        assert insight is not None
        assert insight["type"] == "concept_connections"
        assert "concepts" in insight["details"]
        assert len(insight["details"]["concepts"]) > 0

    async def test_learning_patterns_insight(self, synthesis_agent, sample_exchanges):
        """Test learning pattern analysis"""
        insight = await synthesis_agent._analyze_learning_patterns(sample_exchanges)

        assert insight is not None
        assert insight["type"] == "learning_pattern"
        assert insight["details"]["pattern"] in ["inquiry-driven", "balanced", "reflective"]
        assert 0.0 <= insight["details"]["question_ratio"] <= 1.0

    async def test_sentiment_trends_insight(
        self,
        synthesis_agent,
        sample_exchanges,
        sample_analysis
    ):
        """Test sentiment trend analysis"""
        insight = await synthesis_agent._analyze_sentiment_trends(
            sample_exchanges,
            sample_analysis
        )

        assert insight is not None
        assert insight["type"] == "sentiment_trend"
        assert insight["details"]["sentiment"] in ["positive", "negative", "neutral"]

    async def test_create_summary(self, synthesis_agent, sample_analysis, sample_exchanges):
        """Test summary creation"""
        message = AgentMessage(
            message_type="create_summary",
            sender="test",
            content={
                "exchanges": sample_exchanges,
                "analysis": sample_analysis
            }
        )

        result = await synthesis_agent.handle_message(message)

        assert result.message_type == "synthesis_complete"
        assert "summary" in result.content
        assert "key_points" in result.content
        assert "topics_covered" in result.content
        assert result.content["summary_type"] == "comprehensive"
        assert result.content["exchange_count"] == len(sample_exchanges)

    async def test_empty_summary(self, synthesis_agent):
        """Test summary with no exchanges"""
        message = AgentMessage(
            message_type="create_summary",
            sender="test",
            content={"exchanges": []}
        )

        result = await synthesis_agent.handle_message(message)

        assert result.content["summary_type"] == "empty"
        assert len(result.content["key_points"]) == 0

    async def test_recommend_topics(self, synthesis_agent, sample_analysis):
        """Test topic recommendations"""
        message = AgentMessage(
            message_type="recommend_topics",
            sender="test",
            content={
                "topics": sample_analysis["topics"],
                "concepts": sample_analysis["concepts"]
            }
        )

        result = await synthesis_agent.handle_message(message)

        assert result.message_type == "synthesis_complete"
        assert "recommendations" in result.content
        assert "recommendation_count" in result.content

        # Check recommendation structure
        for rec in result.content["recommendations"]:
            assert "topic" in rec
            assert "reason" in rec
            assert "confidence" in rec
            assert 0.0 <= rec["confidence"] <= 1.0

    async def test_topic_graph_recommendations(self, synthesis_agent):
        """Test recommendations from topic graph"""
        # Machine learning should recommend related topics
        topics = [{"topic": "machine learning", "confidence": 0.9}]

        result = await synthesis_agent._recommend_topics({
            "topics": topics,
            "concepts": []
        })

        assert len(result["recommendations"]) > 0

        # Should include related ML topics
        recommended_topics = [r["topic"] for r in result["recommendations"]]
        assert any("deep learning" in t or "neural" in t for t in recommended_topics)

    async def test_create_schedule(self, synthesis_agent, sample_analysis):
        """Test spaced repetition schedule creation"""
        message = AgentMessage(
            message_type="create_schedule",
            sender="test",
            content={
                "concepts": sample_analysis["concepts"]
            }
        )

        result = await synthesis_agent.handle_message(message)

        assert result.message_type == "synthesis_complete"
        assert "schedule" in result.content
        assert "algorithm" in result.content
        assert result.content["algorithm"] == "SM-2"

        # Check schedule items
        for item in result.content["schedule"]:
            assert "concept" in item
            assert "repetition" in item
            assert "review_date" in item
            assert "days_from_now" in item
            assert "priority" in item
            assert item["priority"] in ["high", "medium", "low"]

    async def test_sm2_algorithm(self, synthesis_agent):
        """Test SM-2 interval calculation"""
        intervals = synthesis_agent._calculate_sm2_intervals(
            quality=4,
            repetition=0
        )

        assert len(intervals) == 5
        assert intervals[0] == 1  # First review: 1 day
        assert intervals[1] == 6  # Second review: 6 days

        # Intervals should increase
        for i in range(1, len(intervals)):
            assert intervals[i] >= intervals[i-1]

    async def test_priority_calculation(self, synthesis_agent):
        """Test review priority calculation"""
        # Early concepts, early repetition = high priority
        assert synthesis_agent._calculate_priority(0, 1) == "high"
        assert synthesis_agent._calculate_priority(2, 1) == "high"

        # Mid-range concepts = medium priority
        assert synthesis_agent._calculate_priority(5, 2) == "medium"

        # Later concepts, later repetitions = low priority
        assert synthesis_agent._calculate_priority(8, 3) == "low"

    async def test_synthesize_all(self, synthesis_agent, sample_analysis, sample_exchanges):
        """Test comprehensive synthesis"""
        message = AgentMessage(
            message_type="synthesize_all",
            sender="test",
            content={
                "analysis": sample_analysis,
                "history": sample_exchanges,
                "exchanges": sample_exchanges,
                "topics": sample_analysis["topics"],
                "concepts": sample_analysis["concepts"]
            }
        )

        result = await synthesis_agent.handle_message(message)

        assert result.message_type == "synthesis_complete"
        assert "insights" in result.content
        assert "summary" in result.content
        assert "recommendations" in result.content
        assert "schedule" in result.content

    async def test_performance_benchmark(self, synthesis_agent, sample_analysis, sample_exchanges):
        """Test processing performance"""
        message = AgentMessage(
            message_type="synthesize_all",
            sender="test",
            content={
                "analysis": sample_analysis,
                "history": sample_exchanges,
                "exchanges": sample_exchanges,
                "topics": sample_analysis["topics"],
                "concepts": sample_analysis["concepts"]
            }
        )

        import time
        start = time.perf_counter()
        result = await synthesis_agent.handle_message(message)
        duration_ms = (time.perf_counter() - start) * 1000

        # Should process in < 800ms as per spec (excluding Claude API calls)
        # Note: This may be higher if Claude API is called
        assert duration_ms < 2000  # Allow extra time for API
        assert result.message_type == "synthesis_complete"

    async def test_metrics_tracking(self, synthesis_agent, sample_analysis):
        """Test metrics tracking"""
        initial_count = synthesis_agent.metrics["messages_processed"]

        message = AgentMessage(
            message_type="generate_insights",
            sender="test",
            content={"analysis": sample_analysis, "history": []}
        )

        await synthesis_agent.handle_message(message)

        assert synthesis_agent.metrics["messages_processed"] == initial_count + 1
        assert synthesis_agent.metrics["total_processing_time_ms"] > 0

        metrics = synthesis_agent.get_metrics()
        assert "avg_processing_time_ms" in metrics
        assert metrics["agent_type"] == "SynthesisAgent"

    async def test_error_handling(self, synthesis_agent):
        """Test error handling for invalid message type"""
        message = AgentMessage(
            message_type="invalid_type",
            sender="test",
            content={}
        )

        result = await synthesis_agent.handle_message(message)

        assert result.message_type == "synthesis_complete"
        assert "error" in result.content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
