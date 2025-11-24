"""
Multi-Agent System for Learning Voice Agent
PATTERN: Agent orchestration with message passing
WHY: Modular, scalable agent architecture for Phase 2
"""

from app.agents.base import (
    BaseAgent,
    AgentMessage,
    MessageType,
    MessageRole,
    AgentMetadata,
)
from app.agents.research_agent import ResearchAgent
from app.agents.analysis_agent import AnalysisAgent
from app.agents.synthesis_agent import SynthesisAgent
from app.agents.conversation_agent import ConversationAgent, conversation_agent_v2
from app.agents.tools import tool_registry, ToolRegistry

__all__ = [
    "BaseAgent",
    "AgentMessage",
    "MessageType",
    "MessageRole",
    "AgentMetadata",
    "ResearchAgent",
    "AnalysisAgent",
    "SynthesisAgent",
    "ConversationAgent",
    "conversation_agent_v2",
    "tool_registry",
    "ToolRegistry",
]
