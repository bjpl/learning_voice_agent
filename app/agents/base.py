"""
Base Agent Class for Multi-Agent System
PATTERN: Message-passing agent with async processing
WHY: Foundation for all specialized agents in Phase 2+
SPEC: Agents communicate via structured messages, maintain state, and coordinate via queues
"""

import uuid
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

from app.logger import get_logger


# Additional enums and classes for ConversationAgent compatibility
class MessageRole(Enum):
    """Message role identifiers for conversation context"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class AgentMetadata:
    """
    Metadata for agent operations (used by ConversationAgent)
    PATTERN: Structured context propagation
    WHY: Track performance, debugging, and analytics
    """
    agent_name: str
    model: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    processing_time_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    tool_calls: int = 0
    error: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return {
            "agent_name": self.agent_name,
            "model": self.model,
            "timestamp": self.timestamp.isoformat(),
            "processing_time_ms": self.processing_time_ms,
            "tokens_used": self.tokens_used,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tool_calls": self.tool_calls,
            "error": self.error,
            "session_id": self.session_id,
            "request_id": self.request_id,
        }


class MessageType(Enum):
    """Standard message types for agent communication"""
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    STATUS = "status"
    RESEARCH_COMPLETE = "research_complete"
    ANALYSIS_COMPLETE = "analysis_complete"
    CONVERSATION_COMPLETE = "conversation_complete"


@dataclass
class AgentMessage:
    """
    SPECIFICATION: Structured message for agent communication
    PATTERN: Immutable message with metadata
    WHY: Type-safe, traceable agent interactions
    """
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    recipient: str = ""
    message_type: MessageType = MessageType.REQUEST
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None  # For tracking related messages
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize message to dictionary"""
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "recipient": self.recipient,
            "message_type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Deserialize message from dictionary"""
        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            sender=data.get("sender", ""),
            recipient=data.get("recipient", ""),
            message_type=MessageType(data.get("message_type", "request")),
            content=data.get("content", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else data.get("timestamp", datetime.utcnow()),
            correlation_id=data.get("correlation_id"),
            metadata=data.get("metadata", {}),
        )


class BaseAgent(ABC):
    """
    SPECIFICATION: Abstract base class for all agents
    PATTERN: Actor model with async message processing
    WHY: Enables scalable, concurrent agent orchestration

    Each agent:
    - Has a unique ID and type
    - Processes messages asynchronously
    - Maintains internal state
    - Can send/receive messages to/from other agents
    - Tracks metrics for monitoring
    """

    def __init__(self, agent_id: Optional[str] = None, agent_type: Optional[str] = None):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.agent_type = agent_type or self.__class__.__name__
        self.logger = get_logger(f"agent.{self.agent_type}", agent_id=self.agent_id)

        # Message queue for async processing
        self.inbox: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self.outbox: asyncio.Queue[AgentMessage] = asyncio.Queue()

        # State management
        self.state: Dict[str, Any] = {}
        self.is_running = False

        # Metrics
        self.metrics = {
            "messages_received": 0,
            "messages_sent": 0,
            "errors": 0,
            "processing_times": [],
            "last_active": None,
        }

        # Message handlers for different types
        self.handlers: Dict[MessageType, Callable] = {}

        self.logger.info("agent_initialized", agent_type=self.agent_type)

    @abstractmethod
    async def process(self, message: AgentMessage) -> AgentMessage:
        """
        Process an incoming message and return a response

        SPECIFICATION: Core agent logic goes here
        PATTERN: Template method - subclasses implement specific behavior
        WHY: Forces consistent interface while allowing customization

        Args:
            message: Incoming message to process

        Returns:
            Response message
        """
        pass

    async def send_message(
        self,
        recipient: str,
        message_type: MessageType,
        content: Dict[str, Any],
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentMessage:
        """
        Send a message to another agent

        PATTERN: Async message passing
        WHY: Non-blocking communication between agents
        """
        message = AgentMessage(
            sender=self.agent_id,
            recipient=recipient,
            message_type=message_type,
            content=content,
            correlation_id=correlation_id,
            metadata=metadata or {},
        )

        await self.outbox.put(message)
        self.metrics["messages_sent"] += 1

        self.logger.debug(
            "message_sent",
            recipient=recipient,
            message_type=message_type.value,
            message_id=message.message_id,
        )

        return message

    async def receive_message(self, message: AgentMessage) -> None:
        """
        Receive a message and queue it for processing

        PATTERN: Async queue-based inbox
        WHY: Decouples message receipt from processing
        """
        await self.inbox.put(message)
        self.metrics["messages_received"] += 1

        self.logger.debug(
            "message_received",
            sender=message.sender,
            message_type=message.message_type.value,
            message_id=message.message_id,
        )

    async def run(self) -> None:
        """
        Main agent loop - processes messages from inbox

        PATTERN: Event loop with graceful shutdown
        WHY: Continuous message processing until stopped
        """
        self.is_running = True
        self.logger.info("agent_started")

        try:
            while self.is_running:
                try:
                    # Wait for messages with timeout to check is_running periodically
                    message = await asyncio.wait_for(self.inbox.get(), timeout=1.0)

                    start_time = datetime.utcnow()

                    try:
                        # Process the message
                        response = await self.process(message)

                        # Send response if generated
                        if response and response.recipient:
                            await self.outbox.put(response)

                        # Update metrics
                        processing_time = (datetime.utcnow() - start_time).total_seconds()
                        self.metrics["processing_times"].append(processing_time)
                        self.metrics["last_active"] = datetime.utcnow()

                        self.logger.debug(
                            "message_processed",
                            message_id=message.message_id,
                            processing_time_ms=processing_time * 1000,
                        )

                    except Exception as e:
                        self.metrics["errors"] += 1
                        self.logger.error(
                            "message_processing_error",
                            message_id=message.message_id,
                            error=str(e),
                            error_type=type(e).__name__,
                            exc_info=True,
                        )

                        # Send error response
                        error_response = AgentMessage(
                            sender=self.agent_id,
                            recipient=message.sender,
                            message_type=MessageType.ERROR,
                            content={
                                "error": str(e),
                                "error_type": type(e).__name__,
                                "original_message_id": message.message_id,
                            },
                            correlation_id=message.message_id,
                        )
                        await self.outbox.put(error_response)

                except asyncio.TimeoutError:
                    # No message received, continue loop
                    continue

        finally:
            self.is_running = False
            self.logger.info("agent_stopped")

    async def stop(self) -> None:
        """
        Stop the agent gracefully

        PATTERN: Graceful shutdown
        WHY: Clean resource cleanup and state persistence
        """
        self.logger.info("agent_stopping")
        self.is_running = False

        # Wait for inbox to be empty (with timeout)
        try:
            await asyncio.wait_for(self.inbox.join(), timeout=5.0)
        except asyncio.TimeoutError:
            self.logger.warning("agent_stop_timeout", pending_messages=self.inbox.qsize())

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get agent performance metrics

        PATTERN: Observability metrics
        WHY: Monitor agent health and performance
        """
        avg_processing_time = (
            sum(self.metrics["processing_times"]) / len(self.metrics["processing_times"])
            if self.metrics["processing_times"]
            else 0
        )

        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "is_running": self.is_running,
            "messages_received": self.metrics["messages_received"],
            "messages_sent": self.metrics["messages_sent"],
            "errors": self.metrics["errors"],
            "avg_processing_time_ms": avg_processing_time * 1000,
            "pending_inbox": self.inbox.qsize(),
            "pending_outbox": self.outbox.qsize(),
            "last_active": self.metrics["last_active"].isoformat() if self.metrics["last_active"] else None,
        }

    def register_handler(self, message_type: MessageType, handler: Callable) -> None:
        """
        Register a handler for a specific message type

        PATTERN: Strategy pattern for message handling
        WHY: Extensible message processing
        """
        self.handlers[message_type] = handler
        self.logger.debug("handler_registered", message_type=message_type.value)


# ============================================================================
# Phase 2: Enhanced Agent Framework Classes
# ============================================================================
# These classes support the new AgentOrchestrator and advanced coordination
# SPARC Phase: Architecture - State management for orchestration


class AgentStatus(str, Enum):
    """
    SPARC: Specification - Agent status enumeration for orchestration

    Lifecycle states for coordinated agents:
    - IDLE: Agent initialized but not processing
    - INITIALIZING: Agent setting up resources
    - WORKING: Agent actively processing tasks
    - WAITING: Agent waiting for resources/responses
    - ERROR: Agent encountered an error
    - SHUTTING_DOWN: Agent cleaning up resources
    - TERMINATED: Agent has shut down
    """
    IDLE = "idle"
    INITIALIZING = "initializing"
    WORKING = "working"
    WAITING = "waiting"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    TERMINATED = "terminated"


@dataclass
class AgentState:
    """
    SPARC: Architecture - Enhanced agent state management for orchestration

    Tracks the complete state of an agent in the orchestration framework:
    - Identity and type information
    - Current status and task
    - Execution context
    - Communication history
    - Performance metrics

    Attributes:
        agent_id: Unique identifier for this agent
        agent_type: Type/class of agent (e.g., 'DialogueAgent')
        status: Current agent status (from AgentStatus enum)
        current_task: Optional description of current task
        context: Shared context/memory dictionary
        history: Message history (limited to recent messages)
        metrics: Performance and usage metrics
        created_at: Agent creation timestamp
        updated_at: Last state update timestamp
    """
    agent_id: str
    agent_type: str
    status: AgentStatus = AgentStatus.IDLE
    current_task: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    history: List[AgentMessage] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def update_status(self, status: AgentStatus, task: Optional[str] = None) -> None:
        """
        SPARC: Refinement - Update agent status

        Thread-safe status update with automatic timestamp tracking.
        """
        self.status = status
        if task is not None:
            self.current_task = task
        self.updated_at = datetime.utcnow()

        # Track status changes in metrics
        status_key = f"status_change_{status.value}"
        self.metrics[status_key] = self.metrics.get(status_key, 0) + 1

    def add_message(self, message: AgentMessage, max_history: int = 100) -> None:
        """
        SPARC: Refinement - Add message to history

        Maintains bounded message history to prevent memory bloat.

        Args:
            message: Message to add
            max_history: Maximum messages to retain (default: 100)
        """
        self.history.append(message)

        # Trim history if it exceeds max
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]

        self.updated_at = datetime.utcnow()

    def increment_metric(self, metric_name: str, value: int = 1) -> None:
        """Increment a metric counter"""
        self.metrics[metric_name] = self.metrics.get(metric_name, 0) + value
        self.updated_at = datetime.utcnow()
