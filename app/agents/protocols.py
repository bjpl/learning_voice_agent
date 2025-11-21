"""
Agent Communication Protocols

SPARC Phase: Architecture
Purpose: Define protocols and interfaces for agent interactions

This module provides:
- Protocol definitions for agent behavior
- Interface contracts for agent communication
- Tool use protocols
- State sharing protocols
"""

from typing import Protocol, Dict, List, Any, Optional, Callable, Awaitable
from datetime import datetime
from app.agents.base import AgentMessage, AgentState, MessageType, AgentStatus


class AgentProtocol(Protocol):
    """
    SPARC: Specification - Agent interface protocol

    Defines the contract that all agents must fulfill.
    Uses Python's Protocol for structural subtyping (duck typing with type safety).

    This allows agents to be interchangeable if they implement these methods,
    regardless of inheritance hierarchy.
    """

    agent_id: str
    state: AgentState

    async def process(self, message: AgentMessage) -> AgentMessage:
        """Process incoming message and return response"""
        ...

    async def initialize(self) -> None:
        """Initialize agent resources"""
        ...

    async def cleanup(self) -> None:
        """Cleanup agent resources"""
        ...

    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle message with error handling wrapper"""
        ...

    async def send_message(
        self,
        recipient: str,
        message_type: MessageType,
        content: Dict[str, Any],
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """Send message to another agent"""
        ...

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get current state snapshot"""
        ...


class MessageProtocol(Protocol):
    """
    SPARC: Specification - Message handling protocol

    Defines how agents handle different message types.
    Agents can implement specific handlers for each message type.
    """

    async def handle_request(self, message: AgentMessage) -> AgentMessage:
        """
        Handle REQUEST message

        Args:
            message: Request message

        Returns:
            Response message
        """
        ...

    async def handle_response(self, message: AgentMessage) -> None:
        """
        Handle RESPONSE message

        Args:
            message: Response message
        """
        ...

    async def handle_event(self, message: AgentMessage) -> None:
        """
        Handle EVENT message (fire-and-forget)

        Args:
            message: Event message
        """
        ...

    async def handle_command(self, message: AgentMessage) -> AgentMessage:
        """
        Handle COMMAND message

        Args:
            message: Command message

        Returns:
            Status message
        """
        ...

    async def handle_status(self, message: AgentMessage) -> None:
        """
        Handle STATUS message

        Args:
            message: Status update message
        """
        ...

    async def handle_error(self, message: AgentMessage) -> None:
        """
        Handle ERROR message

        Args:
            message: Error message
        """
        ...


class StateProtocol(Protocol):
    """
    SPARC: Specification - State management protocol

    Defines how agents manage and share state.
    Supports both local state and shared/distributed state.
    """

    async def get_state(self, key: str) -> Optional[Any]:
        """
        Get state value by key

        Args:
            key: State key

        Returns:
            State value or None
        """
        ...

    async def set_state(self, key: str, value: Any) -> None:
        """
        Set state value

        Args:
            key: State key
            value: State value
        """
        ...

    async def update_state(self, updates: Dict[str, Any]) -> None:
        """
        Batch update multiple state values

        Args:
            updates: Dictionary of key-value updates
        """
        ...

    async def delete_state(self, key: str) -> None:
        """
        Delete state value

        Args:
            key: State key to delete
        """
        ...

    async def get_all_state(self) -> Dict[str, Any]:
        """
        Get all state as dictionary

        Returns:
            Complete state dictionary
        """
        ...

    async def clear_state(self) -> None:
        """Clear all state"""
        ...

    async def share_state(self, agent_id: str, keys: List[str]) -> None:
        """
        Share specific state keys with another agent

        Args:
            agent_id: Target agent ID
            keys: List of state keys to share
        """
        ...


class ToolProtocol(Protocol):
    """
    SPARC: Specification - Tool use protocol

    Defines how agents interact with external tools and capabilities.
    Tools can be LLM functions, API calls, database queries, etc.
    """

    async def register_tool(
        self,
        tool_name: str,
        tool_func: Callable,
        description: str,
        parameters: Dict[str, Any]
    ) -> None:
        """
        Register a tool for agent use

        Args:
            tool_name: Unique tool identifier
            tool_func: Callable tool function
            description: Tool description for LLM
            parameters: Tool parameter schema
        """
        ...

    async def unregister_tool(self, tool_name: str) -> None:
        """
        Unregister a tool

        Args:
            tool_name: Tool to unregister
        """
        ...

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all registered tools

        Returns:
            List of tool definitions
        """
        ...

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> Any:
        """
        Execute a tool with parameters

        Args:
            tool_name: Tool to execute
            parameters: Tool parameters

        Returns:
            Tool execution result
        """
        ...

    async def validate_tool_call(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> bool:
        """
        Validate tool call before execution

        Args:
            tool_name: Tool name
            parameters: Tool parameters

        Returns:
            True if valid, False otherwise
        """
        ...


class CoordinationProtocol(Protocol):
    """
    SPARC: Architecture - Agent coordination protocol

    Defines how agents coordinate with each other for complex tasks.
    Supports patterns like:
    - Task delegation
    - Parallel execution
    - Sequential workflows
    - Consensus building
    """

    async def delegate_task(
        self,
        agent_id: str,
        task: Dict[str, Any]
    ) -> str:
        """
        Delegate task to another agent

        Args:
            agent_id: Target agent ID
            task: Task specification

        Returns:
            Task ID for tracking
        """
        ...

    async def request_collaboration(
        self,
        agent_ids: List[str],
        task: Dict[str, Any]
    ) -> str:
        """
        Request collaboration from multiple agents

        Args:
            agent_ids: List of agent IDs
            task: Collaborative task specification

        Returns:
            Collaboration session ID
        """
        ...

    async def broadcast_event(
        self,
        event_type: str,
        data: Dict[str, Any]
    ) -> None:
        """
        Broadcast event to all agents

        Args:
            event_type: Type of event
            data: Event data
        """
        ...

    async def subscribe_to_events(
        self,
        event_types: List[str],
        handler: Callable[[str, Dict[str, Any]], Awaitable[None]]
    ) -> None:
        """
        Subscribe to specific event types

        Args:
            event_types: List of event types to subscribe to
            handler: Event handler function
        """
        ...

    async def vote(
        self,
        proposal_id: str,
        vote: bool,
        reasoning: Optional[str] = None
    ) -> None:
        """
        Vote on a proposal (consensus building)

        Args:
            proposal_id: Proposal identifier
            vote: True for yes, False for no
            reasoning: Optional vote reasoning
        """
        ...

    async def wait_for_consensus(
        self,
        proposal_id: str,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Wait for consensus on a proposal

        Args:
            proposal_id: Proposal to wait for
            timeout: Maximum wait time in seconds

        Returns:
            Consensus result with votes
        """
        ...


class ErrorHandlingProtocol(Protocol):
    """
    SPARC: Architecture - Error handling protocol

    Defines standardized error handling across agents.
    Supports error recovery, retry logic, and error propagation.
    """

    async def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> None:
        """
        Handle an error

        Args:
            error: Exception that occurred
            context: Error context information
        """
        ...

    async def report_error(
        self,
        error: Exception,
        severity: str,
        context: Dict[str, Any]
    ) -> None:
        """
        Report error to monitoring system

        Args:
            error: Exception that occurred
            severity: Error severity (low, medium, high, critical)
            context: Error context
        """
        ...

    async def retry_operation(
        self,
        operation: Callable[[], Awaitable[Any]],
        max_retries: int = 3,
        backoff: float = 1.0
    ) -> Any:
        """
        Retry an operation with exponential backoff

        Args:
            operation: Async operation to retry
            max_retries: Maximum retry attempts
            backoff: Initial backoff time in seconds

        Returns:
            Operation result
        """
        ...

    async def recover_from_error(
        self,
        error: Exception,
        recovery_strategy: str
    ) -> bool:
        """
        Attempt to recover from an error

        Args:
            error: Exception to recover from
            recovery_strategy: Recovery strategy name

        Returns:
            True if recovery successful, False otherwise
        """
        ...


class MetricsProtocol(Protocol):
    """
    SPARC: Architecture - Metrics and observability protocol

    Defines how agents track and report metrics.
    Supports performance monitoring and debugging.
    """

    async def record_metric(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a metric value

        Args:
            metric_name: Metric name
            value: Metric value
            tags: Optional metric tags
        """
        ...

    async def increment_counter(
        self,
        counter_name: str,
        value: int = 1,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Increment a counter

        Args:
            counter_name: Counter name
            value: Increment value
            tags: Optional counter tags
        """
        ...

    async def record_latency(
        self,
        operation_name: str,
        latency_ms: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record operation latency

        Args:
            operation_name: Operation name
            latency_ms: Latency in milliseconds
            tags: Optional tags
        """
        ...

    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get all recorded metrics

        Returns:
            Dictionary of metrics
        """
        ...
