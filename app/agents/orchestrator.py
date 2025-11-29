"""
Agent Orchestration Engine

SPARC Phase: Architecture & Implementation
Purpose: Coordinate multiple agents, route messages, manage execution

This module provides:
- AgentRegistry: Central registry for agent discovery
- MessageRouter: Route messages between agents
- ExecutionContext: Manage execution state and context
- AgentOrchestrator: Main orchestration engine
"""

from typing import Dict, List, Any, Optional, Set, Callable, Awaitable
from datetime import datetime
from enum import Enum
import asyncio
import logging
from collections import defaultdict
import uuid

from app.agents.base import (
    BaseAgent,
    AgentMessage,
    AgentState,
    AgentStatus,
    MessageType
)
from app.agents.protocols import AgentProtocol

logger = logging.getLogger(__name__)


class RoutingStrategy(str, Enum):
    """
    SPARC: Specification - Message routing strategies

    Defines how messages are routed to agents:
    - DIRECT: Route to specific agent by ID
    - BROADCAST: Send to all agents
    - ROUND_ROBIN: Distribute evenly across agents
    - LOAD_BALANCED: Route based on agent load
    - CAPABILITY: Route based on agent capabilities
    """
    DIRECT = "direct"
    BROADCAST = "broadcast"
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    CAPABILITY = "capability"


class ExecutionMode(str, Enum):
    """
    SPARC: Specification - Execution modes

    - SEQUENTIAL: Execute agents one at a time
    - PARALLEL: Execute agents concurrently
    - PIPELINE: Execute in pipeline (output of one -> input of next)
    """
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"


class AgentRegistry:
    """
    SPARC: Architecture - Agent registry and discovery

    Maintains registry of all agents in the system.
    Supports:
    - Agent registration/unregistration
    - Discovery by type, capability, or status
    - Health monitoring
    - Capability indexing
    """

    def __init__(self):
        """Initialize agent registry"""
        self._agents: Dict[str, AgentProtocol] = {}
        self._types: Dict[str, Set[str]] = defaultdict(set)
        self._capabilities: Dict[str, Set[str]] = defaultdict(set)
        self.logger = logging.getLogger("registry")

    def register(
        self,
        agent: AgentProtocol,
        capabilities: Optional[List[str]] = None
    ) -> None:
        """
        SPARC: Implementation - Register an agent

        Args:
            agent: Agent to register
            capabilities: Optional list of agent capabilities
        """
        agent_id = agent.agent_id
        agent_type = agent.state.agent_type

        # Store agent
        self._agents[agent_id] = agent

        # Index by type
        self._types[agent_type].add(agent_id)

        # Index by capabilities
        if capabilities:
            for capability in capabilities:
                self._capabilities[capability].add(agent_id)

        self.logger.info(
            f"Registered agent {agent_id} of type {agent_type} "
            f"with capabilities: {capabilities or []}"
        )

    def unregister(self, agent_id: str) -> None:
        """
        SPARC: Implementation - Unregister an agent

        Args:
            agent_id: Agent ID to unregister
        """
        if agent_id not in self._agents:
            return

        agent = self._agents[agent_id]
        agent_type = agent.state.agent_type

        # Remove from agents
        del self._agents[agent_id]

        # Remove from type index
        self._types[agent_type].discard(agent_id)
        if not self._types[agent_type]:
            del self._types[agent_type]

        # Remove from capability indexes
        for capability_agents in self._capabilities.values():
            capability_agents.discard(agent_id)

        self.logger.info(f"Unregistered agent {agent_id}")

    def get(self, agent_id: str) -> Optional[AgentProtocol]:
        """Get agent by ID"""
        return self._agents.get(agent_id)

    def get_by_type(self, agent_type: str) -> List[AgentProtocol]:
        """Get all agents of a specific type"""
        agent_ids = self._types.get(agent_type, set())
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]

    def get_by_capability(self, capability: str) -> List[AgentProtocol]:
        """Get all agents with a specific capability"""
        agent_ids = self._capabilities.get(capability, set())
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]

    def get_all(self) -> List[AgentProtocol]:
        """Get all registered agents"""
        return list(self._agents.values())

    def get_healthy(self) -> List[AgentProtocol]:
        """Get all healthy (non-error) agents"""
        return [
            agent for agent in self._agents.values()
            if agent.state.status not in {AgentStatus.ERROR, AgentStatus.TERMINATED}
        ]

    def count(self) -> int:
        """Get total number of registered agents"""
        return len(self._agents)

    def exists(self, agent_id: str) -> bool:
        """Check if agent exists"""
        return agent_id in self._agents


class MessageRouter:
    """
    SPARC: Architecture - Message routing engine

    Routes messages between agents based on routing strategies.
    Supports:
    - Multiple routing strategies
    - Message queuing
    - Priority routing
    - Message filtering
    """

    def __init__(self, registry: AgentRegistry):
        """
        Initialize message router

        Args:
            registry: Agent registry for routing decisions
        """
        self.registry = registry
        self.logger = logging.getLogger("router")

        # Message queue per agent
        self._queues: Dict[str, asyncio.Queue] = {}

        # Round-robin counters
        self._round_robin_counters: Dict[str, int] = defaultdict(int)

        # Routing filters
        self._filters: List[Callable[[AgentMessage], bool]] = []

    async def route(
        self,
        message: AgentMessage,
        strategy: RoutingStrategy = RoutingStrategy.DIRECT
    ) -> List[str]:
        """
        SPARC: Implementation - Route message to target agent(s)

        Args:
            message: Message to route
            strategy: Routing strategy to use

        Returns:
            List of agent IDs message was routed to
        """
        # Apply filters
        if not self._check_filters(message):
            self.logger.debug(f"Message {message.id} filtered out")
            return []

        # Determine target agents based on strategy
        if strategy == RoutingStrategy.DIRECT:
            targets = [message.recipient]
        elif strategy == RoutingStrategy.BROADCAST:
            targets = [a.agent_id for a in self.registry.get_healthy()]
        elif strategy == RoutingStrategy.ROUND_ROBIN:
            targets = self._round_robin_route(message)
        elif strategy == RoutingStrategy.LOAD_BALANCED:
            targets = self._load_balanced_route(message)
        elif strategy == RoutingStrategy.CAPABILITY:
            targets = self._capability_route(message)
        else:
            self.logger.error(f"Unknown routing strategy: {strategy}")
            return []

        # Route to each target
        routed = []
        for target_id in targets:
            agent = self.registry.get(target_id)
            if agent:
                await self._deliver_message(agent, message)
                routed.append(target_id)
            else:
                self.logger.warning(f"Target agent not found: {target_id}")

        self.logger.debug(
            f"Routed message {message.id} to {len(routed)} agents using {strategy.value}"
        )

        return routed

    async def _deliver_message(
        self,
        agent: AgentProtocol,
        message: AgentMessage
    ) -> None:
        """
        SPARC: Refinement - Deliver message to agent

        Args:
            agent: Target agent
            message: Message to deliver
        """
        try:
            # Handle message asynchronously
            asyncio.create_task(agent.handle_message(message))
        except Exception as e:
            self.logger.error(
                f"Error delivering message to {agent.agent_id}: {e}",
                exc_info=True
            )

    def _round_robin_route(self, message: AgentMessage) -> List[str]:
        """Round-robin routing implementation"""
        # Get all healthy agents
        agents = self.registry.get_healthy()
        if not agents:
            return []

        # Get next agent in round-robin
        counter = self._round_robin_counters["default"]
        agent = agents[counter % len(agents)]
        self._round_robin_counters["default"] += 1

        return [agent.agent_id]

    def _load_balanced_route(self, message: AgentMessage) -> List[str]:
        """
        SPARC: Refinement - Load-balanced routing

        Routes to agent with lowest current load (fewest messages in queue).
        """
        agents = self.registry.get_healthy()
        if not agents:
            return []

        # Find agent with minimum queue size
        min_load = float('inf')
        selected_agent = None

        for agent in agents:
            queue_size = self._queues.get(agent.agent_id, asyncio.Queue()).qsize()
            if queue_size < min_load:
                min_load = queue_size
                selected_agent = agent

        return [selected_agent.agent_id] if selected_agent else []

    def _capability_route(self, message: AgentMessage) -> List[str]:
        """
        SPARC: Refinement - Capability-based routing

        Routes based on required capability in message metadata.
        """
        # Check for required capability in metadata
        required_capability = message.metadata.get("required_capability") if message.metadata else None

        if not required_capability:
            self.logger.warning("Capability routing requires 'required_capability' in metadata")
            return []

        # Get agents with capability
        agents = self.registry.get_by_capability(required_capability)

        if not agents:
            self.logger.warning(f"No agents found with capability: {required_capability}")
            return []

        # Use first healthy agent (could be enhanced with load balancing)
        for agent in agents:
            if agent.state.status not in {AgentStatus.ERROR, AgentStatus.TERMINATED}:
                return [agent.agent_id]

        return []

    def add_filter(self, filter_func: Callable[[AgentMessage], bool]) -> None:
        """
        Add message filter

        Args:
            filter_func: Function that returns True if message should be routed
        """
        self._filters.append(filter_func)

    def _check_filters(self, message: AgentMessage) -> bool:
        """Check if message passes all filters"""
        return all(f(message) for f in self._filters)


class ExecutionContext:
    """
    SPARC: Architecture - Execution context manager

    Manages execution state and context for multi-agent workflows.
    Tracks:
    - Execution state
    - Shared context/memory
    - Task dependencies
    - Execution history
    """

    def __init__(self, context_id: Optional[str] = None):
        """
        Initialize execution context

        Args:
            context_id: Optional custom context ID
        """
        self.context_id = context_id or str(uuid.uuid4())
        self.created_at = datetime.utcnow()

        # Shared state
        self.state: Dict[str, Any] = {}

        # Execution history
        self.history: List[Dict[str, Any]] = []

        # Task tracking
        self.tasks: Dict[str, Any] = {}

        # Metrics
        self.metrics: Dict[str, Any] = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "messages_sent": 0,
            "execution_time_ms": 0
        }

        self.logger = logging.getLogger(f"context.{self.context_id[:8]}")

    def set(self, key: str, value: Any) -> None:
        """Set context value"""
        self.state[key] = value
        self._record_event("state_set", {"key": key})

    def get(self, key: str, default: Any = None) -> Any:
        """Get context value"""
        return self.state.get(key, default)

    def update(self, updates: Dict[str, Any]) -> None:
        """Batch update context"""
        self.state.update(updates)
        self._record_event("state_update", {"keys": list(updates.keys())})

    def delete(self, key: str) -> None:
        """Delete context value"""
        if key in self.state:
            del self.state[key]
            self._record_event("state_delete", {"key": key})

    def _record_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Record event in history"""
        self.history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "data": data
        })

    def get_snapshot(self) -> Dict[str, Any]:
        """Get complete context snapshot"""
        return {
            "context_id": self.context_id,
            "created_at": self.created_at.isoformat(),
            "state": self.state,
            "tasks": self.tasks,
            "metrics": self.metrics,
            "history_length": len(self.history)
        }


class AgentOrchestrator:
    """
    SPARC: Architecture - Main orchestration engine

    Coordinates multiple agents to accomplish complex tasks.
    Provides:
    - Agent lifecycle management
    - Message routing and coordination
    - Parallel and sequential execution
    - Error handling and recovery
    - Metrics and monitoring
    """

    def __init__(self):
        """Initialize orchestrator"""
        self.registry = AgentRegistry()
        self.router = MessageRouter(self.registry)
        self.contexts: Dict[str, ExecutionContext] = {}
        self.logger = logging.getLogger("orchestrator")

        # Orchestrator state
        self._running = False
        self._tasks: Set[asyncio.Task] = set()

        # Agent dictionary for direct access (used by tests and for simple routing)
        self.agents: Dict[str, Any] = {}
        self._initialized = False

    async def register_agent(
        self,
        agent: AgentProtocol,
        capabilities: Optional[List[str]] = None
    ) -> None:
        """
        SPARC: Implementation - Register and initialize agent

        Args:
            agent: Agent to register
            capabilities: Optional agent capabilities
        """
        # Register with registry
        self.registry.register(agent, capabilities)

        # Initialize agent if not already initialized
        if not hasattr(agent, '_initialized') or not agent._initialized:
            await agent.initialize()

        self.logger.info(f"Registered and initialized agent: {agent.agent_id}")

    async def unregister_agent(self, agent_id: str) -> None:
        """
        SPARC: Implementation - Unregister and cleanup agent

        Args:
            agent_id: Agent ID to unregister
        """
        agent = self.registry.get(agent_id)
        if agent:
            # Cleanup agent
            await agent.cleanup()

            # Unregister
            self.registry.unregister(agent_id)

            self.logger.info(f"Unregistered agent: {agent_id}")

    async def send_message(
        self,
        message: AgentMessage,
        strategy: RoutingStrategy = RoutingStrategy.DIRECT
    ) -> List[str]:
        """
        SPARC: Implementation - Send message through router

        Args:
            message: Message to send
            strategy: Routing strategy

        Returns:
            List of agent IDs message was delivered to
        """
        return await self.router.route(message, strategy)

    async def execute_sequential(
        self,
        agents: List[AgentProtocol],
        initial_message: AgentMessage,
        context: Optional[ExecutionContext] = None
    ) -> List[AgentMessage]:
        """
        SPARC: Implementation - Execute agents sequentially

        Each agent processes output from previous agent.

        Args:
            agents: List of agents to execute in order
            initial_message: Starting message
            context: Optional execution context

        Returns:
            List of response messages from each agent
        """
        ctx = context or ExecutionContext()
        responses = []
        current_message = initial_message

        for agent in agents:
            try:
                response = await agent.handle_message(current_message)
                responses.append(response)

                # Use response as input for next agent
                if response:
                    current_message = response

                ctx.metrics["tasks_completed"] += 1

            except Exception as e:
                self.logger.error(f"Error in sequential execution: {e}", exc_info=True)
                ctx.metrics["tasks_failed"] += 1
                break

        return responses

    async def execute_parallel(
        self,
        agents: List[AgentProtocol],
        message: AgentMessage,
        context: Optional[ExecutionContext] = None
    ) -> List[AgentMessage]:
        """
        SPARC: Implementation - Execute agents in parallel

        All agents process same message concurrently.

        Args:
            agents: List of agents to execute
            message: Message to send to all agents
            context: Optional execution context

        Returns:
            List of response messages from all agents
        """
        ctx = context or ExecutionContext()

        # Create tasks for all agents
        tasks = [agent.handle_message(message) for agent in agents]

        # Execute in parallel
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Track metrics
        for response in responses:
            if isinstance(response, Exception):
                ctx.metrics["tasks_failed"] += 1
            else:
                ctx.metrics["tasks_completed"] += 1

        # Filter out exceptions
        return [r for r in responses if not isinstance(r, Exception)]

    async def execute_pipeline(
        self,
        agents: List[AgentProtocol],
        initial_message: AgentMessage,
        context: Optional[ExecutionContext] = None
    ) -> AgentMessage:
        """
        SPARC: Implementation - Execute agents as pipeline

        Similar to sequential but passes data through pipeline.

        Args:
            agents: Pipeline stages (agents)
            initial_message: Initial input
            context: Optional execution context

        Returns:
            Final pipeline output
        """
        responses = await self.execute_sequential(agents, initial_message, context)
        return responses[-1] if responses else None

    def create_context(self, context_id: Optional[str] = None) -> ExecutionContext:
        """
        Create new execution context

        Args:
            context_id: Optional custom context ID

        Returns:
            New execution context
        """
        ctx = ExecutionContext(context_id)
        self.contexts[ctx.context_id] = ctx
        return ctx

    def get_context(self, context_id: str) -> Optional[ExecutionContext]:
        """Get execution context by ID"""
        return self.contexts.get(context_id)

    async def start(self) -> None:
        """Start orchestrator"""
        self._running = True
        self.logger.info("Orchestrator started")

    async def stop(self) -> None:
        """Stop orchestrator and cleanup all agents"""
        self._running = False

        # Cleanup all agents
        for agent in self.registry.get_all():
            await self.unregister_agent(agent.agent_id)

        # Cancel pending tasks
        for task in self._tasks:
            task.cancel()

        self.logger.info("Orchestrator stopped")

    async def process(self, message: AgentMessage) -> AgentMessage:
        """
        Process a message by routing to appropriate agent(s)

        SPARC: Implementation - Main message processing entry point

        Args:
            message: Incoming message to process

        Returns:
            Response message from agent(s)
        """
        from datetime import datetime
        import time

        start_time = time.time()
        agents_used = []

        try:
            # Determine which agent(s) to use based on message type and content
            content = message.content if isinstance(message.content, dict) else {"text": str(message.content)}
            text = content.get("text", "")
            enable_research = content.get("enable_research", False)
            enable_analysis = content.get("enable_analysis", False)

            # Route based on message type
            if message.message_type == MessageType.RESEARCH_REQUEST:
                agent = self.agents.get("research")
                if agent:
                    agents_used.append("research")
                    response = await agent.process(message)
                    response.metadata = response.metadata or {}
                    if isinstance(response.metadata, dict):
                        response.metadata["agents_used"] = agents_used
                    return response

            # Default: use conversation agent
            agent = self.agents.get("conversation")
            if agent:
                agents_used.append("conversation")

                # Check if we should also use research
                if enable_research and "research" in self.agents:
                    agents_used.append("research")

                # Check if we should also use analysis
                if enable_analysis and "analysis" in self.agents:
                    agents_used.append("analysis")

                response = await agent.process(message)

                # Ensure response has proper structure
                processing_time = (time.time() - start_time) * 1000

                # Build metadata
                metadata = {}
                if hasattr(response, 'metadata') and response.metadata:
                    if hasattr(response.metadata, 'to_dict'):
                        metadata = response.metadata.to_dict()
                    elif isinstance(response.metadata, dict):
                        metadata = response.metadata.copy()

                metadata["agents_used"] = agents_used
                metadata["processing_time_ms"] = processing_time

                # Create properly formatted response
                return AgentMessage(
                    sender="orchestrator",
                    recipient=message.sender,
                    message_type=MessageType.CONVERSATION_RESPONSE,
                    content={"text": response.content if isinstance(response.content, str) else response.content.get("text", str(response.content))},
                    correlation_id=message.message_id,
                    metadata=metadata,
                )

            # No suitable agent found
            return AgentMessage(
                sender="orchestrator",
                recipient=message.sender,
                message_type=MessageType.AGENT_ERROR,
                content={"error": "No suitable agent found", "text": "I'm sorry, I couldn't process your request."},
                correlation_id=message.message_id,
                metadata={"agents_used": agents_used},
            )

        except Exception as e:
            self.logger.error(f"Error processing message: {e}", exc_info=True)
            return AgentMessage(
                sender="orchestrator",
                recipient=message.sender,
                message_type=MessageType.CONVERSATION_RESPONSE,
                content={"text": "I encountered an error. Please try again.", "error": str(e)},
                correlation_id=message.message_id,
                metadata={"agents_used": agents_used, "error": str(e)},
            )

    def register_agent(self, agent, name: Optional[str] = None) -> None:
        """
        Simple agent registration (sync version for tests)

        Args:
            agent: Agent to register
            name: Optional name/key for the agent
        """
        agent_name = name or getattr(agent, 'agent_id', None) or agent.__class__.__name__.lower()
        self.agents[agent_name] = agent
        self.logger.info(f"Registered agent: {agent_name}")

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            "total_agents": self.registry.count(),
            "healthy_agents": len(self.registry.get_healthy()),
            "active_contexts": len(self.contexts),
            "running": self._running,
            "agents": list(self.agents.keys()),
        }
