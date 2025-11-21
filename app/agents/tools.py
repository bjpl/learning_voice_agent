"""
Tool Implementations for ConversationAgent

SPECIFICATION:
- Search tool: Query knowledge base and past conversations
- Calculator tool: Mathematical operations
- Time/Date tool: Current time and date information
- Memory tool: Store and retrieve facts

PATTERN: Strategy pattern for tool implementations
WHY: Easy to add new tools without modifying agent code
"""
import math
import operator
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from app.logger import conversation_logger


@dataclass
class ToolDefinition:
    """
    Tool definition for Claude API
    PATTERN: Schema definition for function calling
    """
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Optional[Callable] = None


class ToolRegistry:
    """
    Central registry for all available tools
    PATTERN: Registry pattern for dynamic tool management
    WHY: Easy to add/remove tools at runtime
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register all default tools"""
        self.register(self._create_search_tool())
        self.register(self._create_calculator_tool())
        self.register(self._create_datetime_tool())
        self.register(self._create_memory_tool())

    def register(self, tool: ToolDefinition):
        """Register a new tool"""
        self._tools[tool.name] = tool
        conversation_logger.debug("tool_registered", tool_name=tool.name)

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get tool by name"""
        return self._tools.get(name)

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Get all tool schemas for Claude API"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in self._tools.values()
        ]

    def _create_search_tool(self) -> ToolDefinition:
        """
        Search tool for querying knowledge base
        PATTERN: Semantic search over past conversations
        """
        return ToolDefinition(
            name="search_knowledge",
            description="Search through past conversations and learned knowledge to find relevant information. Use this when the user asks about something they mentioned before or to provide context-aware responses.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant past conversations or knowledge"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            },
            handler=self._handle_search
        )

    def _create_calculator_tool(self) -> ToolDefinition:
        """
        Calculator tool for mathematical operations
        PATTERN: Safe expression evaluation
        """
        return ToolDefinition(
            name="calculate",
            description="Perform mathematical calculations including basic arithmetic, trigonometry, and common functions. Supports expressions like '2 + 2', 'sqrt(16)', 'sin(pi/2)', etc.",
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'pi * 2')"
                    }
                },
                "required": ["expression"]
            },
            handler=self._handle_calculate
        )

    def _create_datetime_tool(self) -> ToolDefinition:
        """
        Date/time tool for current time information
        PATTERN: System time query
        """
        return ToolDefinition(
            name="get_datetime",
            description="Get current date, time, day of week, or timezone information. Use when user asks 'what time is it', 'what day is it', etc.",
            input_schema={
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "description": "Format to return: 'full' (complete datetime), 'date' (date only), 'time' (time only), 'day' (day of week)",
                        "enum": ["full", "date", "time", "day"],
                        "default": "full"
                    },
                    "timezone": {
                        "type": "string",
                        "description": "Timezone name (e.g., 'UTC', 'America/New_York'). Defaults to UTC.",
                        "default": "UTC"
                    }
                },
                "required": []
            },
            handler=self._handle_datetime
        )

    def _create_memory_tool(self) -> ToolDefinition:
        """
        Memory tool for storing/retrieving facts
        PATTERN: Key-value storage with context
        """
        return ToolDefinition(
            name="memory_store",
            description="Store or retrieve important facts about the user or conversation. Use to remember preferences, key information, or important context for future conversations.",
            input_schema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action to perform: 'store' to save information or 'retrieve' to get stored information",
                        "enum": ["store", "retrieve"]
                    },
                    "key": {
                        "type": "string",
                        "description": "Identifier for the memory (e.g., 'user_name', 'favorite_topic')"
                    },
                    "value": {
                        "type": "string",
                        "description": "Value to store (only required for 'store' action)"
                    }
                },
                "required": ["action", "key"]
            },
            handler=self._handle_memory
        )

    # Tool Handler Implementations

    async def _handle_search(self, query: str, limit: int = 5, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Handle search tool execution

        TODO: Integrate with actual knowledge base/vector search
        For now, returns mock results based on context
        """
        conversation_logger.info("search_tool_executed", query=query, limit=limit)

        # Mock implementation - in production, would query vector database
        results = []

        if context and "conversation_history" in context:
            history = context["conversation_history"]
            # Simple text search through history
            for exchange in history[-limit:]:
                user_text = exchange.get("user", "")
                agent_text = exchange.get("agent", "")

                if query.lower() in user_text.lower() or query.lower() in agent_text.lower():
                    results.append({
                        "user": user_text,
                        "agent": agent_text,
                        "relevance": 0.8  # Mock relevance score
                    })

        return {
            "success": True,
            "results": results,
            "total_found": len(results),
            "query": query
        }

    async def _handle_calculate(self, expression: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Handle calculator tool execution

        SECURITY: Uses safe evaluation with limited namespace
        """
        conversation_logger.info("calculator_tool_executed", expression=expression)

        # Safe math namespace
        safe_namespace = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
            "e": math.e,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "ceil": math.ceil,
            "floor": math.floor,
        }

        try:
            # Clean and validate expression
            expression = expression.strip()

            # Evaluate with restricted namespace
            result = eval(expression, {"__builtins__": {}}, safe_namespace)

            return {
                "success": True,
                "result": result,
                "expression": expression
            }
        except Exception as e:
            conversation_logger.warning("calculator_error", expression=expression, error=str(e))
            return {
                "success": False,
                "error": f"Could not evaluate expression: {str(e)}",
                "expression": expression
            }

    async def _handle_datetime(self, format: str = "full", timezone: str = "UTC", context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Handle datetime tool execution
        """
        conversation_logger.info("datetime_tool_executed", format=format, timezone=timezone)

        try:
            now = datetime.now(timezone.utc)

            if format == "date":
                result = now.strftime("%Y-%m-%d")
            elif format == "time":
                result = now.strftime("%H:%M:%S %Z")
            elif format == "day":
                result = now.strftime("%A")
            else:  # full
                result = now.strftime("%Y-%m-%d %H:%M:%S %Z")

            return {
                "success": True,
                "datetime": result,
                "format": format,
                "timezone": timezone,
                "iso": now.isoformat()
            }
        except Exception as e:
            conversation_logger.warning("datetime_error", error=str(e))
            return {
                "success": False,
                "error": f"Could not get datetime: {str(e)}"
            }

    async def _handle_memory(self, action: str, key: str, value: Optional[str] = None, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Handle memory tool execution

        TODO: Integrate with actual persistent storage
        Currently uses in-memory context storage
        """
        conversation_logger.info("memory_tool_executed", action=action, key=key)

        if not context:
            context = {}

        if "memory_store" not in context:
            context["memory_store"] = {}

        memory_store = context["memory_store"]

        if action == "store":
            if value is None:
                return {
                    "success": False,
                    "error": "Value required for store action"
                }

            memory_store[key] = {
                "value": value,
                "timestamp": datetime.utcnow().isoformat()
            }

            return {
                "success": True,
                "action": "stored",
                "key": key,
                "value": value
            }

        elif action == "retrieve":
            if key in memory_store:
                stored = memory_store[key]
                return {
                    "success": True,
                    "action": "retrieved",
                    "key": key,
                    "value": stored["value"],
                    "timestamp": stored["timestamp"]
                }
            else:
                return {
                    "success": False,
                    "error": f"No memory found for key: {key}"
                }

        return {
            "success": False,
            "error": f"Unknown action: {action}"
        }


# Global tool registry instance
tool_registry = ToolRegistry()
