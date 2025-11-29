"""
ResearchAgent with Tool Integration
SPECIFICATION: Multi-tool research agent for knowledge retrieval
PATTERN: Tool-augmented agent with async orchestration
WHY: Enable agents to access external knowledge sources (web, papers, databases, code execution)
"""

import asyncio
import json
from defusedxml import ElementTree as ET
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import httpx

from app.agents.base import BaseAgent, AgentMessage, MessageType
from app.logger import api_logger
from app.database import db
from app.resilience import with_retry


class ToolExecutionError(Exception):
    """Raised when a tool execution fails"""
    pass


class ResearchAgent(BaseAgent):
    """
    SPECIFICATION: Research agent with tool use capabilities
    PATTERN: Tool-augmented information retrieval with caching
    WHY: Enables agents to access external knowledge sources

    Capabilities:
    - Web search (Tavily API or DuckDuckGo fallback)
    - Wikipedia search
    - ArXiv paper search
    - Code execution in sandboxes
    - Knowledge base querying (SQLite FTS5)

    Features:
    - Async operations throughout
    - 30s timeout per tool
    - Result caching
    - Rate limiting
    - Comprehensive error handling
    - Metrics tracking
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
        enable_code_execution: bool = False,
    ):
        final_agent_id = agent_id or "research_agent"
        super().__init__(agent_id=final_agent_id, agent_type="ResearchAgent")

        # HTTP client with timeout
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "LearningVoiceAgent-ResearchBot/1.0"}
        )

        # Tool configuration
        self.tavily_api_key = tavily_api_key
        self.enable_code_execution = enable_code_execution

        # Tool registry
        self.tools: Dict[str, Callable] = {
            "web_search": self._web_search,
            "wikipedia": self._wikipedia_search,
            "arxiv": self._arxiv_search,
            "code_execute": self._execute_code,
            "knowledge_base": self._query_knowledge_base,
        }

        # Cache for tool results (simple in-memory cache)
        self.cache: Dict[str, tuple[Any, datetime]] = {}
        self.cache_ttl = timedelta(minutes=15)

        # Rate limiting (simple token bucket)
        self.rate_limits: Dict[str, List[datetime]] = {}
        self.rate_limit_window = timedelta(minutes=1)
        self.rate_limit_max_calls = 10

        # Tool metrics
        self.tool_metrics: Dict[str, Dict[str, Any]] = {
            tool: {"calls": 0, "errors": 0, "total_time": 0.0}
            for tool in self.tools.keys()
        }

        api_logger.info(
            "research_agent_initialized",
            agent_id=self.agent_id,
            tools=list(self.tools.keys()),
            tavily_enabled=bool(tavily_api_key),
            code_execution_enabled=enable_code_execution,
        )

    async def process(self, message: AgentMessage) -> AgentMessage:
        """
        Process research request with tool use

        PATTERN: Multi-tool orchestration with parallel execution
        WHY: Efficient gathering of information from multiple sources
        """
        content = message.content if isinstance(message.content, dict) else {}
        query = content.get("query", "")
        sources_requested = content.get("sources", ["web"])
        max_results = content.get("max_results", 5)

        if not query:
            return AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                message_type=MessageType.AGENT_ERROR,
                content={"error": "No query provided", "results": []},
                correlation_id=message.message_id,
            )

        api_logger.info(
            "research_request_received",
            query=query,
            sources=sources_requested,
            correlation_id=message.message_id,
        )

        # Map sources to tool names
        source_to_tool = {
            "web": "web_search",
            "arxiv": "arxiv",
            "wikipedia": "wikipedia",
        }

        tools_to_use = [source_to_tool.get(s, s) for s in sources_requested]

        # Execute tools in parallel
        tool_results = await self._execute_tools_parallel(
            query=query,
            tools=tools_to_use,
            max_results=max_results,
        )

        # Flatten and normalize results
        all_results = []
        sources_used = []

        for tool_name, result in tool_results.items():
            if isinstance(result, dict) and "results" in result:
                # Normalize source names (duckduckgo -> web, tavily -> web)
                raw_source = result.get("source", tool_name)
                source_name = "web" if raw_source in ("duckduckgo", "tavily", "web_search") else raw_source
                sources_used.append(source_name)

                for item in result["results"]:
                    # Normalize result structure for tests
                    normalized = {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("snippet", item.get("content", item.get("extract", ""))),
                        "source": source_name,
                    }
                    # Add optional fields if present
                    if "authors" in item:
                        normalized["authors"] = item["authors"]
                    if "relevance_score" in item:
                        normalized["relevance_score"] = item["relevance_score"]
                    if "score" in item:
                        normalized["relevance_score"] = item["score"]

                    all_results.append(normalized)

        # Limit total results
        all_results = all_results[:max_results]

        return AgentMessage(
            sender=self.agent_id,
            recipient=message.sender,
            message_type=MessageType.RESEARCH_RESPONSE,
            content={
                "query": query,
                "results": all_results,
                "sources_used": sources_used,
                "timestamp": datetime.utcnow().isoformat(),
            },
            correlation_id=message.message_id,
            metadata={"sources_used": sources_used},
        )

    async def _execute_tools_parallel(
        self,
        query: str,
        tools: List[str],
        max_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Execute multiple tools in parallel

        PATTERN: Async parallel execution with error isolation
        WHY: Faster results and fault tolerance
        """
        tasks = []
        tool_names = []

        for tool_name in tools:
            if tool_name in self.tools:
                tool_names.append(tool_name)
                tasks.append(
                    self._execute_tool_with_metrics(
                        tool_name=tool_name,
                        query=query,
                        max_results=max_results,
                    )
                )

        # Execute all tools concurrently
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        results = {}
        for tool_name, result in zip(tool_names, results_list):
            if isinstance(result, Exception):
                api_logger.error(
                    "tool_execution_failed",
                    tool=tool_name,
                    error=str(result),
                    error_type=type(result).__name__,
                )
                results[tool_name] = {"error": str(result)}
            else:
                results[tool_name] = result

        return results

    async def _execute_tool_with_metrics(
        self,
        tool_name: str,
        query: str,
        max_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Execute a tool with metrics tracking

        PATTERN: Decorator pattern for metrics
        WHY: Consistent metrics collection across all tools
        """
        start_time = datetime.utcnow()

        try:
            # Check cache first
            cache_key = f"{tool_name}:{query}:{max_results}"
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                api_logger.debug("tool_cache_hit", tool=tool_name, query=query)
                return cached_result

            # Check rate limit
            if not self._check_rate_limit(tool_name):
                raise ToolExecutionError(f"Rate limit exceeded for {tool_name}")

            # Execute tool
            api_logger.info("tool_executing", tool=tool_name, query=query)

            # Increment calls before execution so failed calls are still counted
            self.tool_metrics[tool_name]["calls"] += 1

            tool_func = self.tools[tool_name]
            result = await tool_func(query, max_results=max_results)

            # Cache result
            self._cache_result(cache_key, result)

            # Update metrics
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.tool_metrics[tool_name]["total_time"] += execution_time

            api_logger.info(
                "tool_execution_complete",
                tool=tool_name,
                query=query,
                execution_time_ms=execution_time * 1000,
                result_count=len(result.get("results", [])) if isinstance(result, dict) else 0,
            )

            return result

        except Exception as e:
            self.tool_metrics[tool_name]["errors"] += 1
            api_logger.error(
                "tool_execution_error",
                tool=tool_name,
                query=query,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise ToolExecutionError(f"Tool {tool_name} failed: {str(e)}") from e

    # ==================== Tool Implementations ====================

    @with_retry(max_attempts=2, min_wait=1.0)
    async def _web_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Web search using Tavily API or DuckDuckGo fallback

        PATTERN: Primary/fallback strategy
        WHY: Robustness when primary service is unavailable
        """
        # Try Tavily first if API key is available
        if self.tavily_api_key:
            try:
                return await self._tavily_search(query, max_results)
            except Exception as e:
                api_logger.warning(
                    "tavily_search_failed_fallback_to_duckduckgo",
                    error=str(e),
                )

        # Fallback to DuckDuckGo
        return await self._duckduckgo_search(query, max_results)

    async def _tavily_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search using Tavily API"""
        url = "https://api.tavily.com/search"

        payload = {
            "api_key": self.tavily_api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
        }

        response = await self.http_client.post(url, json=payload)
        response.raise_for_status()

        data = response.json()

        return {
            "source": "tavily",
            "query": query,
            "results": [
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0.0),
                }
                for item in data.get("results", [])
            ],
        }

    async def _duckduckgo_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search using DuckDuckGo Instant Answer API

        NOTE: This is a basic implementation. For production, consider using
        the duckduckgo-search library or a more robust scraping solution.
        """
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }

        response = await self.http_client.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        results = []

        # Abstract (main result)
        if data.get("Abstract"):
            results.append({
                "title": data.get("Heading", ""),
                "url": data.get("AbstractURL", ""),
                "content": data.get("Abstract", ""),
                "source": data.get("AbstractSource", ""),
            })

        # Related topics
        for topic in data.get("RelatedTopics", [])[:max_results - 1]:
            if isinstance(topic, dict) and "Text" in topic:
                results.append({
                    "title": topic.get("Text", "")[:100],
                    "url": topic.get("FirstURL", ""),
                    "content": topic.get("Text", ""),
                })

        return {
            "source": "duckduckgo",
            "query": query,
            "results": results[:max_results],
        }

    @with_retry(max_attempts=2, min_wait=1.0)
    async def _wikipedia_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search Wikipedia using the MediaWiki API

        PATTERN: Public API integration
        WHY: High-quality encyclopedic knowledge
        """
        url = "https://en.wikipedia.org/w/api.php"

        # First, search for articles
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": max_results,
        }

        response = await self.http_client.get(url, params=search_params)
        response.raise_for_status()

        data = response.json()

        results = []
        if "query" in data and "search" in data["query"]:
            for item in data["query"]["search"]:
                # Get article extract
                extract_params = {
                    "action": "query",
                    "prop": "extracts",
                    "exintro": True,
                    "explaintext": True,
                    "pageids": item["pageid"],
                    "format": "json",
                }

                extract_response = await self.http_client.get(url, params=extract_params)
                extract_data = extract_response.json()

                page = extract_data.get("query", {}).get("pages", {}).get(str(item["pageid"]), {})

                results.append({
                    "title": item["title"],
                    "snippet": item["snippet"],
                    "url": f"https://en.wikipedia.org/wiki/{item['title'].replace(' ', '_')}",
                    "extract": page.get("extract", "")[:500],  # First 500 chars
                    "pageid": item["pageid"],
                })

        return {
            "source": "wikipedia",
            "query": query,
            "results": results,
        }

    @with_retry(max_attempts=2, min_wait=1.0)
    async def _arxiv_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search arXiv for academic papers

        PATTERN: Academic API integration
        WHY: Access to scientific research papers
        """
        url = "http://export.arxiv.org/api/query"

        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        response = await self.http_client.get(url, params=params)
        response.raise_for_status()

        # Parse Atom XML response
        root = ET.fromstring(response.text)
        namespace = {"atom": "http://www.w3.org/2005/Atom"}

        results = []
        for entry in root.findall("atom:entry", namespace):
            title = entry.find("atom:title", namespace)
            summary = entry.find("atom:summary", namespace)
            published = entry.find("atom:published", namespace)
            link = entry.find("atom:id", namespace)

            # Get authors
            authors = [
                author.find("atom:name", namespace).text
                for author in entry.findall("atom:author", namespace)
                if author.find("atom:name", namespace) is not None
            ]

            results.append({
                "title": title.text.strip() if title is not None else "",
                "authors": authors,
                "summary": summary.text.strip()[:500] if summary is not None else "",  # First 500 chars
                "published": published.text if published is not None else "",
                "url": link.text if link is not None else "",
                "pdf_url": link.text.replace("abs", "pdf") if link is not None else "",
            })

        return {
            "source": "arxiv",
            "query": query,
            "results": results,
        }

    async def _execute_code(
        self,
        query: str,
        max_results: int = 5,
        language: str = "python",
    ) -> Dict[str, Any]:
        """
        Execute code in sandbox (placeholder for E2B or Flow Nexus integration)

        SPECIFICATION: Sandbox code execution
        PATTERN: Security-first code execution
        WHY: Enable agents to run code safely

        NOTE: This is a placeholder. In production, integrate with:
        - E2B sandboxes: https://e2b.dev
        - Flow Nexus sandboxes: via MCP tools
        """
        if not self.enable_code_execution:
            raise ToolExecutionError("Code execution is disabled")

        api_logger.warning(
            "code_execution_not_implemented",
            query=query,
            language=language,
        )

        return {
            "source": "code_sandbox",
            "status": "not_implemented",
            "message": "Code execution requires E2B or Flow Nexus sandbox integration",
            "language": language,
            "code": query,
        }

    async def _query_knowledge_base(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Query internal knowledge base using SQLite FTS5

        PATTERN: Internal knowledge retrieval
        WHY: Access to historical conversation data
        """
        # Ensure database is initialized
        if not db._initialized:
            await db.initialize()

        # Use FTS5 search
        search_results = await db.search_captures(query, limit=max_results)

        results = []
        for row in search_results:
            results.append({
                "id": row["id"],
                "session_id": row["session_id"],
                "timestamp": row["timestamp"],
                "user_text": row["user_text"],
                "agent_text": row["agent_text"],
                "user_snippet": row.get("user_snippet", ""),
                "agent_snippet": row.get("agent_snippet", ""),
            })

        return {
            "source": "knowledge_base",
            "query": query,
            "results": results,
            "total": len(results),
        }

    # ==================== Caching & Rate Limiting ====================

    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if not expired"""
        if cache_key in self.cache:
            result, cached_at = self.cache[cache_key]
            if datetime.utcnow() - cached_at < self.cache_ttl:
                return result
            else:
                # Expired, remove from cache
                del self.cache[cache_key]
        return None

    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache a tool result"""
        self.cache[cache_key] = (result, datetime.utcnow())

        # Simple cache size management (LRU-like)
        if len(self.cache) > 100:
            # Remove oldest entries
            sorted_cache = sorted(
                self.cache.items(),
                key=lambda x: x[1][1],  # Sort by timestamp
            )
            for key, _ in sorted_cache[:20]:  # Remove oldest 20
                del self.cache[key]

    def _check_rate_limit(self, tool_name: str) -> bool:
        """
        Check if tool call is within rate limit

        PATTERN: Token bucket rate limiting
        WHY: Prevent API abuse and manage costs
        """
        now = datetime.utcnow()

        if tool_name not in self.rate_limits:
            self.rate_limits[tool_name] = []

        # Remove old timestamps
        self.rate_limits[tool_name] = [
            ts for ts in self.rate_limits[tool_name]
            if now - ts < self.rate_limit_window
        ]

        # Check if under limit
        if len(self.rate_limits[tool_name]) >= self.rate_limit_max_calls:
            return False

        # Add current timestamp
        self.rate_limits[tool_name].append(now)
        return True

    def get_tool_metrics(self) -> Dict[str, Any]:
        """
        Get tool execution metrics

        PATTERN: Observability for tools
        WHY: Monitor tool performance and reliability
        """
        metrics = {}
        for tool_name, stats in self.tool_metrics.items():
            avg_time = (
                stats["total_time"] / stats["calls"]
                if stats["calls"] > 0
                else 0
            )
            metrics[tool_name] = {
                "calls": stats["calls"],
                "errors": stats["errors"],
                "error_rate": stats["errors"] / stats["calls"] if stats["calls"] > 0 else 0,
                "avg_execution_time_ms": avg_time * 1000,
            }

        metrics["cache_size"] = len(self.cache)

        return metrics

    async def cleanup(self) -> None:
        """
        Cleanup resources

        PATTERN: Resource cleanup
        WHY: Prevent memory leaks and connection issues
        """
        await self.http_client.aclose()
        self.cache.clear()
        api_logger.info("research_agent_cleanup_complete", agent_id=self.agent_id)

    async def __aenter__(self):
        """Context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        await self.cleanup()
