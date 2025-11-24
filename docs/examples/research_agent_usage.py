"""
ResearchAgent Usage Examples
PATTERN: Practical examples for Phase 2 multi-agent system
WHY: Demonstrate how to use ResearchAgent in various scenarios
"""

import asyncio
import os
from app.agents.research_agent import ResearchAgent
from app.agents.base import AgentMessage, MessageType
from app.logger import api_logger


# ==================== Example 1: Basic Research Query ====================

async def example_basic_research():
    """
    Basic research query using Wikipedia
    """
    print("=" * 60)
    print("Example 1: Basic Research Query")
    print("=" * 60)

    async with ResearchAgent() as agent:
        # Create research request
        message = AgentMessage(
            sender="user",
            recipient=agent.agent_id,
            message_type=MessageType.REQUEST,
            content={
                "query": "quantum computing",
                "tools": ["wikipedia"],
                "max_results": 3,
            },
        )

        # Process request
        response = await agent.process(message)

        # Display results
        print(f"\nQuery: {response.content['query']}")
        print(f"Tools used: {response.content['tools_used']}")
        print("\nResults:")

        for tool, data in response.content["results"].items():
            print(f"\n{tool.upper()}:")
            if "results" in data:
                for i, result in enumerate(data["results"], 1):
                    print(f"  {i}. {result.get('title', 'N/A')}")
                    print(f"     URL: {result.get('url', 'N/A')}")
                    print(f"     Snippet: {result.get('snippet', result.get('extract', 'N/A'))[:100]}...")


# ==================== Example 2: Multi-Tool Research ====================

async def example_multi_tool_research():
    """
    Research using multiple tools in parallel
    """
    print("\n" + "=" * 60)
    print("Example 2: Multi-Tool Research")
    print("=" * 60)

    async with ResearchAgent() as agent:
        message = AgentMessage(
            sender="user",
            recipient=agent.agent_id,
            message_type=MessageType.REQUEST,
            content={
                "query": "neural networks deep learning",
                "tools": ["wikipedia", "arxiv", "web_search"],
                "max_results": 2,
            },
        )

        response = await agent.process(message)

        print(f"\nQuery: {response.content['query']}")
        print(f"Tools used: {', '.join(response.content['tools_used'])}")

        # Show results from each tool
        results = response.content["results"]

        if "wikipedia" in results and "results" in results["wikipedia"]:
            print("\nWikipedia Results:")
            for r in results["wikipedia"]["results"][:2]:
                print(f"  - {r['title']}")

        if "arxiv" in results and "results" in results["arxiv"]:
            print("\nArXiv Papers:")
            for r in results["arxiv"]["results"][:2]:
                print(f"  - {r['title']}")
                print(f"    Authors: {', '.join(r['authors'][:3])}")


# ==================== Example 3: Knowledge Base Search ====================

async def example_knowledge_base_search():
    """
    Search internal knowledge base
    """
    print("\n" + "=" * 60)
    print("Example 3: Knowledge Base Search")
    print("=" * 60)

    from app.database import db

    # Initialize database
    await db.initialize()

    # Add some test data
    await db.save_exchange(
        session_id="example-session",
        user_text="What is machine learning?",
        agent_text="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    )

    await db.save_exchange(
        session_id="example-session",
        user_text="Tell me about neural networks",
        agent_text="Neural networks are computing systems inspired by biological neural networks.",
    )

    # Now search the knowledge base
    async with ResearchAgent() as agent:
        message = AgentMessage(
            sender="user",
            recipient=agent.agent_id,
            message_type=MessageType.REQUEST,
            content={
                "query": "machine learning neural",
                "tools": ["knowledge_base"],
                "max_results": 5,
            },
        )

        response = await agent.process(message)

        print(f"\nQuery: {response.content['query']}")
        print("\nKnowledge Base Results:")

        kb_results = response.content["results"]["knowledge_base"]
        for result in kb_results["results"]:
            print(f"\n  Session: {result['session_id']}")
            print(f"  User: {result['user_text'][:80]}...")
            print(f"  Agent: {result['agent_text'][:80]}...")


# ==================== Example 4: Agent Communication ====================

async def example_agent_communication():
    """
    Demonstrate agent-to-agent communication
    """
    print("\n" + "=" * 60)
    print("Example 4: Agent Communication")
    print("=" * 60)

    # Create two agents
    async with ResearchAgent(agent_id="researcher-1") as agent1, \
                ResearchAgent(agent_id="researcher-2") as agent2:

        # Agent 1 sends request to Agent 2
        request = await agent1.send_message(
            recipient=agent2.agent_id,
            message_type=MessageType.REQUEST,
            content={
                "query": "Python programming",
                "tools": ["wikipedia"],
                "max_results": 2,
            },
        )

        # Agent 2 receives and processes
        message = await agent1.outbox.get()
        await agent2.receive_message(message)
        response = await agent2.process(message)

        print(f"Agent 1 ({agent1.agent_id}) → Agent 2 ({agent2.agent_id})")
        print(f"Request: {message.content['query']}")
        print(f"Response type: {response.message_type.value}")
        print(f"Results: {len(response.content['results'])} tool(s)")


# ==================== Example 5: Metrics and Monitoring ====================

async def example_metrics():
    """
    Monitor agent and tool metrics
    """
    print("\n" + "=" * 60)
    print("Example 5: Metrics and Monitoring")
    print("=" * 60)

    async with ResearchAgent() as agent:
        # Make several requests
        for query in ["Python", "JavaScript", "Rust"]:
            message = AgentMessage(
                sender="user",
                recipient=agent.agent_id,
                message_type=MessageType.REQUEST,
                content={
                    "query": query,
                    "tools": ["wikipedia"],
                    "max_results": 1,
                },
            )
            await agent.process(message)

        # Get metrics
        agent_metrics = agent.get_metrics()
        tool_metrics = agent.get_tool_metrics()

        print("\nAgent Metrics:")
        print(f"  Messages received: {agent_metrics['messages_received']}")
        print(f"  Messages sent: {agent_metrics['messages_sent']}")
        print(f"  Errors: {agent_metrics['errors']}")
        print(f"  Avg processing time: {agent_metrics['avg_processing_time_ms']:.2f}ms")

        print("\nTool Metrics:")
        for tool_name, metrics in tool_metrics.items():
            if metrics["calls"] > 0:
                print(f"  {tool_name}:")
                print(f"    Calls: {metrics['calls']}")
                print(f"    Errors: {metrics['errors']}")
                print(f"    Avg execution time: {metrics['avg_execution_time_ms']:.2f}ms")


# ==================== Example 6: Error Handling ====================

async def example_error_handling():
    """
    Demonstrate error handling and recovery
    """
    print("\n" + "=" * 60)
    print("Example 6: Error Handling")
    print("=" * 60)

    async with ResearchAgent() as agent:
        # Request with missing query
        bad_message = AgentMessage(
            sender="user",
            recipient=agent.agent_id,
            message_type=MessageType.REQUEST,
            content={"tools": ["wikipedia"]},  # Missing query!
        )

        response = await agent.process(bad_message)

        print(f"Response type: {response.message_type.value}")
        if response.message_type == MessageType.ERROR:
            print(f"Error: {response.content['error']}")

        # Request with invalid tool
        mixed_message = AgentMessage(
            sender="user",
            recipient=agent.agent_id,
            message_type=MessageType.REQUEST,
            content={
                "query": "test",
                "tools": ["invalid_tool", "wikipedia"],  # One invalid
            },
        )

        response = await agent.process(mixed_message)

        print(f"\nMixed valid/invalid tools:")
        print(f"Response type: {response.message_type.value}")
        if "results" in response.content:
            for tool, result in response.content["results"].items():
                if "error" in result:
                    print(f"  {tool}: ERROR - {result['error']}")
                else:
                    print(f"  {tool}: SUCCESS")


# ==================== Example 7: Caching Demo ====================

async def example_caching():
    """
    Demonstrate result caching
    """
    print("\n" + "=" * 60)
    print("Example 7: Result Caching")
    print("=" * 60)

    async with ResearchAgent() as agent:
        query = "Albert Einstein"

        # First query
        import time
        start = time.time()
        result1 = await agent._wikipedia_search(query, max_results=2)
        time1 = time.time() - start

        # Cache the result
        cache_key = f"wikipedia:{query}:2"
        agent._cache_result(cache_key, result1)

        # Second query (from cache)
        start = time.time()
        result2 = agent._get_cached_result(cache_key)
        time2 = time.time() - start

        print(f"\nQuery: {query}")
        print(f"First query time: {time1*1000:.2f}ms")
        print(f"Cached query time: {time2*1000:.2f}ms")
        print(f"Speedup: {time1/time2:.1f}x")
        print(f"Cache hit: {result2 is not None}")


# ==================== Example 8: Production Usage Pattern ====================

async def example_production_pattern():
    """
    Production-ready usage pattern with proper lifecycle management
    """
    print("\n" + "=" * 60)
    print("Example 8: Production Usage Pattern")
    print("=" * 60)

    # Initialize agent with API keys from environment
    agent = ResearchAgent(
        agent_id="production-researcher",
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        enable_code_execution=False,  # Disabled for safety
    )

    try:
        # Start agent in background
        agent_task = asyncio.create_task(agent.run())

        # Send research request
        await agent.receive_message(
            AgentMessage(
                sender="orchestrator",
                recipient=agent.agent_id,
                message_type=MessageType.REQUEST,
                content={
                    "query": "latest AI research",
                    "tools": ["arxiv", "web_search"],
                    "max_results": 3,
                },
            )
        )

        # Wait for response
        await asyncio.sleep(2)  # Give time to process

        # Get response from outbox
        if not agent.outbox.empty():
            response = await agent.outbox.get()
            print(f"\nReceived response: {response.message_type.value}")
            print(f"Tools used: {response.content.get('tools_used', [])}")

        # Monitor metrics
        metrics = agent.get_metrics()
        print(f"\nAgent Status:")
        print(f"  Running: {metrics['is_running']}")
        print(f"  Messages processed: {metrics['messages_received']}")
        print(f"  Pending: {metrics['pending_inbox']}")

    finally:
        # Clean shutdown
        await agent.stop()
        await agent_task
        await agent.cleanup()
        print("\nAgent shutdown complete")


# ==================== Main Runner ====================

async def main():
    """Run all examples"""
    print("\n🔬 ResearchAgent Usage Examples\n")

    examples = [
        example_basic_research,
        example_multi_tool_research,
        example_knowledge_base_search,
        example_agent_communication,
        example_metrics,
        example_error_handling,
        example_caching,
        example_production_pattern,
    ]

    for example_func in examples:
        try:
            await example_func()
            await asyncio.sleep(0.5)  # Brief pause between examples
        except Exception as e:
            print(f"\n❌ Example failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("✅ All examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())
