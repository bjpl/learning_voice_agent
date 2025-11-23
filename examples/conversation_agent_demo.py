"""
ConversationAgent Demo - Example Usage

Demonstrates the capabilities of the new ConversationAgent with Claude 3.5 Sonnet
"""
import asyncio
from app.agents.conversation_agent import ConversationAgent
from app.agents.base import AgentMessage, MessageRole, MessageType


async def demo_basic_conversation():
    """Demo: Basic conversation without tools"""
    print("\n=== DEMO 1: Basic Conversation ===\n")

    agent = ConversationAgent(enable_tools=False)

    message = AgentMessage(
        role=MessageRole.USER,
        content="Tell me about machine learning in one sentence.",
        message_type=MessageType.TEXT,
    )

    response = await agent.process(message)
    print(f"User: {message.content}")
    print(f"Agent: {response.content}")
    print(f"Tokens used: {response.metadata.tokens_used}")
    print(f"Processing time: {response.metadata.processing_time_ms:.2f}ms")


async def demo_calculator_tool():
    """Demo: Calculator tool usage"""
    print("\n=== DEMO 2: Calculator Tool ===\n")

    agent = ConversationAgent(enable_tools=True)

    message = AgentMessage(
        role=MessageRole.USER,
        content="What is the square root of 144 plus 10?",
        message_type=MessageType.TEXT,
    )

    response = await agent.process(message)
    print(f"User: {message.content}")
    print(f"Agent: {response.content}")


async def demo_memory_tool():
    """Demo: Memory storage and retrieval"""
    print("\n=== DEMO 3: Memory Tool ===\n")

    agent = ConversationAgent(enable_tools=True)

    # First conversation - store preferences
    message1 = AgentMessage(
        role=MessageRole.USER,
        content="My name is Alice and I love Python programming.",
        context={"memory_store": {}},
    )

    response1 = await agent.process(message1)
    print(f"User: {message1.content}")
    print(f"Agent: {response1.content}")

    # Second conversation - recall preferences
    message2 = AgentMessage(
        role=MessageRole.USER,
        content="What's my name and what do I like?",
        context=response1.context,  # Pass along the context with stored memory
    )

    response2 = await agent.process(message2)
    print(f"\nUser: {message2.content}")
    print(f"Agent: {response2.content}")


async def demo_datetime_tool():
    """Demo: DateTime tool usage"""
    print("\n=== DEMO 4: DateTime Tool ===\n")

    agent = ConversationAgent(enable_tools=True)

    message = AgentMessage(
        role=MessageRole.USER,
        content="What day of the week is it?",
        message_type=MessageType.TEXT,
    )

    response = await agent.process(message)
    print(f"User: {message.content}")
    print(f"Agent: {response.content}")


async def demo_context_awareness():
    """Demo: Context-aware multi-turn conversation"""
    print("\n=== DEMO 5: Context-Aware Conversation ===\n")

    agent = ConversationAgent(enable_tools=True)

    conversation_history = []

    # Turn 1
    message1 = AgentMessage(
        role=MessageRole.USER,
        content="I'm learning about neural networks.",
        context={"conversation_history": conversation_history},
    )

    response1 = await agent.process(message1)
    print(f"User: {message1.content}")
    print(f"Agent: {response1.content}\n")

    # Update history
    conversation_history.append({
        "user": message1.content,
        "agent": response1.content,
    })

    # Turn 2
    message2 = AgentMessage(
        role=MessageRole.USER,
        content="What's backpropagation?",
        context={"conversation_history": conversation_history},
    )

    response2 = await agent.process(message2)
    print(f"User: {message2.content}")
    print(f"Agent: {response2.content}\n")

    # Update history
    conversation_history.append({
        "user": message2.content,
        "agent": response2.content,
    })

    # Turn 3 - Reference earlier context
    message3 = AgentMessage(
        role=MessageRole.USER,
        content="How does that relate to what I'm learning?",
        context={"conversation_history": conversation_history},
    )

    response3 = await agent.process(message3)
    print(f"User: {message3.content}")
    print(f"Agent: {response3.content}")


async def demo_search_tool():
    """Demo: Search through conversation history"""
    print("\n=== DEMO 6: Search Tool ===\n")

    agent = ConversationAgent(enable_tools=True)

    # Build up conversation history
    conversation_history = [
        {"user": "I love Python programming", "agent": "That's great!"},
        {"user": "I'm working on machine learning", "agent": "Interesting!"},
        {"user": "Neural networks are fascinating", "agent": "Agreed!"},
        {"user": "I use TensorFlow", "agent": "Good choice!"},
    ]

    message = AgentMessage(
        role=MessageRole.USER,
        content="What did I mention about programming languages?",
        context={"conversation_history": conversation_history},
    )

    response = await agent.process(message)
    print(f"User: {message.content}")
    print(f"Agent: {response.content}")


async def demo_intent_detection():
    """Demo: Intent detection capabilities"""
    print("\n=== DEMO 7: Intent Detection ===\n")

    agent = ConversationAgent()

    test_phrases = [
        "What is machine learning?",
        "Calculate 2 plus 2",
        "What time is it?",
        "Remember my name is Bob",
        "Goodbye!",
        "I think neural networks are cool",
    ]

    for phrase in test_phrases:
        intent = agent.detect_intent(phrase)
        print(f"Phrase: '{phrase}'")
        print(f"Intent: {intent}\n")


async def demo_agent_metrics():
    """Demo: Agent metrics and monitoring"""
    print("\n=== DEMO 8: Agent Metrics ===\n")

    agent = ConversationAgent(enable_tools=True)

    # Process several messages
    for i in range(3):
        message = AgentMessage(
            role=MessageRole.USER,
            content=f"Test message {i+1}: What is {i+1} times 2?",
        )
        await agent.process(message)

    # Get metrics
    metrics = agent.get_metrics()
    print("Agent Performance Metrics:")
    print(f"  Total messages: {metrics['total_messages']}")
    print(f"  Total tokens: {metrics['total_tokens']}")
    print(f"  Total errors: {metrics['total_errors']}")
    print(f"  Avg processing time: {metrics['average_processing_time_ms']:.2f}ms")


async def demo_capabilities():
    """Demo: Agent capabilities reporting"""
    print("\n=== DEMO 9: Agent Capabilities ===\n")

    agent = ConversationAgent(enable_tools=True)

    capabilities = agent.get_capabilities()
    print(f"Agent: {capabilities['agent_name']}")
    print(f"Model: {capabilities['model']}")
    print(f"\nFeatures:")
    for feature, enabled in capabilities['features'].items():
        status = "✓" if enabled else "✗"
        print(f"  {status} {feature}")

    print(f"\nAvailable Tools: {', '.join(capabilities['tools'])}")
    print(f"Max Context Length: {capabilities['max_context_length']} exchanges")


async def demo_error_handling():
    """Demo: Error handling and resilience"""
    print("\n=== DEMO 10: Error Handling ===\n")

    agent = ConversationAgent()

    # Test with various edge cases
    test_cases = [
        ("", "Empty message"),
        ("a" * 10000, "Very long message (truncated)"),
        ("!@#$%^&*()", "Special characters"),
    ]

    for content, description in test_cases:
        try:
            message = AgentMessage(
                role=MessageRole.USER,
                content=content[:100] if len(content) > 100 else content,
            )
            response = await agent.process(message)
            print(f"{description}: ✓ Handled successfully")
        except Exception as e:
            print(f"{description}: ✗ Error - {str(e)}")


async def main():
    """Run all demos"""
    print("=" * 60)
    print("ConversationAgent Demo Suite")
    print("Claude 3.5 Sonnet with Tool Calling")
    print("=" * 60)

    demos = [
        ("Basic Conversation", demo_basic_conversation),
        ("Calculator Tool", demo_calculator_tool),
        ("Memory Tool", demo_memory_tool),
        ("DateTime Tool", demo_datetime_tool),
        ("Context Awareness", demo_context_awareness),
        ("Search Tool", demo_search_tool),
        ("Intent Detection", demo_intent_detection),
        ("Agent Metrics", demo_agent_metrics),
        ("Capabilities", demo_capabilities),
        ("Error Handling", demo_error_handling),
    ]

    for name, demo_func in demos:
        try:
            await demo_func()
            print(f"\n✓ {name} completed successfully")
        except Exception as e:
            print(f"\n✗ {name} failed: {str(e)}")

        await asyncio.sleep(0.5)  # Small delay between demos

    print("\n" + "=" * 60)
    print("Demo suite completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Run demos
    asyncio.run(main())
