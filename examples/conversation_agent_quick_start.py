#!/usr/bin/env python
"""
ConversationAgent Quick Start Guide

This example shows the simplest way to use the new ConversationAgent
with Claude 3.5 Sonnet and tool calling.
"""
import asyncio
from app.agents import ConversationAgent, AgentMessage, MessageRole, MessageType


async def main():
    print("=" * 60)
    print("ConversationAgent Quick Start")
    print("Claude 3.5 Sonnet with Tool Calling")
    print("=" * 60)

    # Step 1: Create agent
    print("\n1. Creating ConversationAgent...")
    agent = ConversationAgent(
        model="claude-3-5-sonnet-20241022",
        enable_tools=True,
        enable_streaming=False
    )
    print(f"   ✓ Agent created with {len(agent.tools)} tools")

    # Step 2: Simple conversation
    print("\n2. Simple conversation (no tools)...")
    message = AgentMessage(
        role=MessageRole.USER,
        content="Hello! Can you introduce yourself?",
        message_type=MessageType.TEXT,
    )

    response = await agent.process(message)
    print(f"   User: {message.content}")
    print(f"   Agent: {response.content}")

    # Step 3: Using calculator tool
    print("\n3. Using calculator tool...")
    message = AgentMessage(
        role=MessageRole.USER,
        content="What is the square root of 256?",
        message_type=MessageType.TEXT,
    )

    response = await agent.process(message)
    print(f"   User: {message.content}")
    print(f"   Agent: {response.content}")

    # Step 4: Context-aware conversation
    print("\n4. Context-aware multi-turn conversation...")
    conversation_history = []

    turns = [
        "I'm learning about Python programming.",
        "What's a good framework for web development?",
        "How does that relate to what I'm learning?"
    ]

    for user_input in turns:
        message = AgentMessage(
            role=MessageRole.USER,
            content=user_input,
            context={"conversation_history": conversation_history}
        )

        response = await agent.process(message)
        print(f"\n   User: {user_input}")
        print(f"   Agent: {response.content}")

        # Update history
        conversation_history.append({
            "user": user_input,
            "agent": response.content
        })

    # Step 5: Check metrics
    print("\n5. Agent Metrics:")
    metrics = agent.get_metrics()
    print(f"   Messages processed: {metrics['messages_received']}")
    print(f"   Messages sent: {metrics['messages_sent']}")
    print(f"   Errors: {metrics['errors']}")
    avg_time = (
        sum(metrics['processing_times']) / len(metrics['processing_times'])
        if metrics['processing_times'] else 0
    )
    print(f"   Avg processing time: {avg_time*1000:.2f}ms")

    # Step 6: Check capabilities
    print("\n6. Agent Capabilities:")
    capabilities = agent.get_capabilities()
    print(f"   Model: {capabilities['model']}")
    print(f"   Tools: {', '.join(capabilities['tools'])}")
    print(f"   Features:")
    for feature, enabled in capabilities['features'].items():
        status = "✓" if enabled else "✗"
        print(f"     {status} {feature}")

    print("\n" + "=" * 60)
    print("Quick start completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
