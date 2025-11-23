#!/usr/bin/env python
"""Quick import verification for ConversationAgent"""

from app.agents import ConversationAgent, conversation_agent_v2, tool_registry

print("✓ All imports successful")
print(f"Model: {conversation_agent_v2.model}")
print(f"Tools: {len(conversation_agent_v2.tools)} registered")

tools = tool_registry.get_all_schemas()
tool_names = [t["name"] for t in tools]
print(f"Tool names: {tool_names}")

# Test capabilities
caps = conversation_agent_v2.get_capabilities()
print(f"\nCapabilities:")
print(f"  Agent Type: {caps['agent_type']}")
print(f"  Model: {caps['model']}")
print(f"  Features:")
for feature, enabled in caps['features'].items():
    status = "✓" if enabled else "✗"
    print(f"    {status} {feature}")

print("\n✓✓✓ ConversationAgent implementation successful! ✓✓✓")
