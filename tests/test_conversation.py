import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.conversation_handler import conversation_handler

async def test_conversation_flow():
    """Test that conversation handler generates responses"""
    # Test with empty context
    response = await conversation_handler.generate_response(
        user_text="Hello, I'm learning about AI",
        context=[],
        session_metadata=None
    )
    
    print(f"User: Hello, I'm learning about AI")
    print(f"Claude: {response}")
    
    # Check response quality
    assert response, "Response should not be empty"
    assert len(response) > 10, "Response too short"
    assert "?" in response or "learning" in response.lower(), "Should engage with topic"
    
    print("✅ Conversation test passed!")
    return True

async def test_follow_up():
    """Test conversation with context"""
    # First exchange
    context = [{
        "user": "I'm studying machine learning",
        "agent": "That's exciting! What aspect of machine learning interests you most?"
    }]
    
    response = await conversation_handler.generate_response(
        user_text="Neural networks seem really complex",
        context=context,
        session_metadata=None
    )
    
    print(f"\nWith context:")
    print(f"User: Neural networks seem really complex")
    print(f"Claude: {response}")
    
    assert response, "Response should not be empty"
    assert len(response.split()) < 50, "Response should be under 3 sentences"
    
    print("✅ Follow-up test passed!")
    return True

async def main():
    """Run all tests"""
    print("🧪 Testing Conversation Handler\n")
    
    try:
        await test_conversation_flow()
        await test_follow_up()
        print("\n✅ All conversation tests passed!")
        return 0
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure your API keys are set in .env")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))