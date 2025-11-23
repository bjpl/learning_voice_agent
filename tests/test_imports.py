"""
Basic import tests to verify system health
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_config_imports():
    """Test that config module imports correctly"""
    try:
        from app.config import settings
        assert settings is not None
        print("✓ Config imports successfully")
        return True
    except ImportError as e:
        print(f"✗ Config import failed: {e}")
        return False

def test_database_imports():
    """Test that database module imports correctly"""
    try:
        from app.database import db
        assert db is not None
        print("✓ Database imports successfully")
        return True
    except ImportError as e:
        print(f"✗ Database import failed: {e}")
        return False

def test_conversation_imports():
    """Test that conversation handler imports correctly"""
    try:
        from app.conversation_handler import conversation_handler
        assert conversation_handler is not None
        print("✓ Conversation handler imports successfully")
        return True
    except ImportError as e:
        print(f"✗ Conversation handler import failed: {e}")
        return False

def test_audio_imports():
    """Test that audio pipeline imports correctly"""
    try:
        from app.audio_pipeline import audio_pipeline
        assert audio_pipeline is not None
        print("✓ Audio pipeline imports successfully")
        return True
    except ImportError as e:
        print(f"✗ Audio pipeline import failed: {e}")
        return False

def test_state_imports():
    """Test that state manager imports correctly"""
    try:
        from app.state_manager import state_manager
        assert state_manager is not None
        print("✓ State manager imports successfully")
        return True
    except ImportError as e:
        print(f"✗ State manager import failed: {e}")
        return False

def main():
    """Run all import tests"""
    print("Running import tests...\n")
    
    tests = [
        test_config_imports,
        test_database_imports,
        test_conversation_imports,
        test_audio_imports,
        test_state_imports
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    passed = sum(results)
    total = len(results)
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All imports working!")
        return 0
    else:
        print("❌ Some imports failed - check dependencies")
        return 1

if __name__ == "__main__":
    exit(main())