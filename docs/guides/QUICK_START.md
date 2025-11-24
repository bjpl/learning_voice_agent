# 📋 Quick Start Guide: Getting Your Learning Voice Agent Running

## Step 1: Set Up API Keys 🔑

### A. Get Your API Keys

1. **Anthropic API Key (for Claude)**
   - Go to: https://console.anthropic.com/
   - Sign up/Login → Settings → API Keys
   - Click "Create Key" → Copy the key starting with `sk-ant-api...`

2. **OpenAI API Key (for Whisper)**
   - Go to: https://platform.openai.com/api-keys
   - Sign up/Login → Create new secret key
   - Copy the key starting with `sk-...`

### B. Configure Environment

```bash
# In your terminal, from the project root:
cd C:\Users\brand\Development\Project_Workspace\learning_voice_agent

# Copy the template to create your .env file
copy .env.local .env

# Open .env in your editor (or use notepad)
notepad .env
```

### C. Edit .env File

Replace the placeholder values:
```env
# BEFORE (placeholders)
ANTHROPIC_API_KEY=sk-ant-test-key-placeholder
OPENAI_API_KEY=sk-test-key-placeholder

# AFTER (your actual keys)
ANTHROPIC_API_KEY=sk-ant-api03-YOUR-ACTUAL-KEY-HERE
OPENAI_API_KEY=sk-YOUR-ACTUAL-OPENAI-KEY-HERE
```

## Step 2: Verify Installation ✅

```bash
# Test that everything imports correctly
python tests/test_imports.py

# Expected output:
# ✅ All imports working!
```

## Step 3: Test Basic Conversation Flow 🎤

### A. Start the Server

```bash
# Install any missing dependencies first
pip install -r requirements.txt

# Start the FastAPI server
python -m uvicorn app.main:app --reload

# Server will start at: http://localhost:8000
```

### B. Test the API

Open a new terminal and test:

```bash
# Test health check
curl http://localhost:8000/

# Test conversation endpoint (Windows PowerShell)
$body = @{
    text = "I'm learning about distributed systems"
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8000/api/conversation" `
    -Method POST `
    -Body $body `
    -ContentType "application/json"

# Or use Python for testing
python -c "
import requests
response = requests.post('http://localhost:8000/api/conversation', 
    json={'text': 'Hello, I am learning Python'})
print(response.json())
"
```

### C. Test the Web Interface

1. Open your browser to: http://localhost:8000/static/index.html
2. Click the microphone button (grant permission when asked)
3. Hold to speak, release to send
4. You should hear Claude's response

## Step 4: Create Your First Test 🧪

Create a new file: `tests/test_conversation.py`

```python
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
```

Run the test:
```bash
python tests/test_conversation.py
```

## Step 5: Add Basic Logging 📝

Create `app/logger.py`:

```python
import logging
import sys

def setup_logger(name: str = "voice_agent"):
    """Set up structured logging"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Console handler with formatting
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

# Global logger instance
logger = setup_logger()
```

Then replace print statements:

```bash
# Quick way to find all print statements
grep -n "print(" app/*.py

# Example replacement in app/conversation_handler.py
# Before: print(f"Claude API error: {e}")
# After: logger.error(f"Claude API error: {e}")
```

## Troubleshooting 🔧

### If imports fail:
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### If API calls fail:
```python
# Test your keys directly
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('Anthropic key:', os.getenv('ANTHROPIC_API_KEY')[:20] + '...')
print('OpenAI key:', os.getenv('OPENAI_API_KEY')[:20] + '...')
"
```

### If Redis isn't running:
```bash
# Windows: Install Redis via WSL or use Docker
docker run -d -p 6379:6379 redis:alpine

# Or disable Redis temporarily in app/config.py
# REDIS_URL=redis://localhost:6379  # Comment out
```

## Success Checklist ✅

- [ ] API keys added to `.env` file
- [ ] `python tests/test_imports.py` passes
- [ ] Server starts with `uvicorn app.main:app`
- [ ] Health check returns JSON at http://localhost:8000/
- [ ] Conversation endpoint responds to POST requests
- [ ] Web interface loads at http://localhost:8000/static/index.html
- [ ] Microphone permission granted in browser
- [ ] Test conversation generates Claude response
- [ ] `tests/test_conversation.py` passes

## Next Steps After Success 🚀

1. **Commit your working state:**
   ```bash
   git add .env
   git commit -m "Add API keys and verify system works"
   ```

2. **Try a real conversation:**
   - Open the web interface
   - Have a 2-3 exchange conversation
   - Check if context is maintained

3. **Start iterating:**
   - Add more tests
   - Improve error handling
   - Enhance the UI

---

**🎯 Goal: Within 30 minutes, you should have a working voice conversation with Claude!**