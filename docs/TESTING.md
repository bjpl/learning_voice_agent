# Testing Documentation

## Overview

This document describes the testing strategy, infrastructure, and best practices for the Learning Voice Agent application.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Fast, isolated unit tests
│   ├── test_conversation_handler.py
│   ├── test_audio_pipeline.py
│   ├── test_state_manager.py
│   └── test_database.py
├── integration/             # Component integration tests
│   ├── test_api_endpoints.py
│   └── test_websocket.py
└── e2e/                     # End-to-end system tests
    └── (future tests)
```

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Specific test file
pytest tests/unit/test_conversation_handler.py

# Specific test function
pytest tests/unit/test_conversation_handler.py::TestConversationHandler::test_detect_intent_question
```

### Run with Coverage

```bash
# Generate coverage report
pytest --cov=app --cov-report=html

# View coverage report
open htmlcov/index.html  # or browse to file:///.../htmlcov/index.html
```

### Run Tests in Parallel (Faster)

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run with auto-detected CPU cores
pytest -n auto
```

## Test Categories

### Unit Tests (tests/unit/)

**Purpose:** Test individual components in isolation with mocked dependencies.

**Characteristics:**
- Very fast (< 100ms per test)
- No external dependencies (APIs, databases, etc.)
- High coverage of edge cases
- Mock all I/O operations

**Example:**
```python
@pytest.mark.asyncio
async def test_generate_response_success(test_conversation_handler):
    """Test successful response generation"""
    response = await test_conversation_handler.generate_response(
        "I'm learning about Python",
        []
    )
    assert response is not None
    assert len(response) > 0
```

**Files:**
- `test_conversation_handler.py` - Claude API interaction, intent detection
- `test_audio_pipeline.py` - Audio transcription, format detection
- `test_state_manager.py` - Redis state management
- `test_database.py` - SQLite operations and FTS5 search

### Integration Tests (tests/integration/)

**Purpose:** Test how components work together with realistic interactions.

**Characteristics:**
- Slower (100ms - 1s per test)
- Test API contracts and data flow
- May use in-memory databases
- Mock external APIs only

**Example:**
```python
def test_conversation_text_input(client):
    """Test conversation with text input"""
    response = client.post("/api/conversation", json={
        "text": "I'm learning about Python"
    })
    assert response.status_code == 200
    assert "session_id" in response.json()
```

**Files:**
- `test_api_endpoints.py` - FastAPI REST endpoints
- `test_websocket.py` - WebSocket real-time communication

### End-to-End Tests (tests/e2e/)

**Purpose:** Test complete user workflows with real services.

**Characteristics:**
- Slowest (1s+ per test)
- Use real database, Redis
- May use test API keys
- Test critical user paths

**Status:** Future implementation

## Test Fixtures

### Commonly Used Fixtures (conftest.py)

#### Database Fixtures

```python
@pytest.fixture
async def test_db():
    """In-memory database for testing"""
    # Returns initialized Database instance
```

```python
@pytest.fixture
async def db_with_data(test_db):
    """Database pre-populated with test data"""
    # Returns database with sample exchanges
```

#### Mock Fixtures

```python
@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for conversation handler"""
    # Returns AsyncMock with predefined responses
```

```python
@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for audio pipeline"""
    # Returns AsyncMock for Whisper API
```

```python
@pytest.fixture
async def mock_redis():
    """Mock Redis client for state manager"""
    # Returns AsyncMock with in-memory storage
```

#### Test Data Fixtures

```python
@pytest.fixture
def sample_context():
    """Sample conversation context"""
    # Returns list of conversation exchanges
```

```python
@pytest.fixture
def sample_audio_wav():
    """Sample WAV audio bytes"""
    # Returns bytes with WAV file header
```

## Coverage Requirements

### Overall Target: 80%+

**Current Coverage by Module:**

| Module | Target | Description |
|--------|--------|-------------|
| conversation_handler.py | 90%+ | Core conversation logic |
| audio_pipeline.py | 85%+ | Audio processing |
| state_manager.py | 85%+ | Session management |
| database.py | 90%+ | Data persistence |
| main.py | 75%+ | API endpoints |
| models.py | 95%+ | Data models |
| config.py | 70%+ | Configuration |

### Viewing Coverage

```bash
# Terminal report
pytest --cov=app --cov-report=term-missing

# HTML report (detailed)
pytest --cov=app --cov-report=html
open htmlcov/index.html

# XML report (for CI/CD)
pytest --cov=app --cov-report=xml
```

### Coverage Reports

After running tests with coverage, you'll find:
- `htmlcov/` - Interactive HTML coverage report
- `coverage.xml` - XML report for CI/CD systems
- `.coverage` - Coverage data file

## Running Specific Tests

### By Marker

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only async tests
pytest -m asyncio

# Run only database tests
pytest -m database
```

### By Pattern

```bash
# Run tests matching pattern
pytest -k "test_conversation"

# Run tests NOT matching pattern
pytest -k "not slow"

# Combine patterns
pytest -k "test_conversation and not error"
```

### By File

```bash
# Single file
pytest tests/unit/test_conversation_handler.py

# Multiple files
pytest tests/unit/test_conversation_handler.py tests/unit/test_database.py

# All files in directory
pytest tests/unit/
```

## Writing Tests

### Test Structure (Arrange-Act-Assert)

```python
@pytest.mark.asyncio
async def test_example(test_conversation_handler):
    # ARRANGE - Set up test data and mocks
    user_text = "I'm learning about Python"
    context = []

    # ACT - Execute the code being tested
    response = await test_conversation_handler.generate_response(
        user_text,
        context
    )

    # ASSERT - Verify the results
    assert response is not None
    assert len(response) > 0
```

### Test Naming Convention

```python
# Pattern: test_<function>_<scenario>
def test_detect_intent_question():
    """Test intent detection for questions"""

def test_generate_response_with_context():
    """Test response generation with conversation context"""

def test_save_exchange_with_metadata():
    """Test saving exchange with metadata"""
```

### Testing Async Code

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    """Test async function"""
    result = await some_async_function()
    assert result is not None
```

### Testing Exceptions

```python
import pytest

def test_validates_input():
    """Test that invalid input raises error"""
    with pytest.raises(ValueError, match="too large"):
        validate_audio(large_audio)
```

### Mocking External Services

```python
from unittest.mock import AsyncMock, patch

@patch('app.conversation_handler.anthropic.AsyncAnthropic')
async def test_with_mocked_api(mock_client):
    """Test with mocked API"""
    # Setup mock
    mock_client.messages.create = AsyncMock(
        return_value=mock_response
    )

    # Test code
    result = await handler.generate_response("test", [])

    # Verify mock was called
    assert mock_client.messages.create.called
```

## Best Practices

### 1. Keep Tests Fast

- Mock external APIs and slow I/O
- Use in-memory databases for unit tests
- Avoid sleep() - use time mocking instead

### 2. Test One Thing

- Each test should verify one behavior
- Use descriptive test names
- Keep assertions focused

### 3. Use Fixtures for Setup

- Don't repeat setup code
- Use conftest.py for shared fixtures
- Leverage pytest's fixture scoping

### 4. Write Readable Tests

```python
# Good - Clear and descriptive
def test_conversation_adds_followup_for_short_input():
    """Test that short inputs trigger follow-up questions"""

# Bad - Unclear purpose
def test_conv_1():
    """Test conversation"""
```

### 5. Test Edge Cases

- Empty inputs
- Maximum values
- Null/None handling
- Error conditions
- Boundary values

### 6. Keep Tests Independent

- Tests should not depend on each other
- Each test should clean up after itself
- Use fixtures for common setup

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11

    - name: Install dependencies
      run: |
        pip install -r requirements.txt

    - name: Run tests
      run: |
        pytest --cov=app --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v2
      with:
        file: ./coverage.xml
```

## Debugging Tests

### Run with Verbose Output

```bash
pytest -vv
```

### Show Print Statements

```bash
pytest -s
```

### Drop into Debugger on Failure

```bash
pytest --pdb
```

### Run Last Failed Tests Only

```bash
pytest --lf
```

### Run Failed First, Then Others

```bash
pytest --ff
```

## Common Issues

### Issue: Tests are slow

**Solution:**
```bash
# Use parallel execution
pytest -n auto

# Run only fast tests during development
pytest -m "not slow"
```

### Issue: Import errors

**Solution:**
```bash
# Ensure app is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Or install in editable mode
pip install -e .
```

### Issue: Async tests not running

**Solution:**
```python
# Add asyncio marker
@pytest.mark.asyncio
async def test_async_function():
    ...

# Or set in pytest.ini
[pytest]
asyncio_mode = auto
```

### Issue: Database tests failing

**Solution:**
```python
# Use in-memory database
db = Database(":memory:")

# Or ensure cleanup
@pytest.fixture
async def test_db():
    db = Database(":memory:")
    await db.initialize()
    yield db
    # Cleanup happens automatically with in-memory
```

## Test Metrics

### Current Status

- **Total Tests:** 80+
- **Coverage:** 85%
- **Execution Time:** ~5 seconds (with mocks)
- **Pass Rate:** 100%

### Quality Gates

Before merging code:
- All tests must pass
- Coverage must be >= 80%
- No critical code smells
- All new code must have tests

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Asyncio](https://pytest-asyncio.readthedocs.io/)
- [Coverage.py](https://coverage.readthedocs.io/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)

## Getting Help

If you encounter issues with tests:

1. Check this documentation
2. Review existing tests for patterns
3. Check pytest output for details
4. Consult the team's testing guidelines

## Future Improvements

- [ ] Add E2E tests for complete workflows
- [ ] Add performance benchmarks
- [ ] Add load testing
- [ ] Add security testing
- [ ] Improve test data factories
- [ ] Add mutation testing
