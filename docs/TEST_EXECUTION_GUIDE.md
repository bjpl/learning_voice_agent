# Test Execution Guide

## Quick Reference

### Install All Dependencies
```bash
cd /home/user/learning_voice_agent

# Install core dependencies
pip install fastapi pydantic pydantic-settings anthropic openai aiosqlite redis python-dotenv

# Install logging and resilience
pip install structlog tenacity circuitbreaker

# Install test dependencies
pip install pytest pytest-asyncio pytest-cov pytest-mock httpx
```

### Run Tests

#### Framework Verification (Guaranteed to Work)
```bash
# Test that pytest is configured correctly
pytest tests/test_framework_verification.py -v

# Expected output: 7/7 tests PASSED
```

#### Unit Tests
```bash
# Run all unit tests
pytest tests/unit/ -v

# Run specific module tests
pytest tests/unit/test_conversation_handler.py -v
pytest tests/unit/test_audio_pipeline.py -v
pytest tests/unit/test_state_manager.py -v
pytest tests/unit/test_database.py -v
```

#### Integration Tests
```bash
# Run all integration tests
pytest tests/integration/ -v

# Run specific integration tests
pytest tests/integration/test_api_endpoints.py -v
pytest tests/integration/test_websocket.py -v
```

#### All Tests with Coverage
```bash
# Run all tests with coverage report
pytest --cov=app --cov-report=html --cov-report=term-missing

# View HTML coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Test by Marker
```bash
# Only unit tests
pytest -m unit -v

# Only integration tests
pytest -m integration -v

# Only async tests
pytest -m asyncio -v

# Only database tests
pytest -m database -v

# Exclude slow tests
pytest -m "not slow" -v
```

### Debugging Tests
```bash
# Show print statements
pytest -s

# Very verbose output
pytest -vv

# Show local variables on failure
pytest -l

# Drop into debugger on failure
pytest --pdb

# Run only last failed tests
pytest --lf

# Run failed tests first
pytest --ff
```

### Parallel Execution (Faster)
```bash
# Install pytest-xdist
pip install pytest-xdist

# Run with auto-detected CPUs
pytest -n auto

# Run with specific number of workers
pytest -n 4
```

## Test Structure

### Files Created
```
/home/user/learning_voice_agent/
├── tests/
│   ├── conftest.py                     # 500+ lines - All fixtures
│   ├── test_framework_verification.py  # Framework validation tests
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_conversation_handler.py  # 20+ tests
│   │   ├── test_audio_pipeline.py        # 25+ tests
│   │   ├── test_state_manager.py         # 20+ tests
│   │   └── test_database.py              # 35+ tests
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_api_endpoints.py         # 20+ tests
│   │   └── test_websocket.py             # 8+ tests
│   └── e2e/
│       └── __init__.py                   # (future)
├── pytest.ini                            # Pytest configuration
├── .coveragerc                           # Coverage configuration
└── docs/
    ├── TESTING.md                        # Comprehensive guide
    └── TEST_SUITE_SUMMARY.md             # This summary
```

### Total Test Count
- **Unit Tests:** 80+ tests
- **Integration Tests:** 25+ tests
- **Framework Tests:** 7 tests
- **Total:** 105+ tests

## Coverage Requirements

### Target: 80%+
```bash
# Check current coverage
pytest --cov=app --cov-report=term-missing

# Generate HTML report
pytest --cov=app --cov-report=html

# Generate XML (for CI/CD)
pytest --cov=app --cov-report=xml
```

### Coverage by Module
| Module | Target | Description |
|--------|--------|-------------|
| conversation_handler.py | 90%+ | Claude API interaction |
| audio_pipeline.py | 85%+ | Audio transcription |
| state_manager.py | 85%+ | Session management |
| database.py | 90%+ | Data persistence |
| main.py | 75%+ | API endpoints |
| models.py | 95%+ | Data models |
| config.py | 70%+ | Configuration |

## Test Categories

### Unit Tests (`tests/unit/`)
- **Fast:** < 100ms per test
- **Isolated:** All external dependencies mocked
- **Focused:** Test single components
- **Coverage:** High coverage of edge cases

### Integration Tests (`tests/integration/`)
- **Realistic:** Test component interactions
- **API Contracts:** Verify request/response flows
- **Speed:** 100ms - 1s per test
- **Mocked:** Only external APIs mocked

### E2E Tests (`tests/e2e/`) - Future
- **Complete:** Full user workflows
- **Real Services:** Minimal mocking
- **Speed:** 1s+ per test
- **Critical Paths:** Test main user journeys

## Common Test Patterns

### Async Test
```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result is not None
```

### Using Fixtures
```python
def test_with_fixture(test_db, sample_context):
    # test_db and sample_context are provided by conftest.py
    result = test_db.save(sample_context)
    assert result > 0
```

### Mocking APIs
```python
from unittest.mock import AsyncMock, patch

@patch('app.conversation_handler.anthropic.AsyncAnthropic')
async def test_with_mock(mock_client):
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    result = await handler.generate_response("test", [])
    assert mock_client.messages.create.called
```

### Testing Exceptions
```python
def test_exception():
    with pytest.raises(ValueError, match="too large"):
        validate_input(large_data)
```

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
        pytest --cov=app --cov-report=xml --cov-report=term-missing

    - name: Upload coverage
      uses: codecov/codecov-action@v2
      with:
        file: ./coverage.xml

    - name: Fail if coverage < 80%
      run: |
        coverage report --fail-under=80
```

## Troubleshooting

### Import Errors
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Or install in editable mode
pip install -e .
```

### Missing Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt

# Install specific missing dependency
pip install <package-name>
```

### Slow Tests
```bash
# Run in parallel
pytest -n auto

# Skip slow tests
pytest -m "not slow"

# Run only fast unit tests
pytest tests/unit/ -m "not slow"
```

### Test Discovery Issues
```bash
# Verify pytest can find tests
pytest --collect-only

# Check for Python syntax errors
python -m py_compile tests/unit/test_*.py
```

### Database Tests Failing
```bash
# Ensure SQLite is available
python -c "import sqlite3; print('SQLite OK')"

# Tests use in-memory DB, no setup needed
pytest tests/unit/test_database.py -v
```

## Performance Benchmarks

### Expected Execution Times
- **Framework Verification:** < 1 second
- **Unit Tests:** 5-10 seconds (with mocks)
- **Integration Tests:** 10-20 seconds (with mocks)
- **Full Suite:** 15-30 seconds (with mocks)

### With Real Services (Future)
- **Full Suite:** 2-5 minutes
- **E2E Tests:** 5-15 minutes

## Best Practices

1. **Run Tests Before Committing**
   ```bash
   pytest tests/unit/ -v
   ```

2. **Check Coverage Regularly**
   ```bash
   pytest --cov=app --cov-report=term-missing
   ```

3. **Write Tests for New Features**
   - Add unit tests for new functions
   - Add integration tests for new endpoints
   - Update fixtures as needed

4. **Keep Tests Fast**
   - Mock external services
   - Use in-memory databases
   - Avoid sleep() in tests

5. **Make Tests Deterministic**
   - No random data (use fixtures)
   - No time-dependent logic
   - Independent test execution

## Resources

- **Pytest Documentation:** https://docs.pytest.org/
- **Pytest Asyncio:** https://pytest-asyncio.readthedocs.io/
- **Coverage.py:** https://coverage.readthedocs.io/
- **FastAPI Testing:** https://fastapi.tiangolo.com/tutorial/testing/
- **Project Testing Guide:** `/home/user/learning_voice_agent/docs/TESTING.md`

## Quick Commands Cheat Sheet

```bash
# Verify framework
pytest tests/test_framework_verification.py -v

# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run unit tests only
pytest tests/unit/ -v

# Run integration tests only
pytest tests/integration/ -v

# Run specific test file
pytest tests/unit/test_conversation_handler.py -v

# Run with markers
pytest -m unit -v

# Run in parallel
pytest -n auto

# Show print output
pytest -s

# Debug mode
pytest --pdb

# Last failed only
pytest --lf

# Coverage report
pytest --cov=app --cov-report=term-missing
```

## Getting Help

1. Check `/home/user/learning_voice_agent/docs/TESTING.md`
2. Review test examples in `tests/unit/`
3. Check pytest documentation
4. Review fixtures in `conftest.py`

---

**Test Suite Status:** ✅ Complete and Ready
**Total Tests:** 105+
**Framework:** ✅ Verified (7/7 tests passing)
**Coverage Target:** 80%+
**Documentation:** ✅ Complete
