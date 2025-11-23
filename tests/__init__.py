"""
Test Suite for Learning Voice Agent

This package contains comprehensive tests for all modules:
- Unit tests: Fast, isolated tests for individual components
- Integration tests: Tests for API endpoints and service interactions
- E2E tests: Full system tests (requires external services)

Test Organization:
- test_config.py: Configuration and settings tests
- test_conversation_handler.py: Claude conversation handler tests
- test_database.py: SQLite database and FTS5 search tests
- test_state_manager.py: Redis state management tests
- test_audio_pipeline.py: Audio transcription pipeline tests
- test_models.py: Pydantic model validation tests
- test_main_integration.py: FastAPI endpoint integration tests
- test_twilio_handler.py: Twilio webhook and TwiML tests

Running Tests:
    # All tests
    pytest tests/ -v

    # Unit tests only (fast)
    pytest tests/ -m unit -v

    # Integration tests
    pytest tests/ -m integration -v

    # With coverage report
    pytest tests/ --cov=app --cov-report=html --cov-report=term-missing

    # Skip slow tests
    pytest tests/ -m "not slow" -v
"""
