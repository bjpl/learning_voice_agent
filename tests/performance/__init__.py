"""
Performance Testing Package

Contains Locust-based load testing infrastructure for the Learning Voice Agent.

Test Scenarios:
- User authentication flow (registration, login, token refresh)
- Conversation API (authenticated requests)
- WebSocket connections
- GDPR compliance endpoints (export, delete)
- Rate limiting validation

Performance Targets:
- Response time: p95 < 500ms
- Success rate: > 99.5%
- No memory leaks over 10 min run

Usage:
    # Run via shell script
    ./scripts/run_load_tests.sh --scenario load

    # Run via locust directly
    locust -f tests/performance/locustfile.py --host http://localhost:8000
"""
