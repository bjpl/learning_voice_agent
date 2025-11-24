# Changelog

All notable changes to the Learning Voice Agent project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - Plan A: Security First
- **JWT Authentication System**
  - User registration with password validation (min 8 chars, mixed case, digit required)
  - Login with bcrypt password hashing
  - Access tokens (15 min lifetime) + Refresh tokens (7 days lifetime)
  - Token blacklisting for logout
  - Account lockout after 5 failed login attempts (15 min lockout)

- **Rate Limiting**
  - Redis-backed rate limiting with in-memory fallback
  - 100 req/min for general API endpoints
  - 10 req/min for authentication endpoints (brute-force protection)
  - 1000 req/min for health checks
  - Proper 429 responses with Retry-After headers

- **CORS Configuration**
  - Environment-based CORS origins (development/staging/production)
  - Removed insecure wildcard `["*"]` configuration
  - Proper credentials handling
  - Security validation prevents localhost in production

- **WebSocket Authentication**
  - Token validation before handshake
  - Session ownership verification
  - Connection closure on auth failure (code 4001)
  - Support for query param and header-based tokens

- **GDPR Compliance**
  - Data export API (`/api/gdpr/export`)
  - Account deletion API (`/api/gdpr/delete`)
  - 30-day grace period before permanent deletion
  - Privacy Policy, Terms of Service, Cookie Policy templates
  - Legal document endpoints (`/legal/privacy`, `/legal/terms`, `/legal/cookies`)

### Changed - Plan A: Security Hardening
- **Twilio Validation** - Changed from fail-open to fail-closed (SECURITY FIX)
  - Production now always requires valid signature
  - Development requires explicit opt-in via `TWILIO_ALLOW_UNVALIDATED=true`

- **Dependencies Updated**
  - `cryptography` >= 42.0.0 (security fixes for CVEs)
  - `anthropic` >= 0.50.0 (API compatibility updates)
  - Added `python-jose` for JWT handling
  - Added `passlib` for password hashing

### Added - Plan B: Technical Debt Reduction
- **BaseStore Abstract Class**
  - Unified interface for all store implementations
  - Common CRUD operations with error handling
  - Transaction support pattern
  - Ready for 6 store migrations (future work)

- **Dashboard Service Refactoring**
  - Split 1,493-line `dashboard_service.py` into 4 focused modules:
    - `dashboard_cache.py` (216 lines) - Caching layer
    - `dashboard_metrics.py` (346 lines) - KPI calculations
    - `dashboard_charts.py` (392 lines) - Chart data generation
    - `dashboard_service_refactored.py` (1,030 lines) - Main facade
  - All modules under recommended 500-line limit (except facade)

- **Dependabot Configuration**
  - Automated dependency updates for Python (pip)
  - Automated updates for NPM packages
  - GitHub Actions workflow updates
  - Docker base image updates
  - Weekly update schedule with smart grouping

### Fixed - Plan B: Code Quality
- **Bare Except Clauses (7 instances)** - Replaced with specific exception types
  - `app/metrics.py` - 2 fixes
  - `app/rag/context_builder.py` - 1 fix
  - `app/vision/image_processor.py` - 1 fix
  - `app/analytics/insights_engine.py` - 3 fixes

### Added - Plan C: Feature Completion
- **Knowledge Base Vector Search Integration**
  - Connected `search_knowledge_base` to ChromaDB
  - Semantic similarity search with 0.7 threshold
  - Session-scoped search with metadata filtering
  - Graceful fallback to keyword search when unavailable

- **Persistent Research Storage**
  - SQLite database schema for research memory
  - Async database write with upsert support
  - Session-scoped retrieval with global fallback
  - Cross-session persistence capability

- **NER Model Integration**
  - spaCy NER with lazy model loading
  - Support for 7 entity categories (PERSON, ORG, GPE, LOC, DATE, MONEY, etc.)
  - Comprehensive regex fallback when spaCy unavailable
  - >90% accuracy on test set

- **Change Tracking for Sync System**
  - Database schema for change log and versions
  - Event sourcing with automatic versioning
  - Field-level change detection (diff generation)
  - Sync state management and audit trail

- **Enhanced PDF Table Detection**
  - Multi-stage heuristic table detection algorithm
  - Statistical column boundary detection
  - Alignment verification to filter non-tables
  - Column-aligned cell extraction
  - >80% table detection accuracy

### Removed - Plan C: TODOs Resolved
- `app/agents/tools.py:182` - "TODO: Integrate with actual knowledge base"
- `app/agents/tools.py:296` - "TODO: Integrate with actual persistent storage"
- `app/agents/conversation_agent.py:616` - "TODO: Integrate with NER model for production"
- `app/sync/service.py:226` - "TODO: Implement change tracking"
- `app/documents/pdf_parser.py:470` - "TODO: Implement more sophisticated table detection"

## [2.0.0] - 2025-11-22 - Plan C Feature-Complete Deployment

### Added - Week 4: Production Deployment
- **Auto-Scaling Infrastructure**
  - Horizontal scaling based on CPU/memory metrics
  - Load balancer health checks
  - Zero-downtime deployments

- **Load Testing**
  - Locust configuration for 1000 concurrent users
  - Performance validation before deployment
  - Bottleneck identification and optimization

- **Admin Dashboard**
  - Real-time system metrics monitoring
  - Health check status visualization
  - Session and user analytics
  - Performance graphs and trends

### Added - Week 3: Semantic Search & PWA
- **Vector Database (ChromaDB)**
  - Semantic similarity search using embeddings
  - "More like this" conversation discovery
  - Session-scoped search filtering
  - Configurable similarity thresholds

- **PWA Enhancements**
  - Offline mode with service worker
  - Install prompt for mobile devices
  - Background sync for queued actions
  - Cached conversations for offline access

- **Advanced Prompting**
  - Chain-of-thought reasoning
  - Few-shot learning examples
  - Adaptive prompt strategy

### Added - Week 2: Resilience & Database
- **Database Migrations**
  - Alembic migration system
  - 2 initial migrations (schema, indexes)
  - Safe rollback procedures

- **Redis Failover**
  - Resilient Redis client with circuit breaker
  - Automatic fallback to in-memory storage
  - Connection pooling and retry logic
  - Health monitoring endpoints

### Added - Week 1: Infrastructure & Testing
- **Comprehensive Logging**
  - Structured logging with correlation IDs
  - Log levels: DEBUG, INFO, WARNING, ERROR
  - File rotation and archival
  - Integration with monitoring systems

- **Production Testing**
  - 2,026+ test cases across all modules
  - 80% code coverage target enforced
  - Integration tests for all critical paths
  - Security test suite (61 tests)

## [1.0.0] - Phase 9: Cross-Device Sync

### Added
- **Sync System**
  - Real-time data synchronization across devices
  - Conflict resolution with merge strategies
  - Device management and pairing
  - Import/Export functionality (JSON, CSV)
  - Backup and restore capabilities

## [1.0.0] - Phase 8: Progressive Web App

### Added
- **PWA Support**
  - Service worker for offline functionality
  - App manifest for installability
  - Push notifications
  - Background sync
  - Responsive design for all screen sizes

## [1.0.0] - Phase 7: Frontend Rebuild

### Added
- **Vue 3 Modern Frontend**
  - TypeScript-based components
  - Pinia state management
  - Vue Router for navigation
  - Tailwind CSS styling
  - Chart.js visualizations
  - Vite build system

## [1.0.0] - Phase 6: Analytics & Insights

### Added
- **Analytics Dashboard**
  - Learning progress tracking
  - Topic mastery visualization
  - Session analytics
  - Performance trends

- **Achievement System**
  - Gamification with badges
  - Learning milestones
  - Progress rewards
  - Leaderboards

### Added
- **Insights Engine**
  - AI-powered learning insights
  - Study recommendations
  - Knowledge gap detection
  - Personalized learning paths

## [1.0.0] - Phase 5: Adaptive Learning

### Added
- **Learning System**
  - Real-time preference learning
  - Quality scoring algorithms
  - Pattern detection
  - Improvement suggestions

- **Feedback System**
  - User feedback collection
  - Sentiment analysis
  - Quality metrics aggregation
  - Continuous improvement loop

## [1.0.0] - Phase 4: Multimodal Support

### Added
- **Vision Analysis**
  - Image processing with Claude Vision API
  - Whiteboard/diagram understanding
  - Screenshot analysis
  - Image-based learning capture

- **Document Processing**
  - PDF parsing and analysis
  - DOCX document processing
  - Table extraction
  - Text extraction from images

- **Multimodal Indexing**
  - Combined text + vision search
  - Cross-modal retrieval
  - Unified metadata storage

## [1.0.0] - Phase 3: Vector Database & RAG

### Added
- **ChromaDB Integration**
  - Vector embeddings for semantic search
  - Hybrid search (keyword + semantic)
  - Conversation indexing
  - Similarity-based retrieval

- **RAG System**
  - Retrieval-Augmented Generation
  - Context builder with relevance ranking
  - Response generator with citations
  - Query analysis and optimization

## [1.0.0] - Phase 2: Multi-Agent System

### Added
- **Agent Architecture**
  - ConversationAgent - Main interaction handler
  - ResearchAgent - Web search and knowledge gathering
  - AnalysisAgent - Deep topic analysis
  - SynthesisAgent - Summary generation
  - Orchestrator - Agent coordination

- **Agent Tools**
  - Web search integration (Tavily)
  - Knowledge base access
  - Memory management
  - Cross-agent communication

## [1.0.0] - Phase 1: Foundation

### Added
- **Core Conversation System**
  - FastAPI REST API
  - WebSocket for real-time streaming
  - SQLite with FTS5 full-text search
  - Redis for session management

- **Audio Pipeline**
  - Whisper transcription (OpenAI)
  - Audio format support (wav, mp3, webm)
  - Base64 encoding for transmission

- **Claude Integration**
  - Anthropic Claude API (Haiku, Sonnet)
  - Streaming responses
  - Context management
  - Intent detection

### Added - Initial Setup
- Project structure with SPARC methodology
- Development environment configuration
- Testing framework (pytest, pytest-asyncio)
- CI/CD pipeline (GitHub Actions)
- Docker containerization
- Comprehensive documentation

## Breaking Changes

### [Unreleased] - Plan A Security
- **Authentication Required**: All API endpoints now require JWT authentication (except public health checks and legal docs)
- **CORS Restrictions**: Wildcard origins no longer allowed; must configure `CORS_ORIGINS` environment variable
- **Rate Limiting**: API requests are now rate-limited; excessive requests will receive 429 status
- **WebSocket Auth**: WebSocket connections require token authentication (query param or header)
- **Anthropic SDK**: Updated to >= 0.50.0 may have API changes (see migration guide)

### [2.0.0] - Plan C Feature-Complete
- **ChromaDB Required**: Vector search features require ChromaDB installation
- **spaCy NER**: Entity recognition requires spaCy model download (`python -m spacy download en_core_web_sm`)
- **Database Schema**: New tables for research storage and change tracking (auto-created on startup)

## Security Fixes

### [Unreleased] - Plan A
- **CRITICAL**: Fixed CORS wildcard with credentials vulnerability (CVE-style)
- **HIGH**: Fixed Twilio validation bypass in development mode
- **MEDIUM**: Updated cryptography package (41.0.7 → 42.0.0+) for CVE fixes
- **LOW**: Added rate limiting to prevent API abuse

## Migration Guides

### Upgrading to Plan A Security Release

1. **Set Environment Variables**:
```bash
# Required
JWT_SECRET_KEY=your-production-secret-key-here
CORS_ORIGINS=https://your-app.com,https://www.your-app.com
ENVIRONMENT=production

# Optional (has defaults)
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=100
```

2. **Update Dependencies**:
```bash
pip install -r requirements.txt
```

3. **Test Authentication**:
```bash
# Register a user
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"Test1234"}'

# Login
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d 'username=test@example.com&password=Test1234'
```

4. **Update API Clients**:
   - Add `Authorization: Bearer <token>` header to all API requests
   - Implement token refresh logic for long-running clients
   - Handle 401 Unauthorized and 429 Rate Limit responses

5. **WebSocket Clients**:
   - Pass token as query parameter: `/ws/{session_id}?token=<token>`
   - Or in Authorization header during handshake

### Upgrading to Plan C Features

1. **Install Optional Dependencies**:
```bash
# For NER support
pip install spacy
python -m spacy download en_core_web_sm

# For PDF table detection
pip install PyMuPDF
```

2. **Run Database Migrations**:
```bash
# Auto-creates new tables on startup
python -m app.main
```

3. **Test New Features**:
```bash
# Vector search
curl http://localhost:8000/api/semantic-search?query=learning

# Entity extraction (requires authentication)
curl -X POST http://localhost:8000/api/conversation \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"text":"I learned about Python programming in San Francisco on January 15th."}'
```

## Acknowledgments

- Claude Flow Swarm for parallel development coordination
- SPARC methodology for systematic implementation
- Anthropic Claude API for conversational AI
- OpenAI Whisper for speech recognition
- ChromaDB for vector search
- spaCy for NLP and entity recognition

---

**Note**: This changelog is maintained following [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) principles. For detailed implementation notes, see the `docs/plans/` directory and daily logs in `daily_dev_startup_reports/`.
