#!/usr/bin/env python3
"""
RuVector Integration Validation Script
PATTERN: Comprehensive validation for production deployment
WHY: Ensure all components are working before deployment

Usage:
    python scripts/validate_ruvector_integration.py

Returns exit code 0 if all validations pass, 1 otherwise.
"""
import sys
import os
import asyncio
from typing import Tuple, List

# Add project root to path for direct execution
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def print_header(title: str):
    """Print formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(name: str, success: bool, details: str = ""):
    """Print validation result."""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"  {status}: {name}")
    if details:
        print(f"         {details}")


async def validate_phase1_imports() -> Tuple[bool, List[str]]:
    """Validate Phase 1: Vector Store Replacement imports."""
    errors = []
    try:
        from app.vector import (
            VectorStoreProtocol,
            LearningVectorStoreProtocol,
            GraphVectorStoreProtocol,
            VectorStoreFactory,
            VectorBackend,
            DualWriteVectorStore,
            get_vector_store
        )
        from app.vector.ruvector_store import RuVectorStore, RUVECTOR_AVAILABLE
        return True, []
    except ImportError as e:
        errors.append(f"Import error: {e}")
        return False, errors


async def validate_phase2_imports() -> Tuple[bool, List[str]]:
    """Validate Phase 2: Knowledge Graph Unification imports."""
    errors = []
    try:
        from app.vector import (
            GraphQueryAdapter,
            NodeType,
            RelationshipType,
            ConversationNode,
            ConceptNode,
            CypherQueryBuilder
        )
        from app.vector.schema import TopicNode, EntityNode, Relationship
        return True, []
    except ImportError as e:
        errors.append(f"Import error: {e}")
        return False, errors


async def validate_phase3_imports() -> Tuple[bool, List[str]]:
    """Validate Phase 3: Learning Integration imports."""
    errors = []
    try:
        from app.learning.vector_feedback_bridge import VectorFeedbackBridge
        from app.learning.learning_metrics import LearningMetrics, get_learning_metrics
        return True, []
    except ImportError as e:
        errors.append(f"Import error: {e}")
        return False, errors


async def validate_phase4_imports() -> Tuple[bool, List[str]]:
    """Validate Phase 4: Semantic Agent Routing imports."""
    errors = []
    try:
        from app.agents.semantic_router import (
            SemanticAgentRouter,
            AgentCapability,
            get_semantic_router
        )
        return True, []
    except ImportError as e:
        errors.append(f"Import error: {e}")
        return False, errors


async def validate_vector_store_creation() -> Tuple[bool, List[str]]:
    """Validate vector store can be created."""
    errors = []
    try:
        from app.vector import VectorStoreFactory, get_vector_store

        # Test factory creation
        store = VectorStoreFactory.create(backend="auto")
        if store is None:
            errors.append("Factory returned None")
            return False, errors

        # Test convenience function
        store2 = get_vector_store()
        if store2 is None:
            errors.append("get_vector_store returned None")
            return False, errors

        return True, []
    except Exception as e:
        errors.append(f"Creation error: {e}")
        return False, errors


async def validate_protocol_compliance() -> Tuple[bool, List[str]]:
    """Validate RuVectorStore implements all protocols."""
    errors = []
    try:
        from app.vector import (
            VectorStoreProtocol,
            LearningVectorStoreProtocol,
            GraphVectorStoreProtocol
        )
        from app.vector.ruvector_store import RuVectorStore

        store = RuVectorStore()

        if not isinstance(store, VectorStoreProtocol):
            errors.append("Does not implement VectorStoreProtocol")
        if not isinstance(store, LearningVectorStoreProtocol):
            errors.append("Does not implement LearningVectorStoreProtocol")
        if not isinstance(store, GraphVectorStoreProtocol):
            errors.append("Does not implement GraphVectorStoreProtocol")

        return len(errors) == 0, errors
    except Exception as e:
        errors.append(f"Protocol check error: {e}")
        return False, errors


async def validate_config_settings() -> Tuple[bool, List[str]]:
    """Validate configuration settings exist."""
    errors = []
    try:
        from app.config import settings

        required_settings = [
            'vector_backend',
            'ruvector_persist_directory',
            'ruvector_enable_learning',
            'ruvector_enable_compression',
            'ruvector_gnn_enabled',
            'vector_ab_test_enabled',
            'enable_graph_queries'
        ]

        for setting in required_settings:
            if not hasattr(settings, setting):
                errors.append(f"Missing setting: {setting}")

        return len(errors) == 0, errors
    except Exception as e:
        errors.append(f"Config error: {e}")
        return False, errors


async def validate_semantic_router() -> Tuple[bool, List[str]]:
    """Validate semantic router functionality."""
    errors = []
    try:
        from app.agents.semantic_router import SemanticAgentRouter

        router = SemanticAgentRouter()
        await router.initialize()

        # Test routing
        agent, confidence = await router.route("Hello, how are you?")
        if agent is None:
            errors.append("Router returned None agent")
        if confidence < 0 or confidence > 1:
            errors.append(f"Invalid confidence: {confidence}")

        # Test get_top_agents
        top = await router.get_top_agents("Search for information", n=3)
        if len(top) == 0:
            errors.append("get_top_agents returned empty list")

        return len(errors) == 0, errors
    except Exception as e:
        errors.append(f"Router error: {e}")
        return False, errors


async def validate_learning_metrics() -> Tuple[bool, List[str]]:
    """Validate learning metrics functionality."""
    errors = []
    try:
        from app.learning.learning_metrics import LearningMetrics

        metrics = LearningMetrics()

        # Test recording
        await metrics.record_search_quality(
            query="test query",
            results=[{"id": "1"}, {"id": "2"}],
            user_selected_index=0
        )

        # Test stats
        stats = await metrics.get_improvement_stats()
        if not hasattr(stats, 'avg_mrr'):
            errors.append("Missing avg_mrr in stats")

        return len(errors) == 0, errors
    except Exception as e:
        errors.append(f"Metrics error: {e}")
        return False, errors


async def run_validation():
    """Run all validations."""
    print_header("RuVector Integration Validation")
    print(f"  Running comprehensive validation suite...")

    all_passed = True
    results = []

    # Phase 1 imports
    print_header("Phase 1: Vector Store Replacement")
    success, errors = await validate_phase1_imports()
    print_result("Core imports", success, "; ".join(errors) if errors else "")
    results.append(("Phase 1 imports", success))
    all_passed = all_passed and success

    # Phase 2 imports
    print_header("Phase 2: Knowledge Graph Unification")
    success, errors = await validate_phase2_imports()
    print_result("Graph imports", success, "; ".join(errors) if errors else "")
    results.append(("Phase 2 imports", success))
    all_passed = all_passed and success

    # Phase 3 imports
    print_header("Phase 3: Learning Integration")
    success, errors = await validate_phase3_imports()
    print_result("Learning imports", success, "; ".join(errors) if errors else "")
    results.append(("Phase 3 imports", success))
    all_passed = all_passed and success

    # Phase 4 imports
    print_header("Phase 4: Semantic Agent Routing")
    success, errors = await validate_phase4_imports()
    print_result("Router imports", success, "; ".join(errors) if errors else "")
    results.append(("Phase 4 imports", success))
    all_passed = all_passed and success

    # Functional validations
    print_header("Functional Validation")

    success, errors = await validate_vector_store_creation()
    print_result("Vector store creation", success, "; ".join(errors) if errors else "")
    results.append(("Store creation", success))
    all_passed = all_passed and success

    success, errors = await validate_protocol_compliance()
    print_result("Protocol compliance", success, "; ".join(errors) if errors else "")
    results.append(("Protocol compliance", success))
    all_passed = all_passed and success

    success, errors = await validate_config_settings()
    print_result("Config settings", success, "; ".join(errors) if errors else "")
    results.append(("Config settings", success))
    all_passed = all_passed and success

    success, errors = await validate_semantic_router()
    print_result("Semantic router", success, "; ".join(errors) if errors else "")
    results.append(("Semantic router", success))
    all_passed = all_passed and success

    success, errors = await validate_learning_metrics()
    print_result("Learning metrics", success, "; ".join(errors) if errors else "")
    results.append(("Learning metrics", success))
    all_passed = all_passed and success

    # Summary
    print_header("Validation Summary")
    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"  Passed: {passed}/{total}")
    print(f"  Status: {'✅ ALL VALIDATIONS PASSED' if all_passed else '❌ SOME VALIDATIONS FAILED'}")

    return all_passed


def main():
    """Main entry point."""
    try:
        success = asyncio.run(run_validation())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n  Validation cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ❌ Validation error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
