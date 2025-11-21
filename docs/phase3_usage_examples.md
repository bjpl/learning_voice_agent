# Phase 3: Knowledge Graph - Usage Examples

## Quick Start

### 1. Basic Setup

```python
import asyncio
from app.knowledge_graph import (
    graph_store,
    concept_extractor,
    query_engine
)

async def initialize():
    """Initialize knowledge graph"""
    await graph_store.initialize()
    print("Knowledge graph ready!")

# Run initialization
asyncio.run(initialize())
```

### 2. Extract Concepts from Text

```python
async def extract_example():
    """Extract concepts from learning text"""

    text = """
    I'm learning about machine learning and deep learning.
    Neural networks are fundamental to deep learning.
    Python and TensorFlow are commonly used tools.
    """

    result = await concept_extractor.extract_from_text(text)

    print(f"Found {result['stats']['concept_count']} concepts:")
    for concept in result['concepts']:
        print(f"  - {concept}")

asyncio.run(extract_example())
```

## Advanced Queries

### 3. Find Related Concepts

```python
related = await query_engine.find_related_concepts(
    concept="machine learning",
    max_depth=2,
    min_strength=0.3,
    limit=10
)

for item in related[:5]:
    print(f"{item['name']} - Relevance: {item['relevance_score']}")
```

### 4. Discover Learning Paths

```python
path = await query_engine.get_learning_path(
    from_concept="python basics",
    to_concept="deep learning"
)

if path['found']:
    print(f"Steps: {path['steps']}, Difficulty: {path['difficulty']}")
    for step in path['detailed_steps']:
        print(f"  {step['step']}. {step['concept']}")
```

### 5. Identify Knowledge Gaps

```python
gaps = await query_engine.identify_knowledge_gaps(
    known_concepts=["python", "statistics"],
    target_concept="deep learning"
)

print(f"Gaps found: {gaps['gap_count']}")
for gap in gaps['gaps'][:3]:
    print(f"  - {gap['concept']} (priority: {gap['priority_score']})")

for rec in gaps['recommendations']:
    print(f"  • {rec}")
```

For complete examples and integration patterns, see the full documentation.
