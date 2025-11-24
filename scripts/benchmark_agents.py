#!/usr/bin/env python3
"""
Agent Performance Benchmark Script

SPECIFICATION:
- Benchmark AnalysisAgent and SynthesisAgent performance
- Measure processing time, throughput, accuracy
- Generate performance report

USAGE:
    python scripts/benchmark_agents.py

    # With custom iterations
    python scripts/benchmark_agents.py --iterations 100

WHY: Validate performance requirements (<500ms analysis, <800ms synthesis)
"""
import asyncio
import time
import statistics
import argparse
from typing import List, Dict, Any

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.agents.analysis_agent import AnalysisAgent
from app.agents.synthesis_agent import SynthesisAgent
from app.agents.base import AgentMessage, MessageType


# Test data
SAMPLE_TEXTS = [
    "I'm learning about machine learning and neural networks for computer vision",
    "Python programming is great for data science and artificial intelligence applications",
    "I want to understand distributed systems, microservices, and cloud architecture",
    "Studying algorithms and data structures to improve my problem-solving skills",
    "Exploring deep learning frameworks like TensorFlow and PyTorch for NLP tasks"
]

SAMPLE_CONVERSATIONS = [
    [
        {"user": "I'm interested in machine learning", "agent": "That's great!"},
        {"user": "Specifically neural networks", "agent": "Excellent choice!"},
        {"user": "Can you recommend resources?", "agent": "Sure! Start with..."}
    ],
    [
        {"user": "Learning Python programming", "agent": "Good choice!"},
        {"user": "For data science applications", "agent": "Very popular!"},
        {"user": "What libraries should I learn?", "agent": "Pandas, NumPy..."}
    ]
]


async def benchmark_analysis_agent(iterations: int = 50) -> Dict[str, Any]:
    """
    Benchmark AnalysisAgent performance

    Args:
        iterations: Number of benchmark iterations

    Returns:
        Performance metrics dictionary
    """
    print(f"\n{'='*60}")
    print("Benchmarking AnalysisAgent")
    print(f"{'='*60}")

    agent = AnalysisAgent(agent_id="benchmark_analysis")

    # Warm-up
    print("Warming up...")
    for text in SAMPLE_TEXTS[:2]:
        message = AgentMessage(
            sender="benchmark",
            recipient=agent.agent_id,
            message_type=MessageType.REQUEST,
            content={"action": "analyze_text", "text": text}
        )
        await agent.process(message)

    # Benchmark text analysis
    print(f"\nRunning {iterations} text analysis iterations...")
    timings = []

    for i in range(iterations):
        text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]

        message = AgentMessage(
            sender="benchmark",
            recipient=agent.agent_id,
            message_type=MessageType.REQUEST,
            content={"action": "analyze_text", "text": text}
        )

        start = time.perf_counter()
        result = await agent.process(message)
        duration_ms = (time.perf_counter() - start) * 1000

        timings.append(duration_ms)

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{iterations}")

    # Calculate statistics
    metrics = {
        "total_iterations": iterations,
        "mean_time_ms": statistics.mean(timings),
        "median_time_ms": statistics.median(timings),
        "min_time_ms": min(timings),
        "max_time_ms": max(timings),
        "stdev_time_ms": statistics.stdev(timings) if len(timings) > 1 else 0,
        "p95_time_ms": statistics.quantiles(timings, n=20)[18],  # 95th percentile
        "p99_time_ms": statistics.quantiles(timings, n=100)[98],  # 99th percentile
        "throughput_per_sec": 1000 / statistics.mean(timings),
        "meets_spec": statistics.median(timings) < 500  # < 500ms requirement
    }

    return metrics


async def benchmark_synthesis_agent(iterations: int = 30) -> Dict[str, Any]:
    """
    Benchmark SynthesisAgent performance

    Args:
        iterations: Number of benchmark iterations

    Returns:
        Performance metrics dictionary
    """
    print(f"\n{'='*60}")
    print("Benchmarking SynthesisAgent")
    print(f"{'='*60}")

    agent = SynthesisAgent(agent_id="benchmark_synthesis")

    # Sample analysis data
    sample_analysis = {
        "topics": [{"topic": "technology", "confidence": 0.9}],
        "concepts": ["machine learning", "neural networks", "algorithms"],
        "sentiment": {"polarity": 0.6, "label": "positive"},
        "keywords": [{"keyword": "learning", "frequency": 5}]
    }

    # Warm-up
    print("Warming up...")
    for _ in range(2):
        message = AgentMessage(
            sender="benchmark",
            recipient=agent.agent_id,
            message_type=MessageType.REQUEST,
            content={"action": "generate_insights", "analysis": sample_analysis, "history": SAMPLE_CONVERSATIONS[0]}
        )
        await agent.process(message)

    # Benchmark different synthesis operations
    operations = [
        "generate_insights",
        "create_summary",
        "recommend_topics",
        "create_schedule"
    ]

    results = {}

    for operation in operations:
        print(f"\nBenchmarking {operation}...")
        timings = []

        for i in range(iterations):
            content = {
                "analysis": sample_analysis,
                "history": SAMPLE_CONVERSATIONS[i % len(SAMPLE_CONVERSATIONS)],
                "exchanges": SAMPLE_CONVERSATIONS[i % len(SAMPLE_CONVERSATIONS)],
                "topics": sample_analysis["topics"],
                "concepts": sample_analysis["concepts"]
            }

            message = AgentMessage(
                sender="benchmark",
                recipient=agent.agent_id,
                message_type=MessageType.REQUEST,
                content={"action": operation, **content}
            )

            start = time.perf_counter()
            result = await agent.process(message)
            duration_ms = (time.perf_counter() - start) * 1000

            timings.append(duration_ms)

        results[operation] = {
            "mean_time_ms": statistics.mean(timings),
            "median_time_ms": statistics.median(timings),
            "min_time_ms": min(timings),
            "max_time_ms": max(timings),
            "meets_spec": statistics.median(timings) < 800  # < 800ms requirement
        }

    return results


def print_analysis_results(metrics: Dict[str, Any]):
    """Print AnalysisAgent benchmark results"""
    print(f"\n{'='*60}")
    print("AnalysisAgent Performance Results")
    print(f"{'='*60}")
    print(f"Total Iterations:    {metrics['total_iterations']}")
    print(f"Mean Time:          {metrics['mean_time_ms']:.2f} ms")
    print(f"Median Time:        {metrics['median_time_ms']:.2f} ms")
    print(f"Min Time:           {metrics['min_time_ms']:.2f} ms")
    print(f"Max Time:           {metrics['max_time_ms']:.2f} ms")
    print(f"Std Dev:            {metrics['stdev_time_ms']:.2f} ms")
    print(f"95th Percentile:    {metrics['p95_time_ms']:.2f} ms")
    print(f"99th Percentile:    {metrics['p99_time_ms']:.2f} ms")
    print(f"Throughput:         {metrics['throughput_per_sec']:.2f} ops/sec")
    print(f"\nMeets Spec (<500ms): {'✅ YES' if metrics['meets_spec'] else '❌ NO'}")


def print_synthesis_results(results: Dict[str, Dict[str, Any]]):
    """Print SynthesisAgent benchmark results"""
    print(f"\n{'='*60}")
    print("SynthesisAgent Performance Results")
    print(f"{'='*60}")

    for operation, metrics in results.items():
        print(f"\n{operation}:")
        print(f"  Mean Time:     {metrics['mean_time_ms']:.2f} ms")
        print(f"  Median Time:   {metrics['median_time_ms']:.2f} ms")
        print(f"  Min Time:      {metrics['min_time_ms']:.2f} ms")
        print(f"  Max Time:      {metrics['max_time_ms']:.2f} ms")
        print(f"  Meets Spec:    {'✅ YES' if metrics['meets_spec'] else '❌ NO'}")


async def main(args):
    """Main benchmark execution"""
    print("Agent Performance Benchmark")
    print("=" * 60)

    # Benchmark AnalysisAgent
    analysis_metrics = await benchmark_analysis_agent(iterations=args.iterations)
    print_analysis_results(analysis_metrics)

    # Benchmark SynthesisAgent
    synthesis_metrics = await benchmark_synthesis_agent(iterations=args.iterations // 2)
    print_synthesis_results(synthesis_metrics)

    # Overall summary
    print(f"\n{'='*60}")
    print("Overall Summary")
    print(f"{'='*60}")

    analysis_pass = analysis_metrics['meets_spec']
    synthesis_pass = all(m['meets_spec'] for m in synthesis_metrics.values())

    print(f"AnalysisAgent:   {'✅ PASS' if analysis_pass else '❌ FAIL'}")
    print(f"SynthesisAgent:  {'✅ PASS' if synthesis_pass else '❌ FAIL'}")
    print(f"\nOverall:         {'✅ ALL TESTS PASSED' if analysis_pass and synthesis_pass else '❌ SOME TESTS FAILED'}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark agent performance")
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of benchmark iterations (default: 50)"
    )

    args = parser.parse_args()
    asyncio.run(main(args))
