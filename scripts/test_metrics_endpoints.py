#!/usr/bin/env python3
"""
Quick test script to verify metrics endpoints are working

Run this script to test the observability features without starting the full server.
"""
import sys
import os

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.metrics import metrics_collector


def test_metrics_collector():
    """Test the MetricsCollector functionality"""
    print("Testing MetricsCollector...")

    # Test HTTP request tracking
    print("\n1. Testing HTTP request tracking...")
    metrics_collector.track_http_request(
        endpoint="/api/conversation",
        method="POST",
        status=200,
        duration=0.5
    )
    print("   ✓ HTTP request tracked")

    # Test Claude API tracking
    print("\n2. Testing Claude API tracking...")
    metrics_collector.track_claude_call(
        model="claude-3-haiku-20240307",
        duration=1.2,
        status="success",
        input_tokens=100,
        output_tokens=50
    )
    print("   ✓ Claude API call tracked")

    # Test Whisper API tracking
    print("\n3. Testing Whisper API tracking...")
    metrics_collector.track_whisper_call(
        duration=0.8,
        status="success",
        audio_duration=5.0
    )
    print("   ✓ Whisper API call tracked")

    # Test cache operations
    print("\n4. Testing cache operations...")
    metrics_collector.track_cache_operation("get", hit=True)
    metrics_collector.track_cache_operation("get", hit=False)
    metrics_collector.track_cache_operation("set", hit=False)
    print("   ✓ Cache operations tracked")

    # Test database queries
    print("\n5. Testing database query tracking...")
    metrics_collector.track_database_query(
        operation="select",
        table="exchanges",
        duration=0.005,
        status="success"
    )
    print("   ✓ Database query tracked")

    # Test conversation exchanges
    print("\n6. Testing conversation exchange tracking...")
    metrics_collector.track_conversation_exchange("question", quality_score=0.85)
    metrics_collector.track_conversation_exchange("statement", quality_score=0.75)
    print("   ✓ Conversation exchanges tracked")

    # Test audio processing
    print("\n7. Testing audio processing tracking...")
    metrics_collector.track_audio_processing(
        operation="transcribe",
        duration=0.8,
        format="wav",
        size_bytes=102400
    )
    print("   ✓ Audio processing tracked")

    # Test session updates
    print("\n8. Testing session tracking...")
    metrics_collector.update_active_sessions(5)
    metrics_collector.update_active_websockets(3)
    print("   ✓ Session metrics updated")

    # Get metrics in dictionary format
    print("\n9. Retrieving metrics dictionary...")
    metrics_dict = metrics_collector.get_metrics_dict()
    print(f"   ✓ Retrieved {len(metrics_dict)} metric categories")
    print(f"   - Application: {metrics_dict['application']}")
    print(f"   - Requests: {metrics_dict['requests']}")
    print(f"   - Sessions: {metrics_dict['sessions']}")
    print(f"   - Cache hit ratio: {metrics_dict['cache']['hit_ratio']:.2%}")
    print(f"   - Total cost: ${metrics_dict['costs']['total_usd']:.4f}")

    # Get Prometheus format
    print("\n10. Testing Prometheus format...")
    prometheus_data = metrics_collector.get_metrics()
    lines = prometheus_data.decode('utf-8').split('\n')
    metric_lines = [l for l in lines if l and not l.startswith('#')]
    print(f"   ✓ Generated {len(metric_lines)} metric lines")

    print("\n✅ All metrics collector tests passed!")
    return True


def print_sample_prometheus_output():
    """Print sample Prometheus metrics"""
    print("\n" + "="*70)
    print("SAMPLE PROMETHEUS METRICS OUTPUT")
    print("="*70)

    # Track some sample metrics
    metrics_collector.track_http_request("/api/conversation", "POST", 200, 0.5)
    metrics_collector.track_claude_call("claude-3-haiku", 1.2, "success", 100, 50)
    metrics_collector.update_active_sessions(3)

    prometheus_data = metrics_collector.get_metrics().decode('utf-8')

    # Print first 50 lines
    lines = prometheus_data.split('\n')[:50]
    for line in lines:
        print(line)

    print(f"\n... ({len(prometheus_data.split(chr(10))) - 50} more lines)")
    print("="*70)


if __name__ == "__main__":
    try:
        # Run tests
        success = test_metrics_collector()

        # Print sample output
        print_sample_prometheus_output()

        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print("✓ Metrics module is working correctly")
        print("✓ All metric types can be tracked")
        print("✓ Both JSON and Prometheus formats are supported")
        print("\nNext steps:")
        print("1. Start the application: uvicorn app.main:app --reload")
        print("2. Access metrics at: http://localhost:8000/metrics")
        print("3. Access health check: http://localhost:8000/api/health")
        print("4. View JSON metrics: http://localhost:8000/api/metrics")
        print("="*70)

        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
