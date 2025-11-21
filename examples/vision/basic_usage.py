"""
Vision Analysis Basic Usage Examples

SPECIFICATION:
- Demonstrate basic vision analysis capabilities
- Show common use cases
- Provide practical examples
- Educational context

ARCHITECTURE:
- Async examples with error handling
- Multiple analysis types
- Best practices demonstration

WHY:
- Help developers understand vision API
- Show integration patterns
- Provide working examples
"""
import asyncio
from pathlib import Path

from app.vision import (
    VisionAnalyzer,
    quick_analyze,
    quick_describe,
    quick_ocr,
)


async def example_basic_analysis():
    """
    Example: Basic image analysis

    Use case: Analyze educational diagrams or screenshots
    """
    print("=" * 60)
    print("EXAMPLE 1: Basic Image Analysis")
    print("=" * 60)

    analyzer = VisionAnalyzer()

    # Analyze an image
    result = await analyzer.analyze_image("examples/vision/sample_diagram.png")

    print(f"Analysis: {result['analysis']}")
    print(f"Tokens used: {result['tokens_used']}")
    print(f"Image metadata: {result['metadata']['size']}")
    print()


async def example_quick_helpers():
    """
    Example: Using quick helper functions

    Use case: Fast analysis without configuration
    """
    print("=" * 60)
    print("EXAMPLE 2: Quick Helper Functions")
    print("=" * 60)

    # Quick describe
    description = await quick_describe("examples/vision/sample_image.png")
    print(f"Quick Description: {description}\n")

    # Quick analyze with custom prompt
    analysis = await quick_analyze(
        "examples/vision/sample_image.png",
        prompt="What educational concepts are shown in this image?"
    )
    print(f"Custom Analysis: {analysis}\n")


async def example_ocr():
    """
    Example: Extract text from images

    Use case: Extract text from screenshots, slides, or documents
    """
    print("=" * 60)
    print("EXAMPLE 3: OCR - Text Extraction")
    print("=" * 60)

    analyzer = VisionAnalyzer()

    # Extract text from screenshot
    result = await analyzer.extract_text_from_image("examples/vision/screenshot.png")

    print(f"Extracted Text:\n{result['text']}\n")
    print(f"Text Blocks: {len(result['text_blocks'])}")

    # Show individual text blocks
    for i, block in enumerate(result['text_blocks'], 1):
        print(f"\nBlock {i}:")
        print(f"  Content: {block['content']}")
        print(f"  Position: {block['position']}")
        print(f"  Confidence: {block['confidence']}")


async def example_diagram_analysis():
    """
    Example: Analyze technical diagrams

    Use case: Understand flowcharts, architecture diagrams, UML
    """
    print("=" * 60)
    print("EXAMPLE 4: Diagram Analysis")
    print("=" * 60)

    analyzer = VisionAnalyzer()

    # Analyze flowchart
    result = await analyzer.analyze_diagram(
        "examples/vision/flowchart.png",
        diagram_type="flowchart"
    )

    print(f"Diagram Type: {result['diagram_type']}")
    print(f"Purpose: {result['purpose']}\n")

    print("Components:")
    for comp in result['components']:
        print(f"  - {comp['label']}: {comp['description']}")

    print("\nConnections:")
    for conn in result['connections']:
        print(f"  - {conn['from']} → {conn['to']}: {conn['label']}")


async def example_image_comparison():
    """
    Example: Compare two images

    Use case: Compare versions, identify changes, analyze differences
    """
    print("=" * 60)
    print("EXAMPLE 5: Image Comparison")
    print("=" * 60)

    analyzer = VisionAnalyzer()

    # Compare two versions of a diagram
    result = await analyzer.compare_images(
        "examples/vision/diagram_v1.png",
        "examples/vision/diagram_v2.png",
        focus="changes"
    )

    print("Similarities:")
    for sim in result['similarities']['visual']:
        print(f"  - {sim}")

    print("\nDifferences:")
    for diff in result['differences']['visual']:
        print(f"  - {diff}")

    print(f"\nRecommendation: {result['recommendation']}")


async def example_batch_processing():
    """
    Example: Process multiple images in batch

    Use case: Analyze entire slide decks or image sets
    """
    print("=" * 60)
    print("EXAMPLE 6: Batch Processing")
    print("=" * 60)

    analyzer = VisionAnalyzer()

    # Analyze multiple slides
    image_paths = [
        "examples/vision/slide_1.png",
        "examples/vision/slide_2.png",
        "examples/vision/slide_3.png",
    ]

    results = await analyzer.batch_analyze_images(
        image_paths,
        prompt="Summarize the main educational concepts in this slide."
    )

    for i, result in enumerate(results, 1):
        if result['success']:
            print(f"\nSlide {i}:")
            print(f"  Summary: {result['result']['analysis'][:100]}...")
        else:
            print(f"\nSlide {i}: Failed - {result['error']}")


async def example_classification():
    """
    Example: Classify images into categories

    Use case: Organize educational materials
    """
    print("=" * 60)
    print("EXAMPLE 7: Image Classification")
    print("=" * 60)

    analyzer = VisionAnalyzer()

    # Classify educational image
    result = await analyzer.classify_image("examples/vision/sample_image.png")

    print(f"Primary Category: {result['primary_category']}")
    print(f"Secondary Categories: {', '.join(result['secondary_categories'])}")
    print(f"Tags: {', '.join(result['tags'])}")
    print(f"Content Type: {result['content_type']}")


async def example_object_detection():
    """
    Example: Detect objects in images

    Use case: Identify educational tools, lab equipment, etc.
    """
    print("=" * 60)
    print("EXAMPLE 8: Object Detection")
    print("=" * 60)

    analyzer = VisionAnalyzer()

    # Detect objects
    result = await analyzer.detect_objects("examples/vision/lab_photo.png")

    print(f"Total Objects: {result['total_objects']}\n")

    print("Detected Objects:")
    for obj in result['objects']:
        print(f"  - {obj['name']} ({obj['size']})")
        print(f"    Location: {obj['location']}")
        print(f"    Count: {obj['count']}")


async def example_educational_analysis():
    """
    Example: Educational-focused analysis

    Use case: Analyze images for learning value
    """
    print("=" * 60)
    print("EXAMPLE 9: Educational Analysis")
    print("=" * 60)

    from app.vision.prompts import get_educational_prompt

    analyzer = VisionAnalyzer()

    # Analyze for educational value
    prompt = get_educational_prompt(
        subject_area="biology",
        grade_level="high school"
    )

    result = await analyzer.analyze_image(
        "examples/vision/biology_diagram.png",
        prompt=prompt
    )

    print(f"Educational Analysis:\n{result['analysis']}\n")


async def example_error_handling():
    """
    Example: Proper error handling

    Use case: Production-ready code with error management
    """
    print("=" * 60)
    print("EXAMPLE 10: Error Handling")
    print("=" * 60)

    from app.vision import VisionAnalysisError, ImageValidationError

    analyzer = VisionAnalyzer()

    # Handle validation errors
    try:
        result = await analyzer.analyze_image("nonexistent.png")
    except ImageValidationError as e:
        print(f"Validation Error: {e}")
    except VisionAnalysisError as e:
        print(f"Analysis Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")

    # Handle API errors with retry
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = await analyzer.analyze_image("examples/vision/sample.png")
            print("Analysis successful!")
            break
        except VisionAnalysisError as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed, retrying...")
                await asyncio.sleep(1)
            else:
                print(f"All attempts failed: {e}")


async def example_caching():
    """
    Example: Using caching for performance

    Use case: Optimize repeated analyses
    """
    print("=" * 60)
    print("EXAMPLE 11: Caching")
    print("=" * 60)

    analyzer = VisionAnalyzer()

    # First analysis (hits API)
    result1 = await analyzer.analyze_image("examples/vision/sample.png")
    print(f"First call - Cached: {result1.get('cached', False)}")

    # Second analysis (uses cache)
    result2 = await analyzer.analyze_image("examples/vision/sample.png")
    print(f"Second call - Cached: {result2.get('cached', False)}")

    # Check cache stats
    stats = analyzer.get_cache_stats()
    print(f"\nCache Stats: {stats}")

    # Clear cache if needed
    analyzer.clear_cache()
    print("Cache cleared")


async def main():
    """Run all examples"""
    print("\n")
    print("=" * 60)
    print("VISION ANALYSIS EXAMPLES")
    print("=" * 60)
    print("\n")

    # Note: Most examples will fail without actual image files
    # This is a demonstration of the API usage

    examples = [
        ("Basic Analysis", example_basic_analysis),
        ("Quick Helpers", example_quick_helpers),
        ("OCR", example_ocr),
        ("Diagram Analysis", example_diagram_analysis),
        ("Image Comparison", example_image_comparison),
        ("Batch Processing", example_batch_processing),
        ("Classification", example_classification),
        ("Object Detection", example_object_detection),
        ("Educational Analysis", example_educational_analysis),
        ("Error Handling", example_error_handling),
        ("Caching", example_caching),
    ]

    for name, example in examples:
        try:
            print(f"\nRunning: {name}")
            await example()
        except Exception as e:
            print(f"Example '{name}' failed: {e}")
            print("(This is expected without actual image files)\n")

    print("\n")
    print("=" * 60)
    print("EXAMPLES COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
