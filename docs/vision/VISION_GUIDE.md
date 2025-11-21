# Vision Analysis System Guide

## Overview

The Vision Analysis System enables multimodal learning experiences by analyzing images, diagrams, screenshots, and other visual content using Claude 3.5 Sonnet's native vision capabilities.

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Components](#core-components)
5. [Usage Examples](#usage-examples)
6. [Configuration](#configuration)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)
9. [API Reference](#api-reference)

## Features

- **Image Analysis**: Comprehensive image description and analysis
- **OCR**: Text extraction from images with position tracking
- **Diagram Analysis**: Technical diagram interpretation (flowcharts, UML, architecture)
- **Image Comparison**: Compare two images for similarities and differences
- **Batch Processing**: Analyze multiple images concurrently
- **Classification**: Categorize images automatically
- **Object Detection**: Identify objects within images
- **Caching**: Built-in result caching for performance
- **Auto-resize**: Automatic image optimization for API limits

## Installation

### Requirements

```bash
# Install dependencies
pip install -r requirements.txt

# Vision-specific dependencies
pip install Pillow>=10.1.0
pip install python-magic>=0.4.27
```

### Environment Setup

```bash
# .env file
ANTHROPIC_API_KEY=your-api-key-here

# Optional: Vision configuration
VISION_MODEL=claude-3-5-sonnet-20241022
VISION_MAX_TOKENS=1024
VISION_TEMPERATURE=0.3
```

## Quick Start

### Basic Image Analysis

```python
from app.vision import VisionAnalyzer
import asyncio

async def analyze_image():
    analyzer = VisionAnalyzer()

    # Analyze an image
    result = await analyzer.analyze_image("path/to/image.png")

    print(result["analysis"])
    print(f"Tokens used: {result['tokens_used']}")

asyncio.run(analyze_image())
```

### Quick Helper Functions

```python
from app.vision import quick_describe, quick_ocr
import asyncio

async def quick_examples():
    # Quick description
    description = await quick_describe("diagram.png")
    print(description)

    # Quick OCR
    text = await quick_ocr("screenshot.png")
    print(text)

asyncio.run(quick_examples())
```

## Core Components

### 1. VisionAnalyzer

Main interface for vision analysis operations.

```python
from app.vision import VisionAnalyzer

analyzer = VisionAnalyzer()

# Available methods:
# - analyze_image()
# - describe_image()
# - extract_text_from_image()
# - analyze_diagram()
# - compare_images()
# - classify_image()
# - detect_objects()
# - batch_analyze_images()
```

### 2. ImageProcessor

Handles image preprocessing and validation.

```python
from app.vision import ImageProcessor

processor = ImageProcessor()

# Process and validate image
result = processor.process_image("image.png")

# Get image info
info = processor.get_image_info("image.png")

# Batch process
results = processor.batch_process_images(["img1.png", "img2.png"])
```

### 3. VisionConfig

Configuration management for vision system.

```python
from app.vision import vision_config, update_vision_config

# Get current config
config = vision_config

# Update config
update_vision_config(
    max_tokens=2048,
    temperature=0.5
)

# Use quality profiles
from app.vision import get_quality_profile

profile = get_quality_profile("quality")  # fast, balanced, quality
```

## Usage Examples

### 1. Educational Diagram Analysis

```python
from app.vision import VisionAnalyzer
from app.vision.prompts import get_educational_prompt

analyzer = VisionAnalyzer()

# Analyze biology diagram for high school
prompt = get_educational_prompt("biology", "high school")
result = await analyzer.analyze_image("cell_diagram.png", prompt=prompt)

print(result["analysis"])
```

### 2. Extract Text from Slides

```python
# Extract text from presentation slide
result = await analyzer.extract_text_from_image("slide.png")

print(f"Full Text: {result['text']}")

# Access structured text blocks
for block in result['text_blocks']:
    print(f"{block['position']}: {block['content']}")
```

### 3. Analyze Technical Diagram

```python
# Analyze flowchart
result = await analyzer.analyze_diagram("flowchart.png", diagram_type="flowchart")

print(f"Purpose: {result['purpose']}")

# List components
for component in result['components']:
    print(f"- {component['label']}: {component['description']}")

# List connections
for conn in result['connections']:
    print(f"- {conn['from']} → {conn['to']}")
```

### 4. Compare Images

```python
# Compare two versions
result = await analyzer.compare_images("v1.png", "v2.png")

print("Similarities:")
for sim in result['similarities']['visual']:
    print(f"  - {sim}")

print("\nDifferences:")
for diff in result['differences']['visual']:
    print(f"  - {diff}")
```

### 5. Batch Processing

```python
# Analyze multiple images
images = ["slide1.png", "slide2.png", "slide3.png"]
results = await analyzer.batch_analyze_images(
    images,
    prompt="Summarize the main concepts"
)

for i, result in enumerate(results, 1):
    if result['success']:
        print(f"Image {i}: {result['result']['analysis']}")
```

## Configuration

### Basic Configuration

```python
from app.vision import update_vision_config

update_vision_config(
    max_tokens=2048,           # Response length
    temperature=0.3,            # Creativity (0.0-1.0)
    auto_resize=True,          # Auto-resize large images
    enable_cache=True,         # Enable result caching
    cache_ttl=3600,            # Cache TTL in seconds
)
```

### Quality Profiles

```python
from app.vision import get_quality_profile, update_vision_config

# Fast processing (lower quality, faster)
fast_profile = get_quality_profile("fast")
update_vision_config(**fast_profile)

# Balanced (default)
balanced_profile = get_quality_profile("balanced")
update_vision_config(**balanced_profile)

# High quality (slower, best results)
quality_profile = get_quality_profile("quality")
update_vision_config(**quality_profile)
```

### Environment Variables

```bash
# Model Configuration
VISION_MODEL=claude-3-5-sonnet-20241022
VISION_MAX_TOKENS=1024
VISION_TEMPERATURE=0.3

# Image Processing
VISION_MAX_FILE_SIZE_MB=5.0
VISION_AUTO_RESIZE=true
VISION_RESIZE_QUALITY=85

# Performance
VISION_ENABLE_CACHE=true
VISION_CACHE_TTL=3600
VISION_MAX_CONCURRENT_OPERATIONS=3

# Storage
VISION_STORAGE_DIRECTORY=./data/vision_storage
VISION_CACHE_DIRECTORY=./data/vision_cache
```

## Best Practices

### 1. Image Preparation

```python
# Validate before analysis
from app.vision import ImageProcessor, ImageValidationError

processor = ImageProcessor()

try:
    validation = processor.validate_image("image.png")
    print(f"Image is valid: {validation['size']}")
except ImageValidationError as e:
    print(f"Invalid image: {e}")
```

### 2. Error Handling

```python
from app.vision import VisionAnalysisError, ImageValidationError

try:
    result = await analyzer.analyze_image("image.png")
except ImageValidationError as e:
    print(f"Image validation failed: {e}")
except VisionAnalysisError as e:
    print(f"Analysis failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### 3. Performance Optimization

```python
# Use caching for repeated analyses
analyzer = VisionAnalyzer()

# First call hits API
result1 = await analyzer.analyze_image("diagram.png")

# Second call uses cache
result2 = await analyzer.analyze_image("diagram.png")
assert result2["cached"] == True

# Clear cache when needed
analyzer.clear_cache()
```

### 4. Batch Processing Efficiently

```python
# Process in optimal batch sizes
from app.vision import vision_config

# Respect concurrency limits
images = ["img1.png", "img2.png", "img3.png", "img4.png"]
results = await analyzer.batch_analyze_images(
    images,
    max_concurrent=vision_config.max_concurrent_operations
)
```

### 5. Custom Prompts

```python
from app.vision.prompts import VisionPromptBuilder, VisionTask

# Build custom prompt
builder = VisionPromptBuilder()
prompt = builder \
    .set_task(VisionTask.ANALYZE) \
    .add_context("This is a learning material for students") \
    .add_constraint("Focus on educational value") \
    .add_instruction("Identify key learning concepts") \
    .build()

result = await analyzer.analyze_image("material.png", prompt=prompt)
```

## Troubleshooting

### Common Issues

#### 1. Image Too Large

```
Error: Image too large: 6.5MB (max: 5MB)
```

**Solution**: Enable auto-resize or manually resize

```python
# Enable auto-resize
from app.vision import update_vision_config
update_vision_config(auto_resize=True)

# Or manually resize
processor = ImageProcessor()
result = processor.process_image("large.png", auto_resize=True)
```

#### 2. API Timeout

```
Error: Vision analysis timed out after 60s
```

**Solution**: Increase timeout

```python
update_vision_config(analysis_timeout=120.0)
```

#### 3. Unsupported Format

```
Error: Unsupported format: BMP
```

**Solution**: Convert to supported format

```python
processor = ImageProcessor()
converted = processor.convert_to_format("image.bmp", "PNG")
```

### Debug Mode

```python
from app.vision import update_vision_config

# Enable detailed logging
update_vision_config(
    log_processing_details=True,
    save_debug_images=True,
    include_metadata=True
)
```

## API Reference

### VisionAnalyzer

#### analyze_image(image_path, prompt=None, max_tokens=None, temperature=None)

Analyze image with custom or default prompt.

**Parameters:**
- `image_path` (str): Path to image file
- `prompt` (str, optional): Custom analysis prompt
- `max_tokens` (int, optional): Max response tokens
- `temperature` (float, optional): Generation temperature

**Returns:** Dict with analysis, metadata, tokens_used

#### describe_image(image_path, detailed=True)

Generate detailed image description.

**Returns:** Dict with structured description

#### extract_text_from_image(image_path)

Extract text using OCR via vision.

**Returns:** Dict with text, text_blocks, metadata

#### analyze_diagram(image_path, diagram_type=None)

Analyze technical diagrams.

**Returns:** Dict with components, connections, interpretation

#### compare_images(image1_path, image2_path, focus=None)

Compare two images.

**Returns:** Dict with similarities, differences, recommendation

#### batch_analyze_images(image_paths, prompt=None, max_concurrent=None)

Batch process multiple images.

**Returns:** List of results

### ImageProcessor

#### validate_image(image_path)

Validate image file.

**Returns:** Validation result dict

#### process_image(image_path, auto_resize=None)

Process image for API.

**Returns:** Processed image data with base64 encoding

#### batch_process_images(image_paths)

Batch process images.

**Returns:** List of processed results

## Integration Examples

### With RAG System

```python
from app.vision import VisionAnalyzer
from app.rag import RAGEngine

analyzer = VisionAnalyzer()
rag = RAGEngine()

# Analyze diagram and store in RAG
result = await analyzer.analyze_diagram("architecture.png")
await rag.add_document(
    content=result["interpretation"],
    metadata={"type": "diagram", "source": "architecture.png"}
)
```

### With Conversation Agent

```python
from app.agents import ConversationAgent
from app.vision import VisionAnalyzer

agent = ConversationAgent()
analyzer = VisionAnalyzer()

# Analyze image shared in conversation
image_analysis = await analyzer.describe_image("shared_image.png")
response = await agent.process(
    message=f"I'm sharing this image: {image_analysis['analysis']}"
)
```

---

## Support

For issues or questions:
- GitHub Issues: [github.com/your-repo/issues](https://github.com/your-repo/issues)
- Documentation: [docs.example.com](https://docs.example.com)

## License

MIT License - See LICENSE file for details
