"""
Vision Analysis Module

SPECIFICATION:
- Claude 3.5 Sonnet vision capabilities
- Image analysis, OCR, diagram interpretation
- Image preprocessing and validation
- Batch processing support

ARCHITECTURE:
- VisionAnalyzer: Main analysis interface
- ImageProcessor: Image preparation
- VisionConfig: Configuration management
- Prompt templates: Task-specific prompts

WHY:
- Enable multimodal learning experiences
- Extract knowledge from visual content
- Support educational image analysis

Example:
    from app.vision import VisionAnalyzer

    analyzer = VisionAnalyzer()
    result = await analyzer.analyze_image("diagram.png")
    description = result["analysis"]

    # OCR
    text_result = await analyzer.extract_text_from_image("screenshot.png")
    text = text_result["text"]

    # Diagram analysis
    diagram = await analyzer.analyze_diagram("flowchart.png")
    components = diagram["components"]
"""

from .vision_analyzer import (
    VisionAnalyzer,
    VisionAnalysisError,
    quick_analyze,
    quick_describe,
    quick_ocr,
)
from .image_processor import (
    ImageProcessor,
    ImageValidationError,
    ImageProcessingError,
    quick_process_image,
    validate_and_prepare_image,
)
from .config import (
    VisionConfig,
    vision_config,
    get_vision_config,
    update_vision_config,
    get_quality_profile,
    validate_image_size,
    validate_image_dimensions,
)
from .prompts import (
    VisionTask,
    VisionPromptTemplates,
    VisionPromptBuilder,
    get_prompt_for_task,
    create_custom_prompt,
    get_educational_prompt,
    get_accessibility_prompt,
)

__all__ = [
    # Main classes
    "VisionAnalyzer",
    "ImageProcessor",
    "VisionConfig",
    # Exceptions
    "VisionAnalysisError",
    "ImageValidationError",
    "ImageProcessingError",
    # Quick functions
    "quick_analyze",
    "quick_describe",
    "quick_ocr",
    "quick_process_image",
    "validate_and_prepare_image",
    # Config
    "vision_config",
    "get_vision_config",
    "update_vision_config",
    "get_quality_profile",
    "validate_image_size",
    "validate_image_dimensions",
    # Prompts
    "VisionTask",
    "VisionPromptTemplates",
    "VisionPromptBuilder",
    "get_prompt_for_task",
    "create_custom_prompt",
    "get_educational_prompt",
    "get_accessibility_prompt",
]

# Version info
__version__ = "1.0.0"
__author__ = "Learning Voice Agent Team"
__description__ = "Vision analysis system using Claude 3.5 Sonnet"
