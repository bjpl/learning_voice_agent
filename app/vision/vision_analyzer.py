"""
Vision Analyzer using Claude 3.5 Sonnet

SPECIFICATION:
- Image analysis using Claude 3.5 Sonnet vision capabilities
- Support for multiple image formats and operations
- OCR via vision, diagram analysis, image comparison
- Structured JSON responses
- Error handling and retry logic
- Async API calls

ARCHITECTURE:
- Anthropic Claude API integration
- Image preprocessing pipeline
- Prompt template system
- Response parsing and validation
- Result caching

WHY:
- Leverage Claude's vision for educational content analysis
- Extract learning from visual materials
- Support multimodal learning experiences
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path

from anthropic import AsyncAnthropic
import json

from app.config import settings
from .config import vision_config, get_vision_config
from .image_processor import ImageProcessor, ImageValidationError, ImageProcessingError
from .prompts import (
    VisionTask,
    VisionPromptTemplates,
    VisionPromptBuilder,
    get_prompt_for_task,
    create_custom_prompt,
)


logger = logging.getLogger(__name__)


class VisionAnalysisError(Exception):
    """Raised when vision analysis fails"""
    pass


class VisionAnalyzer:
    """
    Vision analyzer using Claude 3.5 Sonnet

    PATTERN: Facade pattern wrapping Claude Vision API
    WHY: Simple interface for complex vision analysis

    Example:
        analyzer = VisionAnalyzer()
        result = await analyzer.analyze_image("path/to/image.jpg")
        description = result["analysis"]
    """

    def __init__(self, api_key: Optional[str] = None, config=None):
        """
        Initialize vision analyzer

        Args:
            api_key: Optional Anthropic API key override
            config: Optional VisionConfig override
        """
        self.api_key = api_key or settings.anthropic_api_key
        self.client = AsyncAnthropic(api_key=self.api_key)
        self.config = config or vision_config
        self.processor = ImageProcessor(config=self.config)
        self._cache: Dict[str, any] = {}

    # ========== CORE ANALYSIS METHODS ==========

    async def analyze_image(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_cache: bool = True,
    ) -> Dict[str, any]:
        """
        Analyze image using Claude 3.5 Sonnet vision

        PATTERN: Main entry point for vision analysis
        WHY: Flexible image analysis with customization

        Args:
            image_path: Path to image file
            prompt: Custom prompt (uses default if None)
            max_tokens: Max tokens for response
            temperature: Temperature for generation
            use_cache: Use cached results if available

        Returns:
            Dictionary with analysis results:
            {
                "analysis": "Analysis text or JSON",
                "metadata": {...},
                "tokens_used": int,
                "model": "model_name",
                "cached": bool
            }

        Raises:
            VisionAnalysisError: If analysis fails
        """
        try:
            # Process image
            processed = self.processor.process_image(image_path)

            # Check cache
            cache_key = self._get_cache_key(processed["hash"], prompt)
            if use_cache and self.config.enable_cache and cache_key in self._cache:
                logger.info(f"Using cached result for {image_path}")
                cached_result = self._cache[cache_key]
                cached_result["cached"] = True
                return cached_result

            # Prepare prompt
            if prompt is None:
                prompt = VisionPromptTemplates.DESCRIBE_IMAGE

            # Call Claude Vision API
            result = await self._call_vision_api(
                image_data=processed["image_data"],
                media_type=processed["media_type"],
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Add metadata
            result["metadata"] = processed["metadata"]
            result["cached"] = False

            # Cache result
            if self.config.enable_cache:
                self._cache[cache_key] = result

            return result

        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            raise VisionAnalysisError(f"Failed to analyze image: {str(e)}")

    async def describe_image(
        self,
        image_path: str,
        detailed: bool = True,
        include_metadata: bool = True,
    ) -> Dict[str, any]:
        """
        Generate detailed description of image

        Args:
            image_path: Path to image file
            detailed: Use detailed analysis prompt
            include_metadata: Include image metadata in response

        Returns:
            Dictionary with description and metadata
        """
        prompt = (
            VisionPromptTemplates.DESCRIBE_IMAGE
            if detailed
            else VisionPromptTemplates.QUICK_DESCRIBE
        )

        result = await self.analyze_image(image_path, prompt=prompt)

        # Parse JSON response if structured
        try:
            analysis = json.loads(result["analysis"])
            result["structured_analysis"] = analysis
        except json.JSONDecodeError:
            # Plain text response
            pass

        if not include_metadata:
            result.pop("metadata", None)

        return result

    async def extract_text_from_image(
        self,
        image_path: str,
        include_positions: bool = True,
    ) -> Dict[str, any]:
        """
        Extract text from image using OCR via vision

        PATTERN: Specialized OCR extraction
        WHY: Accurate text extraction with context

        Args:
            image_path: Path to image file
            include_positions: Include text position information

        Returns:
            Dictionary with extracted text and metadata:
            {
                "text": "All extracted text",
                "text_blocks": [...],
                "metadata": {...}
            }
        """
        prompt = VisionPromptTemplates.EXTRACT_TEXT

        result = await self.analyze_image(image_path, prompt=prompt)

        # Parse JSON response
        try:
            text_data = json.loads(result["analysis"])
            return {
                "success": True,
                "text": text_data.get("full_text", ""),
                "text_blocks": text_data.get("text_blocks", []),
                "metadata": text_data.get("metadata", {}),
                "image_metadata": result.get("metadata", {}),
                "tokens_used": result.get("tokens_used", 0),
            }
        except json.JSONDecodeError:
            # Fallback to plain text
            return {
                "success": True,
                "text": result["analysis"],
                "text_blocks": [],
                "metadata": {},
                "image_metadata": result.get("metadata", {}),
                "tokens_used": result.get("tokens_used", 0),
            }

    async def analyze_diagram(
        self,
        image_path: str,
        diagram_type: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Analyze technical diagram or chart

        Args:
            image_path: Path to diagram image
            diagram_type: Optional diagram type hint

        Returns:
            Dictionary with diagram analysis
        """
        prompt = VisionPromptTemplates.ANALYZE_DIAGRAM

        if diagram_type:
            prompt = f"This is a {diagram_type} diagram.\n\n{prompt}"

        result = await self.analyze_image(image_path, prompt=prompt)

        # Parse structured response
        try:
            diagram_data = json.loads(result["analysis"])
            return {
                "success": True,
                "diagram_type": diagram_data.get("diagram_type", "unknown"),
                "purpose": diagram_data.get("purpose", ""),
                "components": diagram_data.get("components", []),
                "connections": diagram_data.get("connections", []),
                "technical_details": diagram_data.get("technical_details", ""),
                "interpretation": diagram_data.get("interpretation", ""),
                "metadata": result.get("metadata", {}),
                "tokens_used": result.get("tokens_used", 0),
            }
        except json.JSONDecodeError:
            return {
                "success": True,
                "analysis": result["analysis"],
                "metadata": result.get("metadata", {}),
                "tokens_used": result.get("tokens_used", 0),
            }

    async def compare_images(
        self,
        image1_path: str,
        image2_path: str,
        focus: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Compare two images

        PATTERN: Multi-image analysis
        WHY: Identify similarities and differences

        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            focus: Optional focus area (visual, content, quality)

        Returns:
            Dictionary with comparison results
        """
        # Process both images
        processed1 = self.processor.process_image(image1_path)
        processed2 = self.processor.process_image(image2_path)

        # Prepare prompt
        prompt = VisionPromptTemplates.COMPARE_IMAGES
        if focus:
            prompt = f"Focus on {focus} aspects.\n\n{prompt}"

        # Call API with multiple images
        result = await self._call_vision_api_multi(
            images=[
                (processed1["image_data"], processed1["media_type"]),
                (processed2["image_data"], processed2["media_type"]),
            ],
            prompt=prompt,
        )

        # Parse response
        try:
            comparison = json.loads(result["analysis"])
            return {
                "success": True,
                "similarities": comparison.get("similarities", {}),
                "differences": comparison.get("differences", {}),
                "recommendation": comparison.get("recommendation", ""),
                "image1_metadata": processed1["metadata"],
                "image2_metadata": processed2["metadata"],
                "tokens_used": result.get("tokens_used", 0),
            }
        except json.JSONDecodeError:
            return {
                "success": True,
                "analysis": result["analysis"],
                "image1_metadata": processed1["metadata"],
                "image2_metadata": processed2["metadata"],
                "tokens_used": result.get("tokens_used", 0),
            }

    async def classify_image(self, image_path: str) -> Dict[str, any]:
        """
        Classify image into categories

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with classification results
        """
        result = await self.analyze_image(
            image_path,
            prompt=VisionPromptTemplates.CLASSIFY_IMAGE
        )

        try:
            classification = json.loads(result["analysis"])
            return {
                "success": True,
                "primary_category": classification.get("primary_category", ""),
                "secondary_categories": classification.get("secondary_categories", []),
                "tags": classification.get("tags", []),
                "content_type": classification.get("content_type", ""),
                "metadata": result.get("metadata", {}),
                "tokens_used": result.get("tokens_used", 0),
            }
        except json.JSONDecodeError:
            return {
                "success": True,
                "analysis": result["analysis"],
                "metadata": result.get("metadata", {}),
                "tokens_used": result.get("tokens_used", 0),
            }

    async def detect_objects(self, image_path: str) -> Dict[str, any]:
        """
        Detect and list objects in image

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with detected objects
        """
        result = await self.analyze_image(
            image_path,
            prompt=VisionPromptTemplates.DETECT_OBJECTS
        )

        try:
            detection = json.loads(result["analysis"])
            return {
                "success": True,
                "objects": detection.get("objects", []),
                "total_objects": detection.get("total_objects", 0),
                "dominant_objects": detection.get("dominant_objects", []),
                "metadata": result.get("metadata", {}),
                "tokens_used": result.get("tokens_used", 0),
            }
        except json.JSONDecodeError:
            return {
                "success": True,
                "analysis": result["analysis"],
                "metadata": result.get("metadata", {}),
                "tokens_used": result.get("tokens_used", 0),
            }

    # ========== CUSTOM ANALYSIS ==========

    async def custom_analysis(
        self,
        image_path: str,
        task: VisionTask,
        context: Optional[str] = None,
        constraints: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, any]:
        """
        Custom vision analysis with task-specific prompts

        Args:
            image_path: Path to image file
            task: Vision task type
            context: Optional context information
            constraints: Optional constraints
            **kwargs: Additional parameters

        Returns:
            Analysis results
        """
        # Build prompt
        prompt = get_prompt_for_task(
            task,
            context=context,
            constraints=constraints or []
        )

        return await self.analyze_image(image_path, prompt=prompt, **kwargs)

    # ========== BATCH PROCESSING ==========

    async def batch_analyze_images(
        self,
        image_paths: List[str],
        prompt: Optional[str] = None,
        max_concurrent: Optional[int] = None,
    ) -> List[Dict[str, any]]:
        """
        Analyze multiple images in batch

        Args:
            image_paths: List of image file paths
            prompt: Common prompt for all images
            max_concurrent: Maximum concurrent operations

        Returns:
            List of analysis results
        """
        max_concurrent = max_concurrent or self.config.max_concurrent_operations

        # Create tasks
        tasks = [
            self.analyze_image(path, prompt=prompt)
            for path in image_paths
        ]

        # Process in batches
        results = []
        for i in range(0, len(tasks), max_concurrent):
            batch = tasks[i:i + max_concurrent]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)

            for path, result in zip(image_paths[i:i + max_concurrent], batch_results):
                if isinstance(result, Exception):
                    results.append({
                        "success": False,
                        "path": path,
                        "error": str(result)
                    })
                else:
                    results.append({
                        "success": True,
                        "path": path,
                        "result": result
                    })

        return results

    # ========== API CALLS ==========

    async def _call_vision_api(
        self,
        image_data: str,
        media_type: str,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, any]:
        """
        Call Claude Vision API

        Args:
            image_data: Base64 encoded image
            media_type: Image media type
            prompt: Analysis prompt
            max_tokens: Max tokens for response
            temperature: Temperature for generation

        Returns:
            API response dictionary
        """
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature

        try:
            message = await asyncio.wait_for(
                self.client.messages.create(
                    model=self.config.vision_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }]
                ),
                timeout=self.config.analysis_timeout
            )

            return {
                "analysis": message.content[0].text,
                "tokens_used": message.usage.input_tokens + message.usage.output_tokens,
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
                "model": self.config.vision_model,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except asyncio.TimeoutError:
            raise VisionAnalysisError(
                f"Vision analysis timed out after {self.config.analysis_timeout}s"
            )
        except Exception as e:
            raise VisionAnalysisError(f"Vision API call failed: {str(e)}")

    async def _call_vision_api_multi(
        self,
        images: List[Tuple[str, str]],
        prompt: str,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, any]:
        """
        Call Claude Vision API with multiple images

        Args:
            images: List of (base64_data, media_type) tuples
            prompt: Analysis prompt
            max_tokens: Max tokens for response

        Returns:
            API response dictionary
        """
        if len(images) > self.config.max_images_per_request:
            raise VisionAnalysisError(
                f"Too many images: {len(images)} (max: {self.config.max_images_per_request})"
            )

        max_tokens = max_tokens or self.config.max_tokens

        # Build content with multiple images
        content = []
        for image_data, media_type in images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data,
                },
            })

        content.append({
            "type": "text",
            "text": prompt,
        })

        try:
            message = await asyncio.wait_for(
                self.client.messages.create(
                    model=self.config.vision_model,
                    max_tokens=max_tokens,
                    messages=[{
                        "role": "user",
                        "content": content,
                    }]
                ),
                timeout=self.config.analysis_timeout
            )

            return {
                "analysis": message.content[0].text,
                "tokens_used": message.usage.input_tokens + message.usage.output_tokens,
                "model": self.config.vision_model,
            }

        except asyncio.TimeoutError:
            raise VisionAnalysisError(
                f"Multi-image analysis timed out after {self.config.analysis_timeout}s"
            )
        except Exception as e:
            raise VisionAnalysisError(f"Multi-image API call failed: {str(e)}")

    # ========== UTILITIES ==========

    def _get_cache_key(self, image_hash: str, prompt: Optional[str]) -> str:
        """Generate cache key"""
        import hashlib
        prompt_hash = hashlib.md5((prompt or "").encode()).hexdigest()
        return f"{image_hash}:{prompt_hash}"

    def clear_cache(self) -> None:
        """Clear analysis cache"""
        self._cache.clear()
        logger.info("Vision analysis cache cleared")

    def get_cache_stats(self) -> Dict[str, any]:
        """Get cache statistics"""
        return {
            "size": len(self._cache),
            "enabled": self.config.enable_cache,
            "ttl": self.config.cache_ttl,
        }


# ========== HELPER FUNCTIONS ==========

async def quick_analyze(image_path: str, prompt: Optional[str] = None) -> str:
    """
    Quick image analysis with simple text response

    Args:
        image_path: Path to image file
        prompt: Optional custom prompt

    Returns:
        Analysis text
    """
    analyzer = VisionAnalyzer()
    result = await analyzer.analyze_image(image_path, prompt=prompt)
    return result["analysis"]


async def quick_describe(image_path: str) -> str:
    """Quick image description"""
    analyzer = VisionAnalyzer()
    result = await analyzer.describe_image(image_path, detailed=False)
    return result["analysis"]


async def quick_ocr(image_path: str) -> str:
    """Quick OCR text extraction"""
    analyzer = VisionAnalyzer()
    result = await analyzer.extract_text_from_image(image_path)
    return result["text"]
