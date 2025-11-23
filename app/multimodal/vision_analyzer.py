"""
Vision Analyzer - Image Analysis with Claude Vision API

SPECIFICATION:
- Analyze images using Claude 3.5 Sonnet with vision
- Extract descriptions, objects, text, scenes
- Support custom analysis prompts
- Cache analysis results
- Handle various image formats

ARCHITECTURE:
- Async Claude API integration
- Base64 image encoding
- Structured analysis results
- Error handling and retries

PATTERN: Service class with Claude Vision API integration
WHY: Centralized image analysis with AI
RESILIENCE: Retry logic, error handling, fallback responses
"""

import base64
import asyncio
from typing import Dict, Optional, List
from pathlib import Path
from datetime import datetime

import anthropic
from app.logger import api_logger
from app.config import settings
from app.resilience import with_retry, with_timeout


class VisionAnalyzer:
    """
    Image analysis using Claude Vision API

    PATTERN: Service class with Claude Vision integration
    WHY: AI-powered image understanding
    """

    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        """
        Initialize vision analyzer

        Args:
            model: Claude model with vision capabilities
        """
        self.model = model
        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

        # Default analysis prompt
        self.default_prompt = """Analyze this image in detail. Provide:
1. A comprehensive description of what you see
2. Key objects, people, or elements present
3. Any text visible in the image
4. The overall scene or context
5. Notable details or interesting aspects

Be specific and thorough in your analysis."""

        api_logger.info(
            "vision_analyzer_initialized",
            model=self.model
        )

    @with_timeout(30)
    @with_retry(max_attempts=3, initial_wait=1.0)
    async def analyze_image(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        max_tokens: int = 1024
    ) -> Dict:
        """
        Analyze image using Claude Vision

        ALGORITHM:
        1. Read image file
        2. Encode to base64
        3. Call Claude Vision API
        4. Parse and structure response
        5. Return analysis results

        Args:
            image_path: Path to image file
            prompt: Custom analysis prompt (uses default if None)
            max_tokens: Maximum tokens for response

        Returns:
            Analysis results dictionary
        """
        start_time = datetime.utcnow()

        try:
            # Read and encode image
            image_data = await self._read_image(image_path)
            media_type = self._detect_media_type(image_path)

            # Prepare prompt
            analysis_prompt = prompt or self.default_prompt

            api_logger.info(
                "vision_analysis_started",
                image_path=image_path,
                media_type=media_type,
                prompt_length=len(analysis_prompt)
            )

            # Call Claude Vision API
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {
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
                                "text": analysis_prompt
                            }
                        ],
                    }
                ],
            )

            # Extract text response
            analysis_text = response.content[0].text

            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            api_logger.info(
                "vision_analysis_complete",
                image_path=image_path,
                response_length=len(analysis_text),
                processing_time_ms=round(processing_time, 2),
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens
            )

            # Structure results
            return {
                "success": True,
                "analysis": analysis_text,
                "model": self.model,
                "processing_time_ms": round(processing_time, 2),
                "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
                "timestamp": start_time.isoformat()
            }

        except anthropic.APIError as e:
            api_logger.error(
                "vision_api_error",
                image_path=image_path,
                error=str(e),
                error_type=type(e).__name__
            )
            return {
                "success": False,
                "error": f"Vision API error: {str(e)}",
                "timestamp": start_time.isoformat()
            }
        except Exception as e:
            api_logger.error(
                "vision_analysis_error",
                image_path=image_path,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            return {
                "success": False,
                "error": f"Analysis error: {str(e)}",
                "timestamp": start_time.isoformat()
            }

    async def analyze_images_batch(
        self,
        image_paths: List[str],
        prompt: Optional[str] = None
    ) -> List[Dict]:
        """
        Analyze multiple images in parallel

        Args:
            image_paths: List of image paths
            prompt: Custom analysis prompt

        Returns:
            List of analysis results
        """
        tasks = [
            self.analyze_image(path, prompt)
            for path in image_paths
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error dicts
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "image_path": image_paths[i]
                })
            else:
                processed_results.append(result)

        return processed_results

    async def extract_text_from_image(self, image_path: str) -> Dict:
        """
        Extract text from image (OCR)

        Args:
            image_path: Path to image

        Returns:
            Extracted text and metadata
        """
        prompt = """Extract and transcribe all visible text from this image.
Preserve the layout and formatting as much as possible.
If there is no text, respond with 'No text detected'."""

        return await self.analyze_image(image_path, prompt, max_tokens=2048)

    async def describe_for_blind_user(self, image_path: str) -> Dict:
        """
        Generate accessibility description

        Args:
            image_path: Path to image

        Returns:
            Accessibility-focused description
        """
        prompt = """Describe this image for a blind user. Include:
1. The main subject or focus
2. Important visual details and context
3. Spatial relationships between elements
4. Colors, if relevant to understanding
5. Any text present
6. The overall mood or atmosphere

Be clear, concise, and informative."""

        return await self.analyze_image(image_path, prompt, max_tokens=1024)

    async def _read_image(self, image_path: str) -> str:
        """
        Read and encode image to base64

        Args:
            image_path: Path to image file

        Returns:
            Base64 encoded image data
        """
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()

            return base64.standard_b64encode(image_bytes).decode('utf-8')

        except Exception as e:
            api_logger.error(
                "image_read_error",
                image_path=image_path,
                error=str(e)
            )
            raise

    def _detect_media_type(self, image_path: str) -> str:
        """
        Detect media type from file extension

        Args:
            image_path: Path to image

        Returns:
            MIME type string
        """
        extension = Path(image_path).suffix.lower()

        media_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }

        return media_types.get(extension, 'image/jpeg')


# Singleton instance
vision_analyzer = VisionAnalyzer()
