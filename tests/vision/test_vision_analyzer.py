"""
Tests for Vision Analyzer

SPECIFICATION:
- Test image analysis functionality
- Test OCR capabilities
- Test diagram analysis
- Test image comparison
- Test batch processing
- Test error handling

ARCHITECTURE:
- pytest-based async tests
- Mock API responses
- Test fixtures for images
- Coverage for all analysis methods

WHY:
- Ensure vision analysis works correctly
- Validate error handling
- Maintain code quality
"""
import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import json
import base64
from PIL import Image
import io

from app.vision import (
    VisionAnalyzer,
    VisionAnalysisError,
    quick_analyze,
    quick_describe,
    quick_ocr,
)
from app.vision.prompts import VisionTask


# ========== FIXTURES ==========

@pytest.fixture
def mock_image_path(tmp_path):
    """Create a temporary test image"""
    image_path = tmp_path / "test_image.png"
    img = Image.new('RGB', (800, 600), color='red')
    img.save(image_path)
    return str(image_path)


@pytest.fixture
def mock_diagram_path(tmp_path):
    """Create a temporary test diagram"""
    diagram_path = tmp_path / "test_diagram.png"
    img = Image.new('RGB', (1024, 768), color='blue')
    img.save(diagram_path)
    return str(diagram_path)


@pytest.fixture
def mock_api_response():
    """Mock Claude API response"""
    return {
        "content": [{
            "text": json.dumps({
                "main_subject": "Test image",
                "visual_elements": {
                    "colors": ["red", "blue"],
                    "composition": "centered",
                    "style": "digital"
                },
                "context": "Test context",
                "notable_details": ["detail1", "detail2"],
                "quality": {
                    "overall": "good",
                    "lighting": "bright",
                    "clarity": "sharp"
                }
            })
        }],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50
        }
    }


@pytest.fixture
def vision_analyzer():
    """Create VisionAnalyzer instance"""
    return VisionAnalyzer(api_key="test-key")


# ========== BASIC TESTS ==========

class TestVisionAnalyzerInit:
    """Test VisionAnalyzer initialization"""

    def test_init_with_default_config(self):
        """Test initialization with default config"""
        analyzer = VisionAnalyzer(api_key="test-key")
        assert analyzer.api_key == "test-key"
        assert analyzer.client is not None
        assert analyzer.processor is not None
        assert analyzer._cache == {}

    def test_init_with_custom_config(self):
        """Test initialization with custom config"""
        from app.vision.config import VisionConfig

        custom_config = VisionConfig(max_tokens=2048)
        analyzer = VisionAnalyzer(api_key="test-key", config=custom_config)
        assert analyzer.config.max_tokens == 2048


class TestImageAnalysis:
    """Test image analysis functionality"""

    @pytest.mark.asyncio
    async def test_analyze_image_success(self, vision_analyzer, mock_image_path, mock_api_response):
        """Test successful image analysis"""
        # Mock the API call
        with patch.object(
            vision_analyzer.client.messages,
            'create',
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = Mock(
                content=[Mock(text=mock_api_response["content"][0]["text"])],
                usage=Mock(
                    input_tokens=mock_api_response["usage"]["input_tokens"],
                    output_tokens=mock_api_response["usage"]["output_tokens"]
                )
            )

            result = await vision_analyzer.analyze_image(mock_image_path)

            assert "analysis" in result
            assert "metadata" in result
            assert "tokens_used" in result
            assert result["tokens_used"] == 150
            assert not result["cached"]

    @pytest.mark.asyncio
    async def test_analyze_image_with_cache(self, vision_analyzer, mock_image_path, mock_api_response):
        """Test image analysis with caching"""
        with patch.object(
            vision_analyzer.client.messages,
            'create',
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = Mock(
                content=[Mock(text=mock_api_response["content"][0]["text"])],
                usage=Mock(
                    input_tokens=100,
                    output_tokens=50
                )
            )

            # First call - should hit API
            result1 = await vision_analyzer.analyze_image(mock_image_path)
            assert not result1["cached"]
            assert mock_create.call_count == 1

            # Second call - should use cache
            result2 = await vision_analyzer.analyze_image(mock_image_path)
            assert result2["cached"]
            assert mock_create.call_count == 1  # No additional API call

    @pytest.mark.asyncio
    async def test_analyze_image_invalid_path(self, vision_analyzer):
        """Test analysis with invalid image path"""
        with pytest.raises(VisionAnalysisError):
            await vision_analyzer.analyze_image("nonexistent.png")

    @pytest.mark.asyncio
    async def test_describe_image(self, vision_analyzer, mock_image_path, mock_api_response):
        """Test image description"""
        with patch.object(
            vision_analyzer.client.messages,
            'create',
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = Mock(
                content=[Mock(text=mock_api_response["content"][0]["text"])],
                usage=Mock(input_tokens=100, output_tokens=50)
            )

            result = await vision_analyzer.describe_image(mock_image_path)

            assert "analysis" in result
            assert "structured_analysis" in result
            assert result["structured_analysis"]["main_subject"] == "Test image"


class TestOCR:
    """Test OCR functionality"""

    @pytest.mark.asyncio
    async def test_extract_text_from_image(self, vision_analyzer, mock_image_path):
        """Test text extraction from image"""
        ocr_response = json.dumps({
            "text_blocks": [
                {
                    "content": "Hello World",
                    "position": "top-left",
                    "style": "bold",
                    "confidence": "high"
                }
            ],
            "full_text": "Hello World",
            "metadata": {
                "total_blocks": 1,
                "language": "en",
                "readability": "high"
            }
        })

        with patch.object(
            vision_analyzer.client.messages,
            'create',
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = Mock(
                content=[Mock(text=ocr_response)],
                usage=Mock(input_tokens=100, output_tokens=50)
            )

            result = await vision_analyzer.extract_text_from_image(mock_image_path)

            assert result["success"]
            assert result["text"] == "Hello World"
            assert len(result["text_blocks"]) == 1
            assert result["text_blocks"][0]["content"] == "Hello World"

    @pytest.mark.asyncio
    async def test_extract_text_fallback(self, vision_analyzer, mock_image_path):
        """Test OCR fallback to plain text"""
        plain_text = "This is plain text response"

        with patch.object(
            vision_analyzer.client.messages,
            'create',
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = Mock(
                content=[Mock(text=plain_text)],
                usage=Mock(input_tokens=100, output_tokens=50)
            )

            result = await vision_analyzer.extract_text_from_image(mock_image_path)

            assert result["success"]
            assert result["text"] == plain_text
            assert result["text_blocks"] == []


class TestDiagramAnalysis:
    """Test diagram analysis"""

    @pytest.mark.asyncio
    async def test_analyze_diagram(self, vision_analyzer, mock_diagram_path):
        """Test diagram analysis"""
        diagram_response = json.dumps({
            "diagram_type": "flowchart",
            "purpose": "Process flow",
            "components": [
                {"type": "start", "label": "Begin", "description": "Start point"}
            ],
            "connections": [
                {"from": "Start", "to": "Process", "type": "arrow", "label": "next"}
            ],
            "technical_details": "Sequential process",
            "interpretation": "This shows a process flow"
        })

        with patch.object(
            vision_analyzer.client.messages,
            'create',
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = Mock(
                content=[Mock(text=diagram_response)],
                usage=Mock(input_tokens=100, output_tokens=50)
            )

            result = await vision_analyzer.analyze_diagram(mock_diagram_path)

            assert result["success"]
            assert result["diagram_type"] == "flowchart"
            assert len(result["components"]) == 1
            assert len(result["connections"]) == 1


class TestImageComparison:
    """Test image comparison"""

    @pytest.mark.asyncio
    async def test_compare_images(self, vision_analyzer, mock_image_path, mock_diagram_path):
        """Test comparing two images"""
        comparison_response = json.dumps({
            "similarities": {
                "visual": ["both are digital images"],
                "content": ["both contain shapes"]
            },
            "differences": {
                "visual": ["different colors"],
                "content": ["different subjects"]
            },
            "recommendation": "Use image1 for presentations"
        })

        with patch.object(
            vision_analyzer.client.messages,
            'create',
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = Mock(
                content=[Mock(text=comparison_response)],
                usage=Mock(input_tokens=200, output_tokens=100)
            )

            result = await vision_analyzer.compare_images(mock_image_path, mock_diagram_path)

            assert result["success"]
            assert "similarities" in result
            assert "differences" in result
            assert result["recommendation"] == "Use image1 for presentations"


class TestBatchProcessing:
    """Test batch image processing"""

    @pytest.mark.asyncio
    async def test_batch_analyze_images(self, vision_analyzer, tmp_path):
        """Test batch analysis of multiple images"""
        # Create multiple test images
        image_paths = []
        for i in range(3):
            path = tmp_path / f"image_{i}.png"
            img = Image.new('RGB', (100, 100), color='red')
            img.save(path)
            image_paths.append(str(path))

        with patch.object(
            vision_analyzer.client.messages,
            'create',
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = Mock(
                content=[Mock(text="Test analysis")],
                usage=Mock(input_tokens=100, output_tokens=50)
            )

            results = await vision_analyzer.batch_analyze_images(image_paths)

            assert len(results) == 3
            for result in results:
                assert result["success"]
                assert "path" in result
                assert "result" in result


class TestUtilityFunctions:
    """Test utility functions"""

    @pytest.mark.asyncio
    async def test_quick_analyze(self, mock_image_path):
        """Test quick_analyze helper"""
        with patch('app.vision.vision_analyzer.VisionAnalyzer') as mock_analyzer_class:
            mock_instance = Mock()
            mock_instance.analyze_image = AsyncMock(return_value={"analysis": "Quick test"})
            mock_analyzer_class.return_value = mock_instance

            result = await quick_analyze(mock_image_path)
            assert result == "Quick test"

    @pytest.mark.asyncio
    async def test_quick_describe(self, mock_image_path):
        """Test quick_describe helper"""
        with patch('app.vision.vision_analyzer.VisionAnalyzer') as mock_analyzer_class:
            mock_instance = Mock()
            mock_instance.describe_image = AsyncMock(return_value={"analysis": "Quick description"})
            mock_analyzer_class.return_value = mock_instance

            result = await quick_describe(mock_image_path)
            assert result == "Quick description"

    @pytest.mark.asyncio
    async def test_quick_ocr(self, mock_image_path):
        """Test quick_ocr helper"""
        with patch('app.vision.vision_analyzer.VisionAnalyzer') as mock_analyzer_class:
            mock_instance = Mock()
            mock_instance.extract_text_from_image = AsyncMock(return_value={"text": "Extracted text"})
            mock_analyzer_class.return_value = mock_instance

            result = await quick_ocr(mock_image_path)
            assert result == "Extracted text"


class TestCaching:
    """Test caching functionality"""

    @pytest.mark.asyncio
    async def test_cache_clear(self, vision_analyzer):
        """Test cache clearing"""
        vision_analyzer._cache = {"key1": "value1", "key2": "value2"}
        vision_analyzer.clear_cache()
        assert vision_analyzer._cache == {}

    def test_cache_stats(self, vision_analyzer):
        """Test cache statistics"""
        vision_analyzer._cache = {"key1": "value1", "key2": "value2"}
        stats = vision_analyzer.get_cache_stats()

        assert stats["size"] == 2
        assert "enabled" in stats
        assert "ttl" in stats


class TestErrorHandling:
    """Test error handling"""

    @pytest.mark.asyncio
    async def test_api_timeout(self, vision_analyzer, mock_image_path):
        """Test API timeout handling"""
        with patch.object(
            vision_analyzer.client.messages,
            'create',
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.side_effect = asyncio.TimeoutError()

            with pytest.raises(VisionAnalysisError, match="timed out"):
                await vision_analyzer.analyze_image(mock_image_path)

    @pytest.mark.asyncio
    async def test_api_error(self, vision_analyzer, mock_image_path):
        """Test API error handling"""
        with patch.object(
            vision_analyzer.client.messages,
            'create',
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.side_effect = Exception("API Error")

            with pytest.raises(VisionAnalysisError):
                await vision_analyzer.analyze_image(mock_image_path)


class TestCustomAnalysis:
    """Test custom analysis"""

    @pytest.mark.asyncio
    async def test_custom_analysis_with_task(self, vision_analyzer, mock_image_path):
        """Test custom analysis with vision task"""
        with patch.object(
            vision_analyzer.client.messages,
            'create',
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = Mock(
                content=[Mock(text="Custom analysis result")],
                usage=Mock(input_tokens=100, output_tokens=50)
            )

            result = await vision_analyzer.custom_analysis(
                mock_image_path,
                task=VisionTask.CLASSIFY,
                context="Educational image"
            )

            assert "analysis" in result
            assert result["analysis"] == "Custom analysis result"


# ========== INTEGRATION TESTS ==========

class TestIntegration:
    """Integration tests with real image processing"""

    @pytest.mark.asyncio
    async def test_full_workflow(self, tmp_path):
        """Test complete analysis workflow"""
        # Create test image
        image_path = tmp_path / "workflow_test.png"
        img = Image.new('RGB', (500, 500), color='green')
        img.save(image_path)

        analyzer = VisionAnalyzer(api_key="test-key")

        with patch.object(
            analyzer.client.messages,
            'create',
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = Mock(
                content=[Mock(text="Analysis complete")],
                usage=Mock(input_tokens=100, output_tokens=50)
            )

            # Process and analyze
            result = await analyzer.analyze_image(str(image_path))

            assert result["analysis"] == "Analysis complete"
            assert "metadata" in result
            assert result["metadata"]["format"] == "PNG"
            assert result["metadata"]["size"] == (500, 500)
