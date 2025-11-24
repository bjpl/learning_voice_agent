"""
Test fixtures for vision components
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from PIL import Image
import tempfile
from pathlib import Path

@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for vision analysis"""
    mock = AsyncMock()

    # Mock successful response
    mock_message = MagicMock()
    mock_message.content = [
        MagicMock(text="A diagram showing system architecture with API Gateway, microservices, and database.")
    ]
    mock_message.usage = MagicMock(input_tokens=1000, output_tokens=150)

    mock.messages.create = AsyncMock(return_value=mock_message)

    return mock

@pytest.fixture
def vision_analyzer(mock_anthropic_client):
    """VisionAnalyzer with mocked API"""
    from app.vision.vision_analyzer import VisionAnalyzer

    analyzer = VisionAnalyzer()
    analyzer.client = mock_anthropic_client
    return analyzer

@pytest.fixture
def sample_image(tmp_path):
    """Generate sample image for testing"""
    def _create_image(name="test.png", size=(800, 600), color="white"):
        img_path = tmp_path / name
        img = Image.new('RGB', size, color=color)
        img.save(img_path)
        return str(img_path)
    return _create_image

@pytest.fixture
def large_image(tmp_path):
    """Generate large image for resize testing"""
    img_path = tmp_path / "large.png"
    img = Image.new('RGB', (4000, 3000), color='blue')
    img.save(img_path)
    return str(img_path)

@pytest.fixture
def invalid_image(tmp_path):
    """Create invalid image file"""
    file_path = tmp_path / "invalid.png"
    file_path.write_text("This is not an image")
    return str(file_path)

@pytest.fixture
def image_with_text(tmp_path):
    """Create image with text for OCR testing"""
    from PIL import ImageDraw, ImageFont

    img_path = tmp_path / "screenshot.png"
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)

    # Add text
    draw.text((100, 100), "Login", fill='black')
    draw.text((100, 150), "Username: _______", fill='black')
    draw.text((100, 200), "Password: _______", fill='black')
    draw.text((100, 250), "[Submit]", fill='black')

    img.save(img_path)
    return str(img_path)
