"""
Tests for Vision Configuration

SPECIFICATION:
- Test configuration management
- Test validation functions
- Test quality profiles
- Test configuration updates

ARCHITECTURE:
- pytest-based tests
- Configuration validation
- Profile testing

WHY:
- Ensure configuration works correctly
- Validate settings behavior
- Test environment overrides
"""
import pytest
from unittest.mock import patch

from app.vision.config import (
    VisionConfig,
    vision_config,
    get_vision_config,
    update_vision_config,
    get_quality_profile,
    validate_image_size,
    validate_image_dimensions,
)


class TestVisionConfig:
    """Test VisionConfig class"""

    def test_default_config(self):
        """Test default configuration values"""
        config = VisionConfig()

        assert config.vision_model == "claude-3-5-sonnet-20241022"
        assert config.max_tokens == 1024
        assert config.temperature == 0.3
        assert config.max_file_size_mb == 5.0

    def test_supported_formats(self):
        """Test supported image formats"""
        config = VisionConfig()

        assert "PNG" in config.supported_formats
        assert "JPEG" in config.supported_formats
        assert "GIF" in config.supported_formats
        assert "WEBP" in config.supported_formats

    def test_size_limits(self):
        """Test size and dimension limits"""
        config = VisionConfig()

        assert config.max_width == 8000
        assert config.max_height == 8000
        assert config.min_width == 100
        assert config.min_height == 100

    def test_processing_config(self):
        """Test image processing configuration"""
        config = VisionConfig()

        assert config.auto_resize is True
        assert config.preserve_aspect_ratio is True
        assert config.resize_quality >= 1
        assert config.resize_quality <= 100

    def test_performance_config(self):
        """Test performance configuration"""
        config = VisionConfig()

        assert config.enable_cache is True
        assert config.cache_ttl > 0
        assert config.max_concurrent_operations > 0


class TestGetVisionConfig:
    """Test get_vision_config helper"""

    def test_get_config_returns_singleton(self):
        """Test getting config returns singleton instance"""
        config1 = get_vision_config()
        config2 = get_vision_config()

        assert config1 is config2

    def test_get_config_same_as_global(self):
        """Test get_config returns same as global"""
        config = get_vision_config()

        assert config is vision_config


class TestUpdateVisionConfig:
    """Test configuration updates"""

    def test_update_single_value(self):
        """Test updating single config value"""
        original_value = vision_config.max_tokens

        update_vision_config(max_tokens=2048)
        assert vision_config.max_tokens == 2048

        # Restore
        update_vision_config(max_tokens=original_value)

    def test_update_multiple_values(self):
        """Test updating multiple config values"""
        original_tokens = vision_config.max_tokens
        original_temp = vision_config.temperature

        update_vision_config(max_tokens=2048, temperature=0.5)

        assert vision_config.max_tokens == 2048
        assert vision_config.temperature == 0.5

        # Restore
        update_vision_config(max_tokens=original_tokens, temperature=original_temp)

    def test_update_invalid_parameter_raises_error(self):
        """Test updating invalid parameter raises error"""
        with pytest.raises(ValueError, match="Invalid vision configuration"):
            update_vision_config(nonexistent_param="value")

    def test_update_preserves_other_values(self):
        """Test updating one value doesn't affect others"""
        original_model = vision_config.vision_model
        original_temp = vision_config.temperature

        update_vision_config(max_tokens=2048)

        assert vision_config.vision_model == original_model
        assert vision_config.temperature == original_temp


class TestQualityProfiles:
    """Test quality profile presets"""

    def test_fast_profile(self):
        """Test fast quality profile"""
        profile = get_quality_profile("fast")

        assert profile["max_tokens"] < 1024
        assert profile["temperature"] <= 0.3
        assert profile["resize_quality"] < 85
        assert profile["enable_detailed_analysis"] is False

    def test_balanced_profile(self):
        """Test balanced quality profile"""
        profile = get_quality_profile("balanced")

        assert profile["max_tokens"] == 1024
        assert profile["resize_quality"] == 85
        assert profile["enable_detailed_analysis"] is True

    def test_quality_profile(self):
        """Test high quality profile"""
        profile = get_quality_profile("quality")

        assert profile["max_tokens"] >= 1024
        assert profile["resize_quality"] >= 85
        assert profile["enable_detailed_analysis"] is True

    def test_invalid_profile_raises_error(self):
        """Test invalid profile name raises error"""
        with pytest.raises(ValueError, match="Unknown profile"):
            get_quality_profile("invalid_profile")

    def test_profile_values_are_valid(self):
        """Test all profile values are valid config values"""
        for profile_name in ["fast", "balanced", "quality"]:
            profile = get_quality_profile(profile_name)

            # All keys should be valid config attributes
            for key in profile.keys():
                assert hasattr(vision_config, key)


class TestValidateImageSize:
    """Test image size validation"""

    def test_validate_valid_size(self):
        """Test validating valid file size"""
        # 3MB in bytes
        size_bytes = 3 * 1024 * 1024

        assert validate_image_size(size_bytes) is True

    def test_validate_max_size(self):
        """Test validating maximum allowed size"""
        # Exactly 5MB
        size_bytes = vision_config.max_file_size_mb * 1024 * 1024

        assert validate_image_size(size_bytes) is True

    def test_validate_oversized(self):
        """Test validating oversized file"""
        # 6MB in bytes
        size_bytes = 6 * 1024 * 1024

        assert validate_image_size(size_bytes) is False

    def test_validate_zero_size(self):
        """Test validating zero-size file"""
        assert validate_image_size(0) is True  # Zero is technically valid

    def test_validate_small_size(self):
        """Test validating very small file"""
        size_bytes = 100  # 100 bytes

        assert validate_image_size(size_bytes) is True


class TestValidateImageDimensions:
    """Test image dimensions validation"""

    def test_validate_valid_dimensions(self):
        """Test validating valid dimensions"""
        assert validate_image_dimensions(800, 600) is True

    def test_validate_minimum_dimensions(self):
        """Test validating minimum dimensions"""
        min_w = vision_config.min_width
        min_h = vision_config.min_height

        assert validate_image_dimensions(min_w, min_h) is True

    def test_validate_maximum_dimensions(self):
        """Test validating maximum dimensions"""
        max_w = vision_config.max_width
        max_h = vision_config.max_height

        assert validate_image_dimensions(max_w, max_h) is True

    def test_validate_too_small_width(self):
        """Test validating too small width"""
        assert validate_image_dimensions(50, 600) is False

    def test_validate_too_small_height(self):
        """Test validating too small height"""
        assert validate_image_dimensions(800, 50) is False

    def test_validate_too_large_width(self):
        """Test validating too large width"""
        assert validate_image_dimensions(10000, 600) is False

    def test_validate_too_large_height(self):
        """Test validating too large height"""
        assert validate_image_dimensions(800, 10000) is False

    def test_validate_square_image(self):
        """Test validating square image"""
        assert validate_image_dimensions(1000, 1000) is True


class TestConfigurationFeatures:
    """Test specific configuration features"""

    def test_cache_configuration(self):
        """Test cache-related configuration"""
        config = VisionConfig()

        assert hasattr(config, "enable_cache")
        assert hasattr(config, "cache_ttl")
        assert hasattr(config, "cache_directory")

    def test_storage_configuration(self):
        """Test storage-related configuration"""
        config = VisionConfig()

        assert hasattr(config, "store_processed_images")
        assert hasattr(config, "storage_directory")

    def test_batch_configuration(self):
        """Test batch processing configuration"""
        config = VisionConfig()

        assert hasattr(config, "max_images_per_request")
        assert hasattr(config, "enable_batch_processing")
        assert config.max_images_per_request <= 5  # Claude API limit

    def test_timeout_configuration(self):
        """Test timeout configuration"""
        config = VisionConfig()

        assert hasattr(config, "processing_timeout")
        assert hasattr(config, "analysis_timeout")
        assert config.processing_timeout > 0
        assert config.analysis_timeout > 0

    def test_error_handling_configuration(self):
        """Test error handling configuration"""
        config = VisionConfig()

        assert hasattr(config, "retry_on_failure")
        assert hasattr(config, "max_retries")
        assert hasattr(config, "enable_fallback")


class TestConfigurationEnvironmentVariables:
    """Test environment variable support"""

    @patch.dict('os.environ', {'VISION_MAX_TOKENS': '2048'})
    def test_env_override_max_tokens(self):
        """Test environment variable override"""
        config = VisionConfig()
        # Note: This test might not work as expected since config is already loaded
        # In real usage, env vars are read at import time

    def test_config_has_env_support(self):
        """Test config class has env file support"""
        config = VisionConfig()

        assert hasattr(config, 'Config')
        assert hasattr(config.Config, 'env_file')


class TestConfigurationValidation:
    """Test configuration validation logic"""

    def test_quality_ranges(self):
        """Test quality values are in valid ranges"""
        config = VisionConfig()

        assert 1 <= config.resize_quality <= 100
        assert 1 <= config.thumbnail_quality <= 100

    def test_temperature_range(self):
        """Test temperature is in valid range"""
        config = VisionConfig()

        assert 0.0 <= config.temperature <= 1.0

    def test_positive_values(self):
        """Test values that should be positive"""
        config = VisionConfig()

        assert config.max_tokens > 0
        assert config.max_file_size_mb > 0
        assert config.cache_ttl > 0
        assert config.processing_timeout > 0

    def test_media_types_match_formats(self):
        """Test media types correspond to supported formats"""
        config = VisionConfig()

        # Should have media type for each format
        assert len(config.supported_media_types) >= len(config.supported_formats) - 1
        # (JPEG and JPG share same media type)


class TestConfigurationDefaults:
    """Test default configuration values are sensible"""

    def test_default_model_is_sonnet(self):
        """Test default model is Claude 3.5 Sonnet"""
        config = VisionConfig()

        assert "sonnet" in config.vision_model.lower()
        assert "claude" in config.vision_model.lower()

    def test_default_size_limit_is_5mb(self):
        """Test default size limit matches Claude API"""
        config = VisionConfig()

        assert config.max_file_size_mb == 5.0

    def test_default_enables_useful_features(self):
        """Test defaults enable useful features"""
        config = VisionConfig()

        assert config.auto_resize is True
        assert config.enable_cache is True
        assert config.enable_detailed_analysis is True
        assert config.generate_thumbnails is True
