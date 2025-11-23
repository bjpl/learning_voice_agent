"""
Vision Analysis Configuration

SPECIFICATION:
- Centralized vision system configuration
- Image processing parameters
- Claude vision model settings
- Format and size validation rules
- Performance optimization settings

ARCHITECTURE:
- Pydantic-based configuration
- Environment variable support
- Sensible defaults with overrides
- Type-safe settings

WHY:
- Single source of truth for vision parameters
- Easy to tune performance vs quality
- Production-ready configuration management
"""
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import List, Optional, Tuple


class VisionConfig(BaseSettings):
    """
    Vision Analysis System Configuration

    PATTERN: Centralized configuration with environment overrides
    WHY: Easy tuning and deployment flexibility
    """

    # ========== MODEL CONFIGURATION ==========

    # Claude vision model
    vision_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        env="VISION_MODEL",
        description="Claude model with vision capabilities"
    )

    # Maximum tokens for vision analysis
    max_tokens: int = Field(
        default=1024,
        env="VISION_MAX_TOKENS",
        description="Maximum tokens for vision response"
    )

    # Temperature for vision analysis
    temperature: float = Field(
        default=0.3,
        env="VISION_TEMPERATURE",
        description="Temperature for vision analysis (0.0-1.0)"
    )

    # Enable detailed analysis
    enable_detailed_analysis: bool = Field(
        default=True,
        env="VISION_ENABLE_DETAILED_ANALYSIS",
        description="Enable detailed image analysis mode"
    )

    # ========== IMAGE FORMAT CONFIGURATION ==========

    # Supported image formats
    supported_formats: List[str] = Field(
        default=["PNG", "JPEG", "JPG", "GIF", "WEBP"],
        env="VISION_SUPPORTED_FORMATS",
        description="Supported image formats"
    )

    # Supported media types for API
    supported_media_types: List[str] = Field(
        default=["image/png", "image/jpeg", "image/gif", "image/webp"],
        env="VISION_SUPPORTED_MEDIA_TYPES",
        description="Supported media types for Claude API"
    )

    # Default format for conversion
    default_format: str = Field(
        default="PNG",
        env="VISION_DEFAULT_FORMAT",
        description="Default format for image conversion"
    )

    # ========== SIZE AND DIMENSION LIMITS ==========

    # Maximum file size (5MB for Claude API)
    max_file_size_mb: float = Field(
        default=5.0,
        env="VISION_MAX_FILE_SIZE_MB",
        description="Maximum image file size in MB"
    )

    # Maximum image dimensions
    max_width: int = Field(
        default=8000,
        env="VISION_MAX_WIDTH",
        description="Maximum image width in pixels"
    )

    max_height: int = Field(
        default=8000,
        env="VISION_MAX_HEIGHT",
        description="Maximum image height in pixels"
    )

    # Minimum image dimensions
    min_width: int = Field(
        default=100,
        env="VISION_MIN_WIDTH",
        description="Minimum image width in pixels"
    )

    min_height: int = Field(
        default=100,
        env="VISION_MIN_HEIGHT",
        description="Minimum image height in pixels"
    )

    # ========== IMAGE PROCESSING CONFIGURATION ==========

    # Auto-resize if oversized
    auto_resize: bool = Field(
        default=True,
        env="VISION_AUTO_RESIZE",
        description="Automatically resize oversized images"
    )

    # Target size when resizing (in MB)
    target_size_mb: float = Field(
        default=4.5,
        env="VISION_TARGET_SIZE_MB",
        description="Target size for resized images (MB)"
    )

    # Resize quality (1-100)
    resize_quality: int = Field(
        default=85,
        env="VISION_RESIZE_QUALITY",
        description="JPEG quality for resizing (1-100)"
    )

    # Preserve aspect ratio
    preserve_aspect_ratio: bool = Field(
        default=True,
        env="VISION_PRESERVE_ASPECT_RATIO",
        description="Preserve aspect ratio when resizing"
    )

    # Target dimensions for resizing
    resize_max_width: int = Field(
        default=4096,
        env="VISION_RESIZE_MAX_WIDTH",
        description="Maximum width when resizing"
    )

    resize_max_height: int = Field(
        default=4096,
        env="VISION_RESIZE_MAX_HEIGHT",
        description="Maximum height when resizing"
    )

    # ========== THUMBNAIL CONFIGURATION ==========

    # Generate thumbnails
    generate_thumbnails: bool = Field(
        default=True,
        env="VISION_GENERATE_THUMBNAILS",
        description="Generate thumbnails for images"
    )

    # Thumbnail dimensions
    thumbnail_size: Tuple[int, int] = Field(
        default=(256, 256),
        env="VISION_THUMBNAIL_SIZE",
        description="Thumbnail dimensions (width, height)"
    )

    # Thumbnail quality
    thumbnail_quality: int = Field(
        default=75,
        env="VISION_THUMBNAIL_QUALITY",
        description="Thumbnail JPEG quality (1-100)"
    )

    # ========== BATCH PROCESSING CONFIGURATION ==========

    # Maximum images per batch
    max_images_per_request: int = Field(
        default=5,
        env="VISION_MAX_IMAGES_PER_REQUEST",
        description="Maximum images per API request (Claude limit)"
    )

    # Enable batch processing
    enable_batch_processing: bool = Field(
        default=True,
        env="VISION_ENABLE_BATCH_PROCESSING",
        description="Enable batch image processing"
    )

    # Batch size for parallel processing
    batch_size: int = Field(
        default=3,
        env="VISION_BATCH_SIZE",
        description="Number of images to process in parallel"
    )

    # ========== CACHING CONFIGURATION ==========

    # Cache processed images
    enable_cache: bool = Field(
        default=True,
        env="VISION_ENABLE_CACHE",
        description="Cache processed images and results"
    )

    # Cache TTL (seconds)
    cache_ttl: int = Field(
        default=3600,
        env="VISION_CACHE_TTL",
        description="Cache TTL in seconds"
    )

    # Cache directory
    cache_directory: str = Field(
        default="./data/vision_cache",
        env="VISION_CACHE_DIRECTORY",
        description="Directory for vision cache"
    )

    # ========== STORAGE CONFIGURATION ==========

    # Store processed images
    store_processed_images: bool = Field(
        default=True,
        env="VISION_STORE_PROCESSED_IMAGES",
        description="Store processed images to disk"
    )

    # Storage directory
    storage_directory: str = Field(
        default="./data/vision_storage",
        env="VISION_STORAGE_DIRECTORY",
        description="Directory for storing processed images"
    )

    # Store thumbnails
    store_thumbnails: bool = Field(
        default=True,
        env="VISION_STORE_THUMBNAILS",
        description="Store generated thumbnails"
    )

    # ========== EXIF AND METADATA ==========

    # Extract EXIF data
    extract_exif: bool = Field(
        default=True,
        env="VISION_EXTRACT_EXIF",
        description="Extract EXIF metadata from images"
    )

    # Preserve metadata on resize
    preserve_metadata: bool = Field(
        default=True,
        env="VISION_PRESERVE_METADATA",
        description="Preserve metadata when processing"
    )

    # ========== PERFORMANCE CONFIGURATION ==========

    # Processing timeout (seconds)
    processing_timeout: float = Field(
        default=30.0,
        env="VISION_PROCESSING_TIMEOUT",
        description="Maximum time for image processing (seconds)"
    )

    # Analysis timeout (seconds)
    analysis_timeout: float = Field(
        default=60.0,
        env="VISION_ANALYSIS_TIMEOUT",
        description="Maximum time for vision analysis (seconds)"
    )

    # Enable parallel processing
    enable_parallel_processing: bool = Field(
        default=True,
        env="VISION_ENABLE_PARALLEL_PROCESSING",
        description="Enable parallel image processing"
    )

    # Max concurrent operations
    max_concurrent_operations: int = Field(
        default=3,
        env="VISION_MAX_CONCURRENT_OPERATIONS",
        description="Maximum concurrent processing operations"
    )

    # ========== ERROR HANDLING ==========

    # Retry failed operations
    retry_on_failure: bool = Field(
        default=True,
        env="VISION_RETRY_ON_FAILURE",
        description="Retry failed vision operations"
    )

    # Maximum retries
    max_retries: int = Field(
        default=3,
        env="VISION_MAX_RETRIES",
        description="Maximum number of retries"
    )

    # Retry delay (seconds)
    retry_delay: float = Field(
        default=1.0,
        env="VISION_RETRY_DELAY",
        description="Delay between retries (seconds)"
    )

    # Graceful degradation
    enable_fallback: bool = Field(
        default=True,
        env="VISION_ENABLE_FALLBACK",
        description="Enable fallback mode on errors"
    )

    # ========== DEBUGGING CONFIGURATION ==========

    # Log processing details
    log_processing_details: bool = Field(
        default=True,
        env="VISION_LOG_PROCESSING_DETAILS",
        description="Log detailed processing information"
    )

    # Include metadata in response
    include_metadata: bool = Field(
        default=True,
        env="VISION_INCLUDE_METADATA",
        description="Include image metadata in responses"
    )

    # Save debug images
    save_debug_images: bool = Field(
        default=False,
        env="VISION_SAVE_DEBUG_IMAGES",
        description="Save debug images during processing"
    )

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


# Singleton instance
vision_config = VisionConfig()


# ========== HELPER FUNCTIONS ==========

def get_vision_config() -> VisionConfig:
    """
    Get vision configuration instance

    Returns:
        VisionConfig singleton instance
    """
    return vision_config


def update_vision_config(**kwargs) -> None:
    """
    Update vision configuration dynamically

    Args:
        **kwargs: Configuration parameters to update

    Example:
        update_vision_config(max_tokens=2048, temperature=0.5)
    """
    global vision_config
    for key, value in kwargs.items():
        if hasattr(vision_config, key):
            setattr(vision_config, key, value)
        else:
            raise ValueError(f"Invalid vision configuration parameter: {key}")


def get_quality_profile(profile: str = "balanced") -> dict:
    """
    Get predefined quality profiles

    PATTERN: Quality vs performance trade-off presets
    WHY: Easy tuning for different use cases

    Args:
        profile: One of "fast", "balanced", "quality"

    Returns:
        Dictionary of configuration overrides
    """
    profiles = {
        "fast": {
            "max_tokens": 512,
            "temperature": 0.1,
            "resize_quality": 70,
            "thumbnail_quality": 60,
            "enable_detailed_analysis": False,
            "extract_exif": False,
        },
        "balanced": {
            "max_tokens": 1024,
            "temperature": 0.3,
            "resize_quality": 85,
            "thumbnail_quality": 75,
            "enable_detailed_analysis": True,
            "extract_exif": True,
        },
        "quality": {
            "max_tokens": 2048,
            "temperature": 0.5,
            "resize_quality": 95,
            "thumbnail_quality": 85,
            "enable_detailed_analysis": True,
            "extract_exif": True,
        },
    }

    if profile not in profiles:
        raise ValueError(f"Unknown profile: {profile}. Choose from: {list(profiles.keys())}")

    return profiles[profile]


def validate_image_size(file_size_bytes: int) -> bool:
    """
    Validate image file size

    Args:
        file_size_bytes: File size in bytes

    Returns:
        True if valid, False otherwise
    """
    max_bytes = vision_config.max_file_size_mb * 1024 * 1024
    return file_size_bytes <= max_bytes


def validate_image_dimensions(width: int, height: int) -> bool:
    """
    Validate image dimensions

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        True if valid, False otherwise
    """
    return (
        vision_config.min_width <= width <= vision_config.max_width
        and vision_config.min_height <= height <= vision_config.max_height
    )
