"""
Image Processor for Vision Analysis

SPECIFICATION:
- Image validation (format, size, dimensions)
- Automatic resizing with quality preservation
- Format conversion and optimization
- Thumbnail generation
- EXIF data extraction
- Base64 encoding for API

ARCHITECTURE:
- Single-responsibility image processing
- Pillow-based image manipulation
- Error handling and validation
- Async support for I/O operations

WHY:
- Ensure images meet Claude API requirements
- Optimize images for best quality/size balance
- Extract useful metadata
- Prepare images for API consumption
"""
import io
import os
import base64
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, BinaryIO
from PIL import Image, ImageOps, ExifTags
import logging
from datetime import datetime

from .config import vision_config, validate_image_size, validate_image_dimensions


logger = logging.getLogger(__name__)


class ImageValidationError(Exception):
    """Raised when image validation fails"""
    pass


class ImageProcessingError(Exception):
    """Raised when image processing fails"""
    pass


class ImageProcessor:
    """
    Image processor for vision analysis

    PATTERN: Single-responsibility processor with validation
    WHY: Ensure images are ready for Claude Vision API

    Example:
        processor = ImageProcessor()
        result = await processor.process_image("path/to/image.jpg")
        # result contains processed image data, metadata, etc.
    """

    def __init__(self, config=None):
        """
        Initialize image processor

        Args:
            config: Optional VisionConfig override
        """
        self.config = config or vision_config
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create necessary directories"""
        directories = [
            self.config.cache_directory,
            self.config.storage_directory,
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    # ========== VALIDATION ==========

    def validate_image(self, image_path: str) -> Dict[str, any]:
        """
        Validate image file

        Args:
            image_path: Path to image file

        Returns:
            Validation result with details

        Raises:
            ImageValidationError: If validation fails
        """
        path = Path(image_path)

        # Check file exists
        if not path.exists():
            raise ImageValidationError(f"Image file not found: {image_path}")

        # Check file size
        file_size = path.stat().st_size
        if not validate_image_size(file_size):
            max_size = self.config.max_file_size_mb
            actual_size = file_size / (1024 * 1024)
            raise ImageValidationError(
                f"Image too large: {actual_size:.2f}MB (max: {max_size}MB)"
            )

        # Open and validate image
        try:
            with Image.open(image_path) as img:
                # Check format
                if img.format not in self.config.supported_formats:
                    raise ImageValidationError(
                        f"Unsupported format: {img.format}. "
                        f"Supported: {', '.join(self.config.supported_formats)}"
                    )

                # Check dimensions
                width, height = img.size
                if not validate_image_dimensions(width, height):
                    raise ImageValidationError(
                        f"Invalid dimensions: {width}x{height}. "
                        f"Must be {self.config.min_width}-{self.config.max_width} x "
                        f"{self.config.min_height}-{self.config.max_height}"
                    )

                return {
                    "valid": True,
                    "format": img.format,
                    "size": (width, height),
                    "file_size": file_size,
                    "mode": img.mode,
                }

        except Exception as e:
            raise ImageValidationError(f"Failed to validate image: {str(e)}")

    def get_image_info(self, image_path: str) -> Dict[str, any]:
        """
        Get comprehensive image information

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with image metadata
        """
        path = Path(image_path)
        file_size = path.stat().st_size

        with Image.open(image_path) as img:
            info = {
                "filename": path.name,
                "path": str(path.absolute()),
                "format": img.format,
                "mode": img.mode,
                "size": img.size,
                "width": img.width,
                "height": img.height,
                "file_size_bytes": file_size,
                "file_size_mb": file_size / (1024 * 1024),
                "aspect_ratio": img.width / img.height if img.height > 0 else 0,
            }

            # Add EXIF data if available and enabled
            if self.config.extract_exif:
                exif_data = self._extract_exif(img)
                if exif_data:
                    info["exif"] = exif_data

            return info

    # ========== IMAGE PROCESSING ==========

    def process_image(self, image_path: str, auto_resize: bool = None) -> Dict[str, any]:
        """
        Process image for vision analysis

        Args:
            image_path: Path to image file
            auto_resize: Override auto-resize setting

        Returns:
            Dictionary with processed image data and metadata
        """
        # Validate image
        validation = self.validate_image(image_path)

        # Get full info
        info = self.get_image_info(image_path)

        # Check if resizing needed
        should_resize = auto_resize if auto_resize is not None else self.config.auto_resize
        file_size_mb = info["file_size_mb"]

        if should_resize and file_size_mb > self.config.max_file_size_mb:
            logger.info(f"Image oversized ({file_size_mb:.2f}MB), resizing...")
            processed_image = self._resize_image(image_path)
            info["resized"] = True
            info["original_size_mb"] = file_size_mb
        else:
            with Image.open(image_path) as img:
                processed_image = img.copy()
            info["resized"] = False

        # Generate base64
        base64_data = self._image_to_base64(processed_image)
        media_type = self._get_media_type(processed_image.format or "PNG")

        # Generate thumbnail if enabled
        thumbnail = None
        if self.config.generate_thumbnails:
            thumbnail = self._create_thumbnail(processed_image)

        # Calculate hash for caching
        image_hash = self._calculate_hash(base64_data)

        result = {
            "image_data": base64_data,
            "media_type": media_type,
            "metadata": info,
            "thumbnail": thumbnail,
            "hash": image_hash,
            "processed_at": datetime.utcnow().isoformat(),
        }

        return result

    def _resize_image(self, image_path: str) -> Image.Image:
        """
        Resize image to meet size requirements

        Args:
            image_path: Path to image file

        Returns:
            Resized PIL Image
        """
        with Image.open(image_path) as img:
            # Ensure RGB mode for JPEG
            if img.mode in ('RGBA', 'LA', 'P'):
                # Convert to RGB
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = rgb_img

            # Calculate new dimensions
            max_width = self.config.resize_max_width
            max_height = self.config.resize_max_height

            if self.config.preserve_aspect_ratio:
                # Preserve aspect ratio
                img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                resized = img
            else:
                # Exact dimensions
                resized = img.resize((max_width, max_height), Image.Resampling.LANCZOS)

            # Apply auto-orientation from EXIF
            if self.config.extract_exif:
                resized = ImageOps.exif_transpose(resized) or resized

            return resized

    def _create_thumbnail(self, image: Image.Image) -> Optional[str]:
        """
        Create thumbnail from image

        Args:
            image: PIL Image

        Returns:
            Base64 encoded thumbnail or None
        """
        try:
            thumb = image.copy()
            thumb.thumbnail(self.config.thumbnail_size, Image.Resampling.LANCZOS)

            # Convert to base64
            buffer = io.BytesIO()
            thumb.save(
                buffer,
                format="JPEG",
                quality=self.config.thumbnail_quality,
                optimize=True
            )
            thumbnail_b64 = base64.b64encode(buffer.getvalue()).decode()

            return thumbnail_b64

        except Exception as e:
            logger.warning(f"Failed to create thumbnail: {e}")
            return None

    # ========== FORMAT CONVERSION ==========

    def convert_to_format(self, image_path: str, target_format: str) -> Image.Image:
        """
        Convert image to target format

        Args:
            image_path: Path to image file
            target_format: Target format (PNG, JPEG, etc.)

        Returns:
            Converted PIL Image
        """
        target_format = target_format.upper()

        if target_format not in self.config.supported_formats:
            raise ImageProcessingError(f"Unsupported target format: {target_format}")

        with Image.open(image_path) as img:
            # Handle transparency for JPEG
            if target_format == "JPEG" and img.mode in ('RGBA', 'LA', 'P'):
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                return rgb_img

            # Convert mode if needed
            if target_format == "JPEG" and img.mode != "RGB":
                return img.convert("RGB")
            elif target_format == "PNG" and img.mode not in ("RGB", "RGBA"):
                return img.convert("RGBA")

            return img.copy()

    # ========== ENCODING ==========

    def _image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string

        Args:
            image: PIL Image

        Returns:
            Base64 encoded string
        """
        buffer = io.BytesIO()

        # Determine format and quality
        format = image.format or self.config.default_format
        save_kwargs = {"format": format}

        if format == "JPEG":
            save_kwargs["quality"] = self.config.resize_quality
            save_kwargs["optimize"] = True
        elif format == "PNG":
            save_kwargs["optimize"] = True

        # Save to buffer
        image.save(buffer, **save_kwargs)

        # Encode to base64
        base64_data = base64.b64encode(buffer.getvalue()).decode()

        return base64_data

    def encode_image_file(self, image_path: str) -> Tuple[str, str]:
        """
        Encode image file to base64

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (base64_data, media_type)
        """
        with Image.open(image_path) as img:
            base64_data = self._image_to_base64(img)
            media_type = self._get_media_type(img.format or "PNG")

        return base64_data, media_type

    # ========== EXIF EXTRACTION ==========

    def _extract_exif(self, image: Image.Image) -> Optional[Dict[str, any]]:
        """
        Extract EXIF metadata from image

        Args:
            image: PIL Image

        Returns:
            Dictionary of EXIF data or None
        """
        try:
            exif_data = {}
            exif = image.getexif()

            if exif:
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    # Convert bytes to string
                    if isinstance(value, bytes):
                        try:
                            value = value.decode()
                        except:
                            value = str(value)
                    exif_data[tag] = value

            return exif_data if exif_data else None

        except Exception as e:
            logger.debug(f"Failed to extract EXIF: {e}")
            return None

    # ========== UTILITIES ==========

    def _get_media_type(self, format: str) -> str:
        """
        Get media type for image format

        Args:
            format: Image format (PNG, JPEG, etc.)

        Returns:
            Media type string
        """
        media_types = {
            "PNG": "image/png",
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "GIF": "image/gif",
            "WEBP": "image/webp",
        }
        return media_types.get(format.upper(), "image/png")

    def _calculate_hash(self, data: str) -> str:
        """
        Calculate hash of image data

        Args:
            data: Image data (base64 string)

        Returns:
            Hash string
        """
        return hashlib.sha256(data.encode()).hexdigest()

    def batch_process_images(self, image_paths: List[str]) -> List[Dict[str, any]]:
        """
        Process multiple images in batch

        Args:
            image_paths: List of image file paths

        Returns:
            List of processed image results
        """
        results = []

        for image_path in image_paths:
            try:
                result = self.process_image(image_path)
                results.append({
                    "success": True,
                    "path": image_path,
                    "data": result
                })
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results.append({
                    "success": False,
                    "path": image_path,
                    "error": str(e)
                })

        return results

    def save_processed_image(
        self,
        image: Image.Image,
        output_path: Optional[str] = None,
        format: Optional[str] = None
    ) -> str:
        """
        Save processed image to disk

        Args:
            image: PIL Image to save
            output_path: Output path (optional)
            format: Output format (optional)

        Returns:
            Path to saved image
        """
        if output_path is None:
            # Generate path in storage directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"processed_{timestamp}.{(format or 'png').lower()}"
            output_path = os.path.join(self.config.storage_directory, filename)

        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save image
        save_format = format or image.format or self.config.default_format
        save_kwargs = {"format": save_format}

        if save_format.upper() == "JPEG":
            save_kwargs["quality"] = self.config.resize_quality
            save_kwargs["optimize"] = True
        elif save_format.upper() == "PNG":
            save_kwargs["optimize"] = True

        image.save(output_path, **save_kwargs)

        logger.info(f"Saved processed image to {output_path}")
        return output_path


# ========== HELPER FUNCTIONS ==========

def quick_process_image(image_path: str) -> Dict[str, any]:
    """
    Quick image processing with defaults

    Args:
        image_path: Path to image file

    Returns:
        Processed image data
    """
    processor = ImageProcessor()
    return processor.process_image(image_path)


def validate_and_prepare_image(image_path: str) -> Tuple[str, str]:
    """
    Validate and prepare image for API

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (base64_data, media_type)
    """
    processor = ImageProcessor()
    result = processor.process_image(image_path)
    return result["image_data"], result["media_type"]
