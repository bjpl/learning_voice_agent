"""
Unit Tests for Audio Pipeline
Tests audio transcription, format detection, and validation
"""
import pytest
import base64
import io
from unittest.mock import AsyncMock, MagicMock, patch
from app.audio_pipeline import (
    AudioPipeline,
    AudioData,
    AudioFormat,
    WhisperStrategy
)


class TestAudioFormat:
    """Test AudioFormat enum"""

    def test_audio_formats_exist(self):
        """Test all expected audio formats are defined"""
        assert AudioFormat.WAV
        assert AudioFormat.MP3
        assert AudioFormat.OGG
        assert AudioFormat.WEBM
        assert AudioFormat.RAW


class TestAudioData:
    """Test AudioData dataclass"""

    def test_creation(self):
        """Test AudioData creation"""
        data = AudioData(
            content=b"test",
            format=AudioFormat.WAV,
            sample_rate=44100,
            duration=5.0,
            source="test"
        )

        assert data.content == b"test"
        assert data.format == AudioFormat.WAV
        assert data.sample_rate == 44100
        assert data.duration == 5.0
        assert data.source == "test"

    def test_creation_with_defaults(self):
        """Test AudioData creation with default values"""
        data = AudioData(content=b"test", format=AudioFormat.MP3)

        assert data.sample_rate is None
        assert data.duration is None
        assert data.source == "unknown"


class TestWhisperStrategy:
    """Test WhisperStrategy transcription"""

    @pytest.mark.asyncio
    async def test_transcribe_success(self, mock_openai_client):
        """Test successful transcription"""
        strategy = WhisperStrategy()
        strategy.client = mock_openai_client

        audio = AudioData(
            content=b"test audio data",
            format=AudioFormat.WAV
        )

        result = await strategy.transcribe(audio)

        assert result == "This is transcribed text"
        assert mock_openai_client.audio.transcriptions.create.called

    @pytest.mark.asyncio
    async def test_transcribe_api_error(self, mock_openai_client):
        """Test transcription with API error"""
        strategy = WhisperStrategy()
        strategy.client = mock_openai_client

        mock_openai_client.audio.transcriptions.create.side_effect = Exception("API Error")

        audio = AudioData(content=b"test", format=AudioFormat.WAV)

        with pytest.raises(Exception):
            await strategy.transcribe(audio)


class TestAudioPipeline:
    """Test AudioPipeline class"""

    def test_initialization(self):
        """Test pipeline initialization"""
        pipeline = AudioPipeline()

        assert pipeline.transcription_strategy is not None
        assert isinstance(pipeline.transcription_strategy, WhisperStrategy)
        assert pipeline.max_duration > 0

    def test_detect_format_wav(self, sample_audio_wav):
        """Test WAV format detection"""
        pipeline = AudioPipeline()

        format = pipeline._detect_format(sample_audio_wav)

        assert format == AudioFormat.WAV

    def test_detect_format_mp3(self, sample_audio_mp3):
        """Test MP3 format detection"""
        pipeline = AudioPipeline()

        format = pipeline._detect_format(sample_audio_mp3)

        assert format == AudioFormat.MP3

    def test_detect_format_ogg(self):
        """Test OGG format detection"""
        pipeline = AudioPipeline()
        ogg_data = b'OggS' + b'\x00' * 100

        format = pipeline._detect_format(ogg_data)

        assert format == AudioFormat.OGG

    def test_detect_format_webm(self):
        """Test WEBM format detection"""
        pipeline = AudioPipeline()
        webm_data = b'webm' + b'\x00' * 100

        format = pipeline._detect_format(webm_data)

        assert format == AudioFormat.WEBM

    def test_detect_format_raw(self):
        """Test RAW format detection for unknown formats"""
        pipeline = AudioPipeline()
        unknown_data = b'\x00\x01\x02\x03' + b'\x00' * 100

        format = pipeline._detect_format(unknown_data)

        assert format == AudioFormat.RAW

    def test_validate_audio_success(self, sample_audio_wav):
        """Test successful audio validation"""
        pipeline = AudioPipeline()
        audio = AudioData(content=sample_audio_wav, format=AudioFormat.WAV)

        result = pipeline._validate_audio(audio)

        assert result is True

    def test_validate_audio_too_large(self):
        """Test validation fails for too large audio"""
        pipeline = AudioPipeline()
        large_audio = b'\x00' * (26 * 1024 * 1024)  # 26MB
        audio = AudioData(content=large_audio, format=AudioFormat.WAV)

        with pytest.raises(ValueError, match="too large"):
            pipeline._validate_audio(audio)

    def test_validate_audio_unsupported_format(self):
        """Test validation fails for unsupported format"""
        pipeline = AudioPipeline()
        audio = AudioData(content=b"test", format=AudioFormat.RAW)

        with pytest.raises(ValueError, match="Unsupported format"):
            pipeline._validate_audio(audio)

    @pytest.mark.asyncio
    async def test_transcribe_audio_success(self, test_audio_pipeline, sample_audio_wav):
        """Test successful audio transcription"""
        result = await test_audio_pipeline.transcribe_audio(
            sample_audio_wav,
            source="test"
        )

        assert result == "This is transcribed text."
        # The fixture mocks transcribe() directly, so verify that was called
        assert test_audio_pipeline.transcription_strategy.transcribe.called

    @pytest.mark.asyncio
    async def test_transcribe_audio_with_format_hint(self, test_audio_pipeline, sample_audio_wav):
        """Test transcription with format hint"""
        result = await test_audio_pipeline.transcribe_audio(
            sample_audio_wav,
            source="test",
            format_hint="wav"
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_transcribe_base64_success(self, test_audio_pipeline, sample_audio_base64):
        """Test base64 audio transcription"""
        result = await test_audio_pipeline.transcribe_base64(
            sample_audio_base64,
            source="browser"
        )

        assert result is not None

    def test_clean_transcript_basic(self):
        """Test basic transcript cleaning"""
        pipeline = AudioPipeline()

        result = pipeline._clean_transcript("  Hello   world  ")

        assert result == "Hello world."

    def test_clean_transcript_removes_artifacts(self):
        """Test artifact removal"""
        pipeline = AudioPipeline()

        result = pipeline._clean_transcript("Hello [BLANK_AUDIO] world [INAUDIBLE] test...")

        assert "[BLANK_AUDIO]" not in result
        assert "[INAUDIBLE]" not in result
        assert result == "Hello  world  test."

    def test_clean_transcript_adds_punctuation(self):
        """Test punctuation is added"""
        pipeline = AudioPipeline()

        result = pipeline._clean_transcript("Hello world")

        assert result.endswith(".")

    def test_clean_transcript_preserves_punctuation(self):
        """Test existing punctuation is preserved"""
        pipeline = AudioPipeline()

        result = pipeline._clean_transcript("Hello world!")

        assert result == "Hello world!"

    def test_clean_transcript_empty(self):
        """Test cleaning empty transcript"""
        pipeline = AudioPipeline()

        result = pipeline._clean_transcript("")

        assert result == ""

    def test_clean_transcript_whitespace_only(self):
        """Test cleaning whitespace-only transcript"""
        pipeline = AudioPipeline()

        result = pipeline._clean_transcript("   ")

        assert result == ""

    @pytest.mark.asyncio
    async def test_transcribe_stream_basic(self, test_audio_pipeline):
        """Test streaming transcription"""
        async def audio_stream():
            # Simulate chunks of audio
            chunk = b'RIFF' + b'\x00' * (1024 * 512)  # 512KB chunk
            yield chunk
            yield chunk

        results = []
        async for text in test_audio_pipeline.transcribe_stream(audio_stream(), AudioFormat.WAV):
            results.append(text)

        # Should have processed chunks
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_transcribe_audio_performance(self, test_audio_pipeline, sample_audio_wav, timing):
        """Test transcription performance"""
        with timing:
            await test_audio_pipeline.transcribe_audio(sample_audio_wav)

        # Should be fast with mocked API (< 100ms)
        assert timing.elapsed < 0.1

    @pytest.mark.asyncio
    async def test_transcribe_multiple_formats(self, test_audio_pipeline):
        """Test transcription with different audio formats"""
        formats = [
            (b'RIFF' + b'\x00' * 100, "wav"),
            (b'\xff\xfb' + b'\x00' * 100, "mp3"),
            (b'OggS' + b'\x00' * 100, "ogg"),
        ]

        for audio_data, format_hint in formats:
            result = await test_audio_pipeline.transcribe_audio(
                audio_data,
                format_hint=format_hint
            )
            assert result is not None

    @pytest.mark.asyncio
    async def test_transcribe_validates_before_processing(self, test_audio_pipeline):
        """Test that validation occurs before transcription"""
        # Create audio that's too large
        large_audio = b'RIFF' + b'\x00' * (26 * 1024 * 1024)

        with pytest.raises(ValueError, match="too large"):
            await test_audio_pipeline.transcribe_audio(large_audio)
