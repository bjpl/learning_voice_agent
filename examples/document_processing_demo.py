#!/usr/bin/env python3
"""
Document Processing Pipeline Demo

Demonstrates the capabilities of the Phase 4 document processing system.

Usage:
    python examples/document_processing_demo.py

Features demonstrated:
- Processing multiple document formats
- Extracting text, metadata, and structure
- Generating chunks for RAG
- Custom configuration
- Error handling
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.documents import DocumentProcessor, DocumentConfig
from app.documents.config import PresetConfigs


async def demo_basic_processing():
    """Demo 1: Basic document processing"""
    print("\n" + "="*60)
    print("DEMO 1: Basic Document Processing")
    print("="*60)

    processor = DocumentProcessor()

    # Process sample text file
    sample_text = Path(__file__).parent.parent / "data/sample_documents/sample_text.txt"

    if sample_text.exists():
        print(f"\nProcessing: {sample_text.name}")
        result = await processor.process_document(str(sample_text))

        print(f"✓ Format: {result['format']}")
        print(f"✓ Text length: {result['text_length']} characters")
        print(f"✓ Number of chunks: {result['num_chunks']}")
        print(f"✓ Processing time: {result['processing_time_seconds']:.2f}s")

        # Show first chunk
        if result['chunks']:
            print(f"\nFirst chunk preview:")
            print("-" * 60)
            chunk = result['chunks'][0]
            print(chunk['text'][:200] + "...")
            print("-" * 60)
    else:
        print(f"Sample file not found: {sample_text}")


async def demo_markdown_structure():
    """Demo 2: Markdown structure extraction"""
    print("\n" + "="*60)
    print("DEMO 2: Markdown Structure Extraction")
    print("="*60)

    processor = DocumentProcessor()

    # Process sample markdown file
    sample_md = Path(__file__).parent.parent / "data/sample_documents/sample_markdown.md"

    if sample_md.exists():
        print(f"\nProcessing: {sample_md.name}")
        result = await processor.process_document(str(sample_md))

        print(f"✓ Format: {result['format']}")
        print(f"✓ Text length: {result['text_length']} characters")

        # Show structure
        structure = result['structure']
        if 'headings' in structure:
            print(f"\n📑 Headings found: {len(structure['headings'])}")
            for heading in structure['headings'][:5]:  # First 5
                indent = "  " * (heading['level'] - 1)
                print(f"  {indent}{'#' * heading['level']} {heading['text']}")

        if 'code_blocks' in structure:
            print(f"\n💻 Code blocks found: {len(structure['code_blocks'])}")
            for i, block in enumerate(structure['code_blocks'][:3], 1):
                print(f"  {i}. Language: {block['language']}, Lines: {block['length']}")

        if 'links' in structure:
            print(f"\n🔗 Links found: {len(structure['links'])}")
            for link in structure['links'][:5]:
                print(f"  • [{link['text']}]({link['url']})")
    else:
        print(f"Sample file not found: {sample_md}")


async def demo_custom_config():
    """Demo 3: Custom configuration"""
    print("\n" + "="*60)
    print("DEMO 3: Custom Configuration")
    print("="*60)

    # Create custom config
    config = DocumentConfig(
        chunk_size=500,  # Smaller chunks
        chunk_overlap=100,
        max_file_size=5 * 1024 * 1024,  # 5MB
    )

    processor = DocumentProcessor(config)

    sample_text = Path(__file__).parent.parent / "data/sample_documents/sample_text.txt"

    if sample_text.exists():
        print(f"\nProcessing with custom config:")
        print(f"  • Chunk size: {config.chunk_size} tokens")
        print(f"  • Chunk overlap: {config.chunk_overlap} tokens")

        result = await processor.process_document(str(sample_text))

        print(f"\n✓ Generated {result['num_chunks']} chunks (vs. default would be ~{result['num_chunks'] // 2})")

        # Show chunk distribution
        print("\nChunk sizes:")
        for i, chunk in enumerate(result['chunks'][:5], 1):
            print(f"  Chunk {i}: {chunk['word_count']} words, {chunk['char_count']} chars")


async def demo_preset_configs():
    """Demo 4: Preset configurations"""
    print("\n" + "="*60)
    print("DEMO 4: Preset Configurations")
    print("="*60)

    sample_text = Path(__file__).parent.parent / "data/sample_documents/sample_text.txt"

    if not sample_text.exists():
        print(f"Sample file not found: {sample_text}")
        return

    presets = {
        "Fast Processing": PresetConfigs.fast_processing(),
        "Comprehensive": PresetConfigs.comprehensive_extraction(),
        "RAG Optimized": PresetConfigs.rag_optimized(),
    }

    for name, config in presets.items():
        processor = DocumentProcessor(config)
        result = await processor.process_document(str(sample_text))

        print(f"\n{name}:")
        print(f"  • Chunks: {result['num_chunks']}")
        print(f"  • Time: {result['processing_time_seconds']:.3f}s")
        print(f"  • Settings: chunk_size={config.chunk_size}, workers={config.parallel_workers}")


async def demo_metadata_extraction():
    """Demo 5: Metadata extraction"""
    print("\n" + "="*60)
    print("DEMO 5: Metadata Extraction")
    print("="*60)

    processor = DocumentProcessor()

    sample_md = Path(__file__).parent.parent / "data/sample_documents/sample_markdown.md"

    if sample_md.exists():
        print(f"\nExtracting metadata from: {sample_md.name}")

        metadata = await processor.extract_metadata(str(sample_md))

        print("\n📊 Metadata:")
        print(f"  • Format: {metadata.get('format')}")
        print(f"  • File size: {metadata.get('file_size', 0) / 1024:.1f} KB")
        print(f"  • Lines: {metadata.get('num_lines', 0)}")
        print(f"  • Words: {metadata.get('num_words', 0)}")
        print(f"  • Characters: {metadata.get('num_characters', 0)}")

        if metadata.get('is_markdown'):
            print(f"  • Is Markdown: Yes")
            print(f"  • Headings: {metadata.get('num_headings', 0)}")
            print(f"  • Code blocks: {metadata.get('num_code_blocks', 0)}")
            print(f"  • Links: {metadata.get('num_links', 0)}")


async def demo_format_detection():
    """Demo 6: Format detection"""
    print("\n" + "="*60)
    print("DEMO 6: Automatic Format Detection")
    print("="*60)

    processor = DocumentProcessor()

    test_files = [
        "document.pdf",
        "report.docx",
        "notes.txt",
        "readme.md",
        "README.markdown",
        "data.doc",
    ]

    print("\nSupported formats:")
    formats = processor.get_supported_formats()
    print(f"  {', '.join(formats)}")

    print("\nFormat detection:")
    for filename in test_files:
        is_supported = processor.is_format_supported(filename)
        status = "✓" if is_supported else "✗"
        print(f"  {status} {filename}")


async def demo_error_handling():
    """Demo 7: Error handling"""
    print("\n" + "="*60)
    print("DEMO 7: Error Handling")
    print("="*60)

    processor = DocumentProcessor()

    # Test 1: Nonexistent file
    print("\nTest 1: Nonexistent file")
    try:
        await processor.process_document("/nonexistent/file.txt")
        print("  ✗ Should have raised FileNotFoundError")
    except FileNotFoundError as e:
        print(f"  ✓ Caught FileNotFoundError: {str(e)[:50]}...")

    # Test 2: Unsupported format
    print("\nTest 2: Unsupported format")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        f.write(b"test content")
        temp_path = f.name

    try:
        await processor.process_document(temp_path)
        print("  ✗ Should have raised UnsupportedFormatError")
    except Exception as e:
        print(f"  ✓ Caught {type(e).__name__}: {str(e)[:50]}...")
    finally:
        Path(temp_path).unlink()

    # Test 3: File too large
    print("\nTest 3: File size limit")
    config = DocumentConfig(max_file_size=100)  # 100 bytes
    processor = DocumentProcessor(config)

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"x" * 1000)  # 1000 bytes
        temp_path = f.name

    try:
        await processor.process_document(temp_path)
        print("  ✗ Should have raised error for large file")
    except Exception as e:
        print(f"  ✓ Caught error: {str(e)[:50]}...")
    finally:
        Path(temp_path).unlink()


async def main():
    """Run all demos"""
    print("\n" + "🚀" * 30)
    print("DOCUMENT PROCESSING PIPELINE - DEMO")
    print("Phase 4: Comprehensive Document Extraction")
    print("🚀" * 30)

    demos = [
        ("Basic Processing", demo_basic_processing),
        ("Markdown Structure", demo_markdown_structure),
        ("Custom Configuration", demo_custom_config),
        ("Preset Configurations", demo_preset_configs),
        ("Metadata Extraction", demo_metadata_extraction),
        ("Format Detection", demo_format_detection),
        ("Error Handling", demo_error_handling),
    ]

    for name, demo_func in demos:
        try:
            await demo_func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {str(e)}")

    print("\n" + "="*60)
    print("DEMO COMPLETE ✨")
    print("="*60)
    print("\nFor more information, see:")
    print("  • docs/PHASE4_DELIVERABLES.md")
    print("  • app/documents/README (in __init__.py)")
    print("  • tests/documents/ for usage examples")
    print()


if __name__ == "__main__":
    asyncio.run(main())
