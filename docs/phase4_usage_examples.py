"""
Phase 4: Multi-Modal Storage Usage Examples

Demonstrates how to use the multi-modal storage system for
images, documents, and audio files.
"""

import asyncio
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.storage import (
    file_manager,
    metadata_store,
    multimodal_indexer,
    storage_config
)


async def example_1_save_and_retrieve_image():
    """Example 1: Save and retrieve an image file"""

    print("\n" + "="*60)
    print("Example 1: Save and Retrieve Image")
    print("="*60 + "\n")

    # Initialize
    await file_manager.initialize()

    # Simulate image upload (in practice, this would be from a file upload)
    image_data = b"fake_image_data_here"  # Replace with actual image bytes

    # Save image
    result = await file_manager.save_file(
        file_data=image_data,
        original_filename="vacation_photo.jpg",
        file_type="image",
        session_id="session_001",
        metadata={
            "source": "user_upload",
            "camera": "iPhone 13",
            "location": "Beach"
        }
    )

    print(f"✅ Image saved:")
    print(f"   File ID: {result['file_id']}")
    print(f"   Path: {result['stored_path']}")
    print(f"   Size: {result['file_size']} bytes")
    print(f"   Deduplicated: {result['deduplicated']}")
    print()

    # Retrieve image
    retrieved_data = await file_manager.get_file(result['file_id'])
    print(f"✅ Image retrieved: {len(retrieved_data)} bytes")
    print()

    return result['file_id']


async def example_2_index_image_with_vision_analysis():
    """Example 2: Index image with vision analysis results"""

    print("\n" + "="*60)
    print("Example 2: Index Image with Vision Analysis")
    print("="*60 + "\n")

    # Initialize
    await multimodal_indexer.initialize()

    # First save the image (reuse from example 1)
    image_data = b"fake_image_data"
    result = await file_manager.save_file(
        file_data=image_data,
        original_filename="cat_photo.jpg",
        file_type="image",
        session_id="session_002"
    )

    file_id = result['file_id']

    # Simulate vision API analysis results
    vision_analysis = {
        "description": "A tabby cat sitting on a wooden fence in a garden",
        "objects": ["cat", "fence", "garden", "plants", "grass"],
        "labels": [
            {"name": "cat", "confidence": 0.98},
            {"name": "tabby", "confidence": 0.87},
            {"name": "fence", "confidence": 0.92},
            {"name": "outdoor", "confidence": 0.95}
        ],
        "colors": ["brown", "green", "gray"],
        "safe_search": {"adult": "unlikely", "violence": "unlikely"}
    }

    # Index the image
    await multimodal_indexer.index_image(
        file_id=file_id,
        vision_analysis=vision_analysis,
        session_id="session_002",
        metadata={"analyzed_by": "google_vision_api"}
    )

    print(f"✅ Image indexed:")
    print(f"   File ID: {file_id}")
    print(f"   Objects detected: {', '.join(vision_analysis['objects'])}")
    print(f"   Description: {vision_analysis['description']}")
    print()

    return file_id


async def example_3_index_pdf_document():
    """Example 3: Index a PDF document with extracted text"""

    print("\n" + "="*60)
    print("Example 3: Index PDF Document")
    print("="*60 + "\n")

    # Initialize
    await file_manager.initialize()
    await multimodal_indexer.initialize()

    # Simulate PDF upload
    pdf_data = b"fake_pdf_binary_data"
    result = await file_manager.save_file(
        file_data=pdf_data,
        original_filename="machine_learning_paper.pdf",
        file_type="pdf",
        session_id="session_003",
        metadata={
            "source": "research",
            "category": "AI"
        }
    )

    file_id = result['file_id']

    # Simulate text extraction (in practice, use PyPDF2 or similar)
    extracted_text = """
    Machine Learning: A Comprehensive Overview

    Machine learning is a subset of artificial intelligence that enables
    systems to learn and improve from experience without being explicitly
    programmed. It focuses on the development of computer programs that
    can access data and use it to learn for themselves.

    Neural networks are a key component of deep learning, which is a
    subset of machine learning. They are inspired by the structure and
    function of biological neural networks in the human brain.

    Applications of machine learning include natural language processing,
    computer vision, recommendation systems, and autonomous vehicles.
    """

    # Index the document
    await multimodal_indexer.index_document(
        file_id=file_id,
        extracted_text=extracted_text,
        session_id="session_003",
        document_metadata={
            "pages": 1,
            "author": "AI Researcher",
            "title": "Machine Learning: A Comprehensive Overview"
        }
    )

    print(f"✅ PDF indexed:")
    print(f"   File ID: {file_id}")
    print(f"   Text length: {len(extracted_text)} characters")
    print(f"   Chunks created: ~{len(extracted_text) // 1000 + 1}")
    print()

    return file_id


async def example_4_search_similar_files():
    """Example 4: Search for similar files using semantic search"""

    print("\n" + "="*60)
    print("Example 4: Search Similar Files")
    print("="*60 + "\n")

    # Initialize
    await multimodal_indexer.initialize()

    # First, index some files (reuse from previous examples)
    await example_2_index_image_with_vision_analysis()
    await example_3_index_pdf_document()

    # Search for files about "cats"
    print("🔍 Searching for: 'cats in outdoor settings'")
    results = await multimodal_indexer.search_similar_files(
        query="cats in outdoor settings",
        file_type="image",
        n_results=5
    )

    print(f"\n✅ Found {len(results)} similar files:")
    for i, result in enumerate(results, 1):
        print(f"\n   {i}. Score: {result['score']:.3f}")
        if 'file_metadata' in result:
            fm = result['file_metadata']
            print(f"      File: {fm.get('original_filename', 'N/A')}")
            print(f"      Type: {fm.get('file_type', 'N/A')}")
            print(f"      Uploaded: {fm.get('uploaded_at', 'N/A')}")

    print()

    # Search for documents about "machine learning"
    print("🔍 Searching for: 'machine learning and neural networks'")
    results = await multimodal_indexer.search_similar_files(
        query="machine learning and neural networks",
        file_type=None,  # Search all types
        n_results=5
    )

    print(f"\n✅ Found {len(results)} similar files")
    print()


async def example_5_deduplication():
    """Example 5: Demonstrate file deduplication"""

    print("\n" + "="*60)
    print("Example 5: File Deduplication")
    print("="*60 + "\n")

    # Initialize
    await file_manager.initialize()

    # Upload same file twice
    file_data = b"exact_same_file_content_12345"

    # First upload
    result1 = await file_manager.save_file(
        file_data=file_data,
        original_filename="document_v1.txt",
        file_type="txt",
        session_id="session_004"
    )

    print(f"✅ First upload:")
    print(f"   File ID: {result1['file_id']}")
    print(f"   Hash: {result1['file_hash'][:16]}...")
    print(f"   Deduplicated: {result1['deduplicated']}")
    print()

    # Second upload (same content, different name)
    result2 = await file_manager.save_file(
        file_data=file_data,
        original_filename="document_v2_copy.txt",
        file_type="txt",
        session_id="session_005"
    )

    print(f"✅ Second upload:")
    print(f"   File ID: {result2['file_id']}")
    print(f"   Hash: {result2['file_hash'][:16]}...")
    print(f"   Deduplicated: {result2['deduplicated']}")

    if result2['deduplicated']:
        print(f"   ⚡ Reused existing file: {result2['original_file_id']}")
        print(f"   💾 Disk space saved: {len(file_data)} bytes")

    print()


async def example_6_list_and_cleanup():
    """Example 6: List files and perform cleanup"""

    print("\n" + "="*60)
    print("Example 6: List Files and Cleanup")
    print("="*60 + "\n")

    # Initialize
    await file_manager.initialize()

    # List all files
    all_files = await file_manager.list_files(limit=10)
    print(f"📁 Total files: {len(all_files)}")
    for file in all_files[:5]:  # Show first 5
        print(f"   - {file['original_filename']} ({file['file_type']}, {file['file_size']} bytes)")
    print()

    # List files by session
    session_files = await file_manager.list_files(session_id="session_002")
    print(f"📁 Files in session_002: {len(session_files)}")
    print()

    # List files by type
    image_files = await file_manager.list_files(file_type="image")
    print(f"🖼️  Image files: {len(image_files)}")
    print()

    # Get storage statistics
    stats = await file_manager.get_storage_stats()
    print(f"📊 Storage statistics:")
    print(f"   Total files: {stats.get('total_files', 0)}")
    print(f"   Total size: {stats.get('total_bytes', 0) / (1024*1024):.2f} MB")
    print(f"   Unique sessions: {stats.get('unique_sessions', 0)}")
    if 'by_type' in stats:
        print(f"   By type:")
        for type_stat in stats['by_type']:
            print(f"      - {type_stat['file_type']}: {type_stat['count']} files")
    print()

    # Dry run cleanup
    print("🧹 Running cleanup (dry run)...")
    cleanup_stats = await file_manager.cleanup_old_files(dry_run=True)
    print(f"   Would delete: {cleanup_stats['deleted_count']} files")
    print(f"   Would free: {cleanup_stats['freed_mb']:.2f} MB")
    print()


async def example_7_knowledge_graph_integration():
    """Example 7: Integration with knowledge graph"""

    print("\n" + "="*60)
    print("Example 7: Knowledge Graph Integration")
    print("="*60 + "\n")

    # Initialize
    await file_manager.initialize()
    await metadata_store.initialize()
    await multimodal_indexer.initialize()

    # Save and index a document with specific concepts
    doc_data = b"fake_document_data"
    result = await file_manager.save_file(
        file_data=doc_data,
        original_filename="ai_concepts.txt",
        file_type="txt",
        session_id="session_006"
    )

    file_id = result['file_id']

    # Index with extracted concepts
    text = "Artificial intelligence and machine learning are transforming healthcare"
    await multimodal_indexer.index_document(
        file_id=file_id,
        extracted_text=text,
        session_id="session_006"
    )

    # Get linked concepts
    concepts = await metadata_store.get_file_concepts(file_id)

    print(f"✅ File indexed with concepts:")
    print(f"   File ID: {file_id}")
    print(f"   Linked concepts: {len(concepts)}")
    for concept in concepts[:5]:
        print(f"      - {concept['concept_name']} (confidence: {concept['confidence']:.2f})")
    print()


async def main():
    """Run all examples"""

    print("\n" + "="*70)
    print(" "*15 + "PHASE 4: MULTI-MODAL STORAGE EXAMPLES")
    print("="*70)

    try:
        # Run examples
        await example_1_save_and_retrieve_image()
        await example_2_index_image_with_vision_analysis()
        await example_3_index_pdf_document()
        await example_4_search_similar_files()
        await example_5_deduplication()
        await example_6_list_and_cleanup()
        await example_7_knowledge_graph_integration()

        print("\n" + "="*70)
        print(" "*20 + "ALL EXAMPLES COMPLETED")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
