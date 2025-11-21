#!/usr/bin/env python3
"""
Phase 4 Verification Script

Verifies that all Phase 4 components are correctly installed and functional.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def verify_imports():
    """Verify all Phase 4 modules can be imported"""
    print("=" * 60)
    print("1. Verifying Imports")
    print("=" * 60)

    try:
        from app.storage import (
            storage_config,
            metadata_store,
            file_manager,
            multimodal_indexer,
            StorageConfig,
            MetadataStore,
            FileManager,
            MultiModalIndexer
        )
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


async def verify_configuration():
    """Verify storage configuration"""
    print("\n" + "=" * 60)
    print("2. Verifying Configuration")
    print("=" * 60)

    from app.storage import storage_config

    print(f"✅ Base directory: {storage_config.base_directory}")
    print(f"✅ Metadata DB: {storage_config.metadata_db_path}")
    print(f"✅ Max user storage: {storage_config.max_storage_per_user_gb} GB")
    print(f"✅ Retention days: {storage_config.retention_days}")
    print(f"✅ Deduplication: {storage_config.deduplication_enabled}")

    # Check file types
    print(f"\n✅ Configured file types: {len(storage_config.file_types)}")
    for file_type, config in storage_config.file_types.items():
        print(f"   - {file_type}: {config.max_size_mb}MB, {config.retention_days} days")

    return True


async def verify_database_schema():
    """Verify database schema is installed"""
    print("\n" + "=" * 60)
    print("3. Verifying Database Schema")
    print("=" * 60)

    import aiosqlite

    db_path = "./data/storage_metadata.db"

    # Check if migration has been run
    if not Path(db_path).exists():
        print(f"⚠️  Database not found: {db_path}")
        print("   Run: python scripts/run_phase4_migration.py")
        return False

    try:
        async with aiosqlite.connect(db_path) as db:
            # Check tables
            cursor = await db.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table'
                AND name IN ('multimodal_files', 'file_analysis', 'file_concept_links')
            """)
            tables = [row[0] for row in await cursor.fetchall()]

            if len(tables) == 3:
                print(f"✅ All tables present: {', '.join(tables)}")
            else:
                print(f"❌ Missing tables. Found: {tables}")
                return False

            # Check indexes
            cursor = await db.execute("""
                SELECT COUNT(*) FROM sqlite_master
                WHERE type='index'
                AND (name LIKE 'idx_file_%' OR name LIKE 'idx_analysis_%' OR name LIKE 'idx_concept_%')
            """)
            index_count = (await cursor.fetchone())[0]
            print(f"✅ Indexes found: {index_count}")

            # Check triggers
            cursor = await db.execute("""
                SELECT COUNT(*) FROM sqlite_master
                WHERE type='trigger'
                AND name LIKE 'file_analysis_%'
            """)
            trigger_count = (await cursor.fetchone())[0]
            print(f"✅ Triggers found: {trigger_count}")

            return True

    except Exception as e:
        print(f"❌ Database verification failed: {e}")
        return False


async def verify_file_manager():
    """Verify file manager initialization"""
    print("\n" + "=" * 60)
    print("4. Verifying File Manager")
    print("=" * 60)

    try:
        from app.storage import file_manager

        await file_manager.initialize()
        print("✅ File manager initialized")

        # Check base directory exists
        base_path = Path(file_manager.config.base_directory)
        if base_path.exists():
            print(f"✅ Base directory exists: {base_path}")
        else:
            print(f"⚠️  Base directory not found: {base_path}")

        return True

    except Exception as e:
        print(f"❌ File manager verification failed: {e}")
        return False


async def verify_metadata_store():
    """Verify metadata store initialization"""
    print("\n" + "=" * 60)
    print("5. Verifying Metadata Store")
    print("=" * 60)

    try:
        from app.storage import metadata_store

        await metadata_store.initialize()
        print("✅ Metadata store initialized")

        # Get stats
        stats = await metadata_store.get_storage_stats()
        print(f"✅ Current stats:")
        print(f"   - Files: {stats.get('total_files', 0)}")
        print(f"   - Sessions: {stats.get('unique_sessions', 0)}")
        print(f"   - Total size: {stats.get('total_bytes', 0)} bytes")

        return True

    except Exception as e:
        print(f"❌ Metadata store verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def verify_indexer():
    """Verify indexer initialization"""
    print("\n" + "=" * 60)
    print("6. Verifying Indexer")
    print("=" * 60)

    try:
        from app.storage import multimodal_indexer

        # Check if vector store is available
        vector_enabled = multimodal_indexer.enable_vector
        kg_enabled = multimodal_indexer.enable_kg

        print(f"✅ Vector indexing: {vector_enabled}")
        print(f"✅ Knowledge graph: {kg_enabled}")

        if not vector_enabled and not kg_enabled:
            print("⚠️  Both vector and KG disabled - indexer will have limited functionality")

        # Initialize (may fail if dependencies not available)
        try:
            await multimodal_indexer.initialize()
            print("✅ Indexer initialized")
        except Exception as e:
            print(f"⚠️  Indexer initialization warning: {e}")
            print("   (This is OK if you haven't set up vector store or Neo4j)")

        return True

    except Exception as e:
        print(f"❌ Indexer verification failed: {e}")
        return False


async def verify_basic_workflow():
    """Test a basic file upload workflow"""
    print("\n" + "=" * 60)
    print("7. Testing Basic Workflow")
    print("=" * 60)

    try:
        from app.storage import file_manager

        # Initialize
        await file_manager.initialize()

        # Create test file
        test_data = b"Phase 4 verification test file content"

        # Save file
        result = await file_manager.save_file(
            file_data=test_data,
            original_filename="test_verification.txt",
            file_type="txt",
            session_id="verification_test"
        )

        print(f"✅ File saved:")
        print(f"   - ID: {result['file_id']}")
        print(f"   - Size: {result['file_size']} bytes")
        print(f"   - Deduplicated: {result['deduplicated']}")

        # Retrieve file
        retrieved_data = await file_manager.get_file(result['file_id'])
        if retrieved_data == test_data:
            print(f"✅ File retrieved successfully")
        else:
            print(f"❌ File data mismatch")
            return False

        # Get metadata
        metadata = await file_manager.get_file_metadata(result['file_id'])
        if metadata:
            print(f"✅ Metadata retrieved")
        else:
            print(f"❌ Metadata not found")
            return False

        # Delete test file
        deleted = await file_manager.delete_file(result['file_id'])
        if deleted:
            print(f"✅ File deleted successfully")
        else:
            print(f"❌ File deletion failed")
            return False

        return True

    except Exception as e:
        print(f"❌ Basic workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def check_dependencies():
    """Check required dependencies"""
    print("\n" + "=" * 60)
    print("8. Checking Dependencies")
    print("=" * 60)

    dependencies = {
        "aiosqlite": "aiosqlite",
        "aiofiles": "aiofiles",
        "chromadb": "chromadb (optional for vector search)",
        "sentence-transformers": "sentence_transformers (optional for embeddings)",
        "neo4j": "neo4j (optional for knowledge graph)"
    }

    all_ok = True
    for module_name, display_name in dependencies.items():
        try:
            __import__(module_name.replace("-", "_"))
            print(f"✅ {display_name}")
        except ImportError:
            if "optional" in display_name:
                print(f"⚠️  {display_name} - not installed (optional)")
            else:
                print(f"❌ {display_name} - not installed (required)")
                all_ok = False

    return all_ok


async def main():
    """Run all verification checks"""
    print("\n" + "=" * 70)
    print(" " * 20 + "PHASE 4 VERIFICATION")
    print("=" * 70 + "\n")

    results = []

    # Run all checks
    results.append(("Imports", await verify_imports()))
    results.append(("Configuration", await verify_configuration()))
    results.append(("Database Schema", await verify_database_schema()))
    results.append(("File Manager", await verify_file_manager()))
    results.append(("Metadata Store", await verify_metadata_store()))
    results.append(("Indexer", await verify_indexer()))
    results.append(("Basic Workflow", await verify_basic_workflow()))
    results.append(("Dependencies", await check_dependencies()))

    # Summary
    print("\n" + "=" * 70)
    print(" " * 25 + "VERIFICATION SUMMARY")
    print("=" * 70 + "\n")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:.<50} {status}")

    print("\n" + "=" * 70)
    print(f"Total: {passed}/{total} checks passed")
    print("=" * 70 + "\n")

    if passed == total:
        print("🎉 Phase 4 verification COMPLETE!")
        print("\nNext steps:")
        print("1. Run usage examples: python docs/phase4_usage_examples.py")
        print("2. Create tests: pytest tests/storage/")
        print("3. Integrate with main application")
        return 0
    else:
        print("⚠️  Some checks failed. Please review errors above.")
        print("\nCommon issues:")
        print("1. Run migration: python scripts/run_phase4_migration.py")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Check configuration in app/storage/config.py")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
