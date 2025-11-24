#!/usr/bin/env python3
"""
Phase 4 Storage Migration Script

Applies Phase 4 database schema for multi-modal storage
"""

import asyncio
import aiosqlite
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.logger import get_logger

logger = get_logger(__name__)


async def run_migration():
    """Run Phase 4 database migration"""

    migration_file = Path(__file__).parent / "phase4_storage_migration.sql"
    db_path = "./data/storage_metadata.db"

    if not migration_file.exists():
        logger.error("migration_file_not_found", path=str(migration_file))
        return False

    try:
        logger.info("phase4_migration_started", db_path=db_path)

        # Read migration SQL
        with open(migration_file, 'r') as f:
            migration_sql = f.read()

        # Execute migration
        async with aiosqlite.connect(db_path) as db:
            # Split by semicolon and execute each statement
            statements = [s.strip() for s in migration_sql.split(';') if s.strip()]

            for i, statement in enumerate(statements):
                # Skip comments
                if statement.startswith('--'):
                    continue

                try:
                    await db.execute(statement)
                    logger.debug(f"migration_statement_executed", index=i+1)
                except Exception as e:
                    # Some statements may fail if objects already exist
                    if "already exists" not in str(e):
                        logger.warning(
                            "migration_statement_warning",
                            index=i+1,
                            error=str(e)
                        )

            await db.commit()

        # Verify migration
        async with aiosqlite.connect(db_path) as db:
            # Check tables
            cursor = await db.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table'
                AND name IN ('multimodal_files', 'file_analysis', 'file_concept_links')
            """)
            tables = [row[0] for row in await cursor.fetchall()]

            if len(tables) == 3:
                logger.info(
                    "phase4_migration_complete",
                    tables_created=tables
                )
                return True
            else:
                logger.error(
                    "phase4_migration_incomplete",
                    tables_found=tables,
                    expected=3
                )
                return False

    except Exception as e:
        logger.error(
            "phase4_migration_failed",
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        return False


async def verify_schema():
    """Verify Phase 4 schema is correctly installed"""

    db_path = "./data/storage_metadata.db"

    try:
        async with aiosqlite.connect(db_path) as db:
            # Count tables
            cursor = await db.execute("""
                SELECT COUNT(*) FROM sqlite_master
                WHERE type='table'
                AND name LIKE 'multimodal_%' OR name LIKE 'file_%'
            """)
            table_count = (await cursor.fetchone())[0]

            # Count indexes
            cursor = await db.execute("""
                SELECT COUNT(*) FROM sqlite_master
                WHERE type='index'
                AND name LIKE 'idx_file_%' OR name LIKE 'idx_analysis_%'
            """)
            index_count = (await cursor.fetchone())[0]

            # Count triggers
            cursor = await db.execute("""
                SELECT COUNT(*) FROM sqlite_master
                WHERE type='trigger'
                AND name LIKE 'file_analysis_%'
            """)
            trigger_count = (await cursor.fetchone())[0]

            logger.info(
                "schema_verification_complete",
                tables=table_count,
                indexes=index_count,
                triggers=trigger_count
            )

            return table_count >= 3 and index_count >= 5 and trigger_count >= 3

    except Exception as e:
        logger.error("schema_verification_failed", error=str(e))
        return False


async def main():
    """Main migration function"""

    print("=" * 60)
    print("Phase 4: Multi-Modal Storage and Indexing Migration")
    print("=" * 60)
    print()

    # Create data directory
    Path("./data").mkdir(exist_ok=True)

    # Run migration
    print("Running migration...")
    success = await run_migration()

    if not success:
        print("❌ Migration failed!")
        return 1

    print("✅ Migration completed successfully")
    print()

    # Verify schema
    print("Verifying schema...")
    verified = await verify_schema()

    if not verified:
        print("⚠️  Schema verification failed!")
        return 1

    print("✅ Schema verified successfully")
    print()

    print("=" * 60)
    print("Phase 4 migration complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Initialize storage components:")
    print("   from app.storage import file_manager, metadata_store, multimodal_indexer")
    print("   await file_manager.initialize()")
    print("   await metadata_store.initialize()")
    print("   await multimodal_indexer.initialize()")
    print()
    print("2. Start uploading multi-modal files")
    print("3. Run tests: pytest tests/storage/")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
