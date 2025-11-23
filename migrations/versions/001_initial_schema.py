"""Initial schema with captures table and FTS5

Revision ID: 001_initial_schema
Revises:
Create Date: 2024-01-14

PATTERN: Baseline migration capturing existing schema
WHY: Enables future schema changes with version control
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Create initial database schema.

    Tables:
    - captures: Main conversation storage
    - captures_fts: FTS5 virtual table for search
    - Triggers: Keep FTS5 in sync with captures
    """
    # Main captures table
    op.execute("""
        CREATE TABLE IF NOT EXISTS captures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            user_text TEXT NOT NULL,
            agent_text TEXT NOT NULL,
            metadata TEXT
        )
    """)

    # FTS5 virtual table for full-text search
    op.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS captures_fts
        USING fts5(
            session_id UNINDEXED,
            user_text,
            agent_text,
            content=captures,
            content_rowid=id
        )
    """)

    # Trigger: Insert into FTS after captures insert
    op.execute("""
        CREATE TRIGGER IF NOT EXISTS captures_ai
        AFTER INSERT ON captures BEGIN
            INSERT INTO captures_fts(rowid, session_id, user_text, agent_text)
            VALUES (new.id, new.session_id, new.user_text, new.agent_text);
        END
    """)

    # Trigger: Delete from FTS after captures delete
    op.execute("""
        CREATE TRIGGER IF NOT EXISTS captures_ad
        AFTER DELETE ON captures BEGIN
            DELETE FROM captures_fts WHERE rowid = old.id;
        END
    """)

    # Trigger: Update FTS after captures update
    op.execute("""
        CREATE TRIGGER IF NOT EXISTS captures_au
        AFTER UPDATE ON captures BEGIN
            UPDATE captures_fts
            SET user_text = new.user_text, agent_text = new.agent_text
            WHERE rowid = new.id;
        END
    """)

    # Index for session queries
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_session_timestamp
        ON captures(session_id, timestamp DESC)
    """)


def downgrade() -> None:
    """
    Drop all tables and triggers.
    WARNING: This will delete all data.
    """
    op.execute("DROP TRIGGER IF EXISTS captures_au")
    op.execute("DROP TRIGGER IF EXISTS captures_ad")
    op.execute("DROP TRIGGER IF EXISTS captures_ai")
    op.execute("DROP TABLE IF EXISTS captures_fts")
    op.execute("DROP INDEX IF EXISTS idx_session_timestamp")
    op.execute("DROP TABLE IF EXISTS captures")
