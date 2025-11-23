"""Add sessions table for persistent session metadata

Revision ID: 002_add_session_metadata
Revises: 001_initial_schema
Create Date: 2024-01-14

PATTERN: Schema evolution migration
WHY: Support persistent session tracking and cleanup jobs
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '002_add_session_metadata'
down_revision: Union[str, None] = '001_initial_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Add sessions table for tracking session lifecycle.

    This enables:
    - Persistent session metadata beyond Redis TTL
    - Session cleanup job tracking
    - Analytics and reporting
    """
    op.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
            ended_at DATETIME,
            exchange_count INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            metadata TEXT,
            status TEXT DEFAULT 'active' CHECK(status IN ('active', 'ended', 'expired', 'cleaned'))
        )
    """)

    # Index for cleanup queries (find expired sessions)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_sessions_status_activity
        ON sessions(status, last_activity)
    """)

    # Index for session lookups
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_sessions_session_id
        ON sessions(session_id)
    """)

    # Add foreign key reference in captures (SQLite doesn't enforce, but documents intent)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_captures_session_id
        ON captures(session_id)
    """)


def downgrade() -> None:
    """Remove sessions table and related indexes."""
    op.execute("DROP INDEX IF EXISTS idx_captures_session_id")
    op.execute("DROP INDEX IF EXISTS idx_sessions_session_id")
    op.execute("DROP INDEX IF EXISTS idx_sessions_status_activity")
    op.execute("DROP TABLE IF EXISTS sessions")
