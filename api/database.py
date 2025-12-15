"""
Database - SQLite storage for workflow metadata.

Syncs with filesystem workflows on startup.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional
from contextlib import contextmanager
from loguru import logger
from pydantic import BaseModel


# =============================================================================
# MODELS
# =============================================================================


class WorkflowRecord(BaseModel):
    """Workflow database record."""
    workflow_id: str
    name: str
    description: Optional[str] = None
    created_at: str
    updated_at: str


# =============================================================================
# DATABASE
# =============================================================================


class WorkflowDatabase:
    """SQLite database for workflow metadata."""

    def __init__(self, db_path: str = "data/workflows.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """Initialize database schema."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workflows (
                    workflow_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.commit()
        logger.info(f"📦 Workflow database ready: {self.db_path}")

    def create(
        self,
        workflow_id: str,
        name: str,
        description: Optional[str] = None,
    ) -> WorkflowRecord:
        """Create a new workflow record."""
        now = datetime.utcnow().isoformat()

        with self._connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO workflows (workflow_id, name, description, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (workflow_id, name, description, now, now),
                )
                conn.commit()
            except sqlite3.IntegrityError:
                raise ValueError(f"Workflow '{workflow_id}' already exists")

        logger.info(f"✅ Created workflow: {workflow_id}")
        return WorkflowRecord(
            workflow_id=workflow_id,
            name=name,
            description=description,
            created_at=now,
            updated_at=now,
        )

    def get(self, workflow_id: str) -> Optional[WorkflowRecord]:
        """Get workflow by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM workflows WHERE workflow_id = ?",
                (workflow_id,),
            ).fetchone()

        if row:
            return WorkflowRecord(**dict(row))
        return None

    def list_all(self) -> list[WorkflowRecord]:
        """List all workflows."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM workflows ORDER BY created_at DESC"
            ).fetchall()

        return [WorkflowRecord(**dict(row)) for row in rows]

    def exists(self, workflow_id: str) -> bool:
        """Check if workflow exists."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM workflows WHERE workflow_id = ?",
                (workflow_id,),
            ).fetchone()
        return row is not None

    def sync_filesystem(self, results_dir: str):
        """
        Sync database with filesystem workflows.
        
        Adds any filesystem workflows not in DB (for backwards compatibility).
        """
        results_path = Path(results_dir)
        if not results_path.exists():
            return

        synced = 0
        for workflow_dir in results_path.iterdir():
            if not workflow_dir.is_dir():
                continue

            workflow_id = workflow_dir.name
            if not self.exists(workflow_id):
                # Add filesystem workflow to database
                created_at = datetime.fromtimestamp(
                    workflow_dir.stat().st_ctime
                ).isoformat()

                with self._connect() as conn:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO workflows 
                        (workflow_id, name, description, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (workflow_id, workflow_id, None, created_at, created_at),
                    )
                    conn.commit()
                synced += 1

        if synced:
            logger.info(f"🔄 Synced {synced} filesystem workflows to database")


# =============================================================================
# SINGLETON
# =============================================================================

_db: Optional[WorkflowDatabase] = None


def get_workflow_db() -> WorkflowDatabase:
    """Get or create the workflow database singleton."""
    global _db
    if _db is None:
        _db = WorkflowDatabase()
    return _db