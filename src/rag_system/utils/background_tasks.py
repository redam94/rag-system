"""
Background Task Executor for Non-Blocking Operations

Allows chat queries to run in the background while users navigate the app.

IMPORTANT: Task status is persisted to disk so task info survives page refreshes.
"""

import asyncio
import threading
import json
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from pathlib import Path
from loguru import logger
from enum import Enum


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BackgroundTask:
    """Represents a background task with status tracking."""
    
    def __init__(self, task_id: str, description: str):
        self.task_id = task_id
        self.description = description
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        self.started_at = None
        self.completed_at = None
        self.progress = 0.0
        self.current_step = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for session state storage."""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "status": self.status.value,
            "result": None,  # Don't serialize result (may not be JSON-safe)
            "error": str(self.error) if self.error else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "current_step": self.current_step
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BackgroundTask':
        """Create task from dictionary."""
        task = cls(data["task_id"], data["description"])
        task.status = TaskStatus(data["status"])
        task.error = data.get("error")
        task.progress = data.get("progress", 0.0)
        task.current_step = data.get("current_step", "")
        
        if data.get("started_at"):
            task.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            task.completed_at = datetime.fromisoformat(data["completed_at"])
        
        return task


class BackgroundTaskManager:
    """
    Manages background task execution for non-blocking operations.
    
    Features:
    - Run async workflows in background threads
    - Track task status and progress
    - Persist task status to disk (survives page refresh!)
    - Store results in memory (lost on refresh - rerun if needed)
    - Allow navigation while tasks run
    """
    
    def __init__(self, cache_dir: str = "cache/tasks"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.tasks: Dict[str, BackgroundTask] = {}
        self.executor = None
        
        # Load persisted task status
        self._load_task_status()
        
        logger.info("🔧 BackgroundTaskManager initialized")
    
    def _get_status_file(self) -> Path:
        """Get path to task status file."""
        return self.cache_dir / "task_status.json"
    
    def _load_task_status(self):
        """Load task status from disk."""
        status_file = self._get_status_file()
        
        if status_file.exists():
            try:
                with open(status_file, 'r') as f:
                    data = json.load(f)
                
                for task_data in data:
                    task = BackgroundTask.from_dict(task_data)
                    self.tasks[task.task_id] = task
                
                logger.info(f"📥 Loaded {len(self.tasks)} task statuses from disk")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load task status: {e}")
    
    def _save_task_status(self):
        """Save task status to disk."""
        status_file = self._get_status_file()
        
        try:
            data = [task.to_dict() for task in self.tasks.values()]
            
            with open(status_file, 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            logger.error(f"❌ Failed to save task status: {e}")
    
    def submit_task(
        self,
        task_id: str,
        description: str,
        async_func: Callable,
        *args,
        **kwargs
    ) -> BackgroundTask:
        """
        Submit an async task to run in the background.
        
        Args:
            task_id: Unique identifier for the task
            description: Human-readable description
            async_func: Async function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            BackgroundTask object for tracking
        """
        task = BackgroundTask(task_id, description)
        self.tasks[task_id] = task
        
        # Save status immediately
        self._save_task_status()
        
        # Run in background thread
        thread = threading.Thread(
            target=self._run_async_task,
            args=(task, async_func, args, kwargs),
            daemon=True
        )
        thread.start()
        
        logger.info(f"🚀 Submitted background task: {task_id}")
        return task
    
    def _run_async_task(
        self,
        task: BackgroundTask,
        async_func: Callable,
        args: tuple,
        kwargs: dict
    ):
        """Run async function in a new event loop (for threading)."""
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            self._save_task_status()  # Save running status
            
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(async_func(*args, **kwargs))
                task.result = result
                task.status = TaskStatus.COMPLETED
                logger.info(f"✅ Task completed: {task.task_id}")
            finally:
                loop.close()
                
        except Exception as e:
            task.error = e
            task.status = TaskStatus.FAILED
            logger.error(f"❌ Task failed: {task.task_id} - {str(e)}")
            logger.debug(e, exc_info=True)
        
        finally:
            task.completed_at = datetime.now()
            self._save_task_status()  # Save final status
    
    def get_task(self, task_id: str) -> Optional[BackgroundTask]:
        """Get task by ID."""
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> Dict[str, BackgroundTask]:
        """Get all tasks."""
        return self.tasks
    
    def is_running(self, task_id: str) -> bool:
        """Check if task is currently running."""
        task = self.tasks.get(task_id)
        return task.status == TaskStatus.RUNNING if task else False
    
    def is_completed(self, task_id: str) -> bool:
        """Check if task completed successfully."""
        task = self.tasks.get(task_id)
        return task.status == TaskStatus.COMPLETED if task else False
    
    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Remove old completed/failed tasks."""
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        
        to_remove = []
        for task_id, task in self.tasks.items():
            if task.completed_at and task.completed_at < cutoff:
                to_remove.append(task_id)
        
        for task_id in to_remove:
            del self.tasks[task_id]
        
        if to_remove:
            self._save_task_status()  # Persist cleanup
            logger.info(f"🗑️ Cleaned up {len(to_remove)} old tasks")


# Singleton instance for app-wide use
_task_manager = None

def get_task_manager() -> BackgroundTaskManager:
    """Get or create the singleton task manager."""
    global _task_manager
    if _task_manager is None:
        _task_manager = BackgroundTaskManager()
    return _task_manager