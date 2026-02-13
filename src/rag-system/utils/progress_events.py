"""
Progress Event System

Streams workflow progress updates to the UI via file-based events.
Enables real-time status updates during background task execution.
"""

import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from loguru import logger


class EventType(Enum):
    STAGE_START = "stage_start"
    STAGE_END = "stage_end"
    PROGRESS = "progress"
    CODE_GENERATED = "code_generated"
    CODE_EXECUTING = "code_executing"
    CODE_OUTPUT = "code_output"
    ERROR = "error"
    RAG_QUERY = "rag_query"
    WEB_SEARCH = "web_search"
    PLOT_ANALYZED = "plot_analyzed"
    SUMMARY = "summary"


@dataclass
class ProgressEvent:
    """Single progress event."""
    event_type: str
    stage: str
    message: str
    timestamp: str
    data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "stage": self.stage,
            "message": self.message,
            "timestamp": self.timestamp,
            "data": self.data or {}
        }


class ProgressEmitter:
    """
    Emits progress events to a file for UI consumption.
    
    Uses JSON Lines format for easy streaming reads.
    """
    
    def __init__(self, task_id: str, cache_dir: str = "cache/progress"):
        self.task_id = task_id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.event_file = self.cache_dir / f"{task_id}_events.jsonl"
        self.event_count = 0
        
        # Clear old events
        if self.event_file.exists():
            self.event_file.unlink()
        
        logger.debug(f"📡 ProgressEmitter initialized: {self.event_file}")
    
    def emit(
        self,
        event_type: EventType,
        stage: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """Emit a progress event."""
        event = ProgressEvent(
            event_type=event_type.value,
            stage=stage,
            message=message,
            timestamp=datetime.now().isoformat(),
            data=data
        )
        
        try:
            with open(self.event_file, 'a') as f:
                f.write(json.dumps(event.to_dict()) + "\n")
            
            self.event_count += 1
            logger.info(f"📡 [{stage}] {message}")
            
        except Exception as e:
            logger.error(f"Failed to emit event: {e}")
    
    def stage_start(self, stage: str, description: str = ""):
        """Emit stage start event."""
        self.emit(EventType.STAGE_START, stage, f"Starting: {description or stage}")
    
    def stage_end(self, stage: str, success: bool = True):
        """Emit stage end event."""
        status = "completed" if success else "failed"
        self.emit(EventType.STAGE_END, stage, f"Stage {status}", {"success": success})
    
    def progress(self, stage: str, message: str, percent: Optional[float] = None):
        """Emit general progress update."""
        data = {"percent": percent} if percent is not None else None
        self.emit(EventType.PROGRESS, stage, message, data)
    
    def code_generated(self, stage: str, code_preview: str):
        """Emit code generation event."""
        self.emit(EventType.CODE_GENERATED, stage, "Code generated", {"preview": code_preview[:200]})
    
    def code_output(self, stage: str, output: str, is_error: bool = False):
        """Emit code execution output."""
        self.emit(
            EventType.ERROR if is_error else EventType.CODE_OUTPUT,
            stage,
            "Error output" if is_error else "Execution output",
            {"output": output[:500]}
        )
    
    def web_search(self, stage: str, query: str, results_count: int):
        """Emit web search event."""
        self.emit(EventType.WEB_SEARCH, stage, f"Web search: {query}", {"results": results_count})
    
    def rag_query(self, stage: str, found_count: int):
        """Emit RAG query event."""
        self.emit(EventType.RAG_QUERY, stage, f"Found {found_count} relevant context chunks")
    
    def plot_analyzed(self, stage: str, plot_name: str, insight: str):
        """Emit plot analysis event."""
        self.emit(EventType.PLOT_ANALYZED, stage, f"Analyzed: {plot_name}", {"insight": insight[:200]})


class ProgressReader:
    """
    Reads progress events from file.
    
    Used by UI to display real-time updates.
    """
    
    def __init__(self, task_id: str, cache_dir: str = "cache/progress"):
        self.task_id = task_id
        self.cache_dir = Path(cache_dir)
        self.event_file = self.cache_dir / f"{task_id}_events.jsonl"
        self.last_read_line = 0
    
    def get_all_events(self) -> List[Dict[str, Any]]:
        """Get all events for the task."""
        if not self.event_file.exists():
            return []
        
        events = []
        try:
            with open(self.event_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        except Exception as e:
            logger.warning(f"Error reading events: {e}")
        
        return events
    
    def get_new_events(self) -> List[Dict[str, Any]]:
        """Get only new events since last read."""
        all_events = self.get_all_events()
        new_events = all_events[self.last_read_line:]
        self.last_read_line = len(all_events)
        return new_events
    
    def get_latest_status(self) -> Optional[Dict[str, Any]]:
        """Get the latest status summary."""
        events = self.get_all_events()
        if not events:
            return None
        
        # Find current stage
        current_stage = None
        stage_status = {}
        
        for event in events:
            stage = event.get("stage", "unknown")
            event_type = event.get("event_type")
            
            if event_type == "stage_start":
                current_stage = stage
                stage_status[stage] = "running"
            elif event_type == "stage_end":
                stage_status[stage] = "completed" if event.get("data", {}).get("success") else "failed"
        
        return {
            "current_stage": current_stage,
            "stage_status": stage_status,
            "total_events": len(events),
            "latest_message": events[-1].get("message") if events else None,
            "latest_event": events[-1] if events else None
        }
    
    def cleanup(self):
        """Remove event file."""
        if self.event_file.exists():
            self.event_file.unlink()


def get_emitter(task_id: str) -> ProgressEmitter:
    """Get or create progress emitter for a task."""
    return ProgressEmitter(task_id)


def get_reader(task_id: str) -> ProgressReader:
    """Get progress reader for a task."""
    return ProgressReader(task_id)