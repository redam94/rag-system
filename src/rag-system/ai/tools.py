"""
Agent Tools - File search and workflow utilities.

Provides tools for agents to:
- Search files in workflow directory
- Read file contents
- List available artifacts
- Query previous outputs
"""

import os
import fnmatch
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger


@dataclass
class FileInfo:
    """Information about a file."""
    name: str
    path: str
    size: int
    extension: str
    relative_path: str


class WorkflowFileTools:
    """
    File tools scoped to a workflow directory.
    
    Provides safe file access within the workflow's results folder.
    """
    
    def __init__(self, workflow_dir: Path):
        self.workflow_dir = Path(workflow_dir).resolve()
        if not self.workflow_dir.exists():
            self.workflow_dir.mkdir(parents=True, exist_ok=True)
    
    def list_files(
        self,
        pattern: str = "*",
        extensions: Optional[List[str]] = None,
        stage: Optional[str] = None,
        recursive: bool = True,
    ) -> List[FileInfo]:
        """
        List files in the workflow directory.
        
        Args:
            pattern: Glob pattern (e.g., "*.csv", "plot_*")
            extensions: Filter by extensions (e.g., [".csv", ".png"])
            stage: Filter to specific stage directory
            recursive: Search subdirectories
            
        Returns:
            List of FileInfo objects
        """
        search_dir = self.workflow_dir
        if stage:
            # Find stage directory (may have numeric prefix)
            for d in self.workflow_dir.iterdir():
                if d.is_dir() and stage in d.name:
                    search_dir = d
                    break
        
        files = []
        glob_method = search_dir.rglob if recursive else search_dir.glob
        
        for file_path in glob_method(pattern):
            if not file_path.is_file():
                continue
            
            if extensions:
                if file_path.suffix.lower() not in [e.lower() for e in extensions]:
                    continue
            
            try:
                rel_path = file_path.relative_to(self.workflow_dir) if stage else file_path.relative_to(self.workflow_dir)
                files.append(FileInfo(
                    name=file_path.name,
                    path=str(file_path),
                    size=file_path.stat().st_size,
                    extension=file_path.suffix,
                    relative_path=str(rel_path),
                ))
            except Exception as e:
                logger.warning(f"Error reading file info: {e}")
        
        return files
    
    def search_files(
        self,
        query: str,
        extensions: Optional[List[str]] = None,
    ) -> List[FileInfo]:
        """
        Search for files by name containing query string.
        
        Args:
            query: Search string (case-insensitive)
            extensions: Filter by extensions
            
        Returns:
            Matching files
        """
        all_files = self.list_files(extensions=extensions)
        query_lower = query.lower()
        
        return [f for f in all_files if query_lower in f.name.lower()]
    
    def read_file(
        self,
        file_path: str,
        max_chars: int = 10000,
    ) -> Optional[str]:
        """
        Read file contents safely.
        
        Args:
            file_path: Path to file (relative or absolute)
            max_chars: Maximum characters to read
            
        Returns:
            File contents or None if not readable
        """
        # Resolve path
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workflow_dir / path
        
        path = path.resolve()
        
        # Security: ensure file is within workflow directory
        try:
            path.relative_to(self.workflow_dir)
        except ValueError:
            logger.warning(f"Attempted to read file outside workflow: {path}")
            return None
        
        if not path.exists():
            return None
        
        # Read based on extension
        try:
            if path.suffix.lower() in [".csv", ".txt", ".md", ".py", ".json"]:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read(max_chars)
                    if len(content) == max_chars:
                        content += "\n... [truncated]"
                    return content
            else:
                return f"[Binary file: {path.name}, size: {path.stat().st_size} bytes]"
        except Exception as e:
            logger.error(f"Error reading file {path}: {e}")
            return None
    
    def get_stage_files(self, stage: str) -> Dict[str, List[FileInfo]]:
        """
        Get all files for a stage, organized by type.
        
        Returns:
            Dict with keys: plots, data, code, models, other
        """
        result = {
            "plots": [],
            "data": [],
            "code": [],
            "models": [],
            "other": [],
        }
        
        files = self.list_files(stage=stage)
        
        for f in files:
            ext = f.extension.lower()
            if ext in [".png", ".jpg", ".jpeg", ".svg"]:
                result["plots"].append(f)
            elif ext in [".csv", ".json", ".xlsx", ".parquet"]:
                result["data"].append(f)
            elif ext in [".py"]:
                result["code"].append(f)
            elif ext in [".pkl", ".joblib", ".h5", ".pt"]:
                result["models"].append(f)
            else:
                result["other"].append(f)
        
        return result
    
    def get_latest_output(self, stage: str) -> Optional[str]:
        """Get the latest console output for a stage."""
        stage_files = self.list_files(
            pattern="console_output_*.txt",
            stage=stage,
        )
        
        if not stage_files:
            return None
        
        # Sort by name (timestamp) and get latest
        latest = sorted(stage_files, key=lambda f: f.name)[-1]
        return self.read_file(latest.path)
    
    def format_file_listing(self, files: List[FileInfo], max_files: int = 20) -> str:
        """Format file list for inclusion in prompts."""
        if not files:
            return "No files found."
        
        lines = [f"Found {len(files)} files:"]
        for f in files[:max_files]:
            size_kb = f.size / 1024
            lines.append(f"  - {f.relative_path} ({size_kb:.1f} KB)")
        
        if len(files) > max_files:
            lines.append(f"  ... and {len(files) - max_files} more")
        
        return "\n".join(lines)


def create_file_tools_for_code(workflow_dir: Path) -> str:
    """
    Generate Python code that provides file tools within executed code.
    
    This is injected into generated code to give the code agent
    access to workflow files.
    """
    return f'''
# === File Tools (auto-injected) ===
import os
from pathlib import Path

WORKFLOW_DIR = Path("{workflow_dir}")

def list_workflow_files(pattern="*", stage=None):
    """List files in the workflow directory."""
    search_dir = WORKFLOW_DIR
    if stage:
        for d in WORKFLOW_DIR.iterdir():
            if d.is_dir() and stage in d.name:
                search_dir = d
                break
    return list(search_dir.rglob(pattern))

def read_workflow_file(relative_path, max_chars=10000):
    """Read a file from the workflow directory."""
    path = WORKFLOW_DIR / relative_path
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read(max_chars)

def get_data_files(stage=None):
    """Get all CSV/JSON data files."""
    return list_workflow_files("*.csv", stage) + list_workflow_files("*.json", stage)

def get_previous_output(stage):
    """Get the latest console output from a stage."""
    files = sorted(list_workflow_files(f"console_output_*.txt", stage))
    if files:
        return read_workflow_file(files[-1].relative_to(WORKFLOW_DIR))
    return None

# === End File Tools ===
'''


# =============================================================================
# TOOL DEFINITIONS FOR LANGCHAIN
# =============================================================================

def make_file_search_tool(workflow_dir: Path):
    """Create a LangChain-compatible file search tool."""
    tools = WorkflowFileTools(workflow_dir)
    
    def search_files(query: str, extensions: str = "") -> str:
        """
        Search for files in the workflow directory.
        
        Args:
            query: Search term to match in file names
            extensions: Comma-separated extensions (e.g., ".csv,.json")
        
        Returns:
            List of matching files with paths
        """
        ext_list = [e.strip() for e in extensions.split(",") if e.strip()] or None
        files = tools.search_files(query, ext_list)
        return tools.format_file_listing(files)
    
    return search_files


def make_file_read_tool(workflow_dir: Path):
    """Create a LangChain-compatible file read tool."""
    tools = WorkflowFileTools(workflow_dir)
    
    def read_file(file_path: str) -> str:
        """
        Read contents of a file in the workflow directory.
        
        Args:
            file_path: Relative path to the file
        
        Returns:
            File contents or error message
        """
        content = tools.read_file(file_path)
        if content is None:
            return f"Error: Could not read file '{file_path}'"
        return content
    
    return read_file


def make_list_stages_tool(workflow_dir: Path):
    """Create a tool to list workflow stages."""
    
    def list_stages() -> str:
        """
        List all stages in the current workflow.
        
        Returns:
            List of stage directories with file counts
        """
        workflow_path = Path(workflow_dir)
        if not workflow_path.exists():
            return "No workflow directory found."
        
        stages = []
        for d in sorted(workflow_path.iterdir()):
            if d.is_dir():
                file_count = sum(1 for _ in d.rglob("*") if _.is_file())
                stages.append(f"  - {d.name}: {file_count} files")
        
        if not stages:
            return "No stages found in workflow."
        
        return "Workflow stages:\n" + "\n".join(stages)
    
    return list_stages