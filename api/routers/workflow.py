"""
Workflow Router - Endpoints for workflow execution.

Provides:
- POST /workflow/run - Execute workflow and return latest message
- GET /workflow/status/{task_id} - Check workflow status
- GET /workflow/stages/{workflow_id} - List stages for workflow
"""

import asyncio
import tempfile
import uuid
from pathlib import Path
from fastapi import UploadFile, File
from datetime import datetime
from typing import Optional, Dict, Any, Annotated
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, status
from loguru import logger

from langchain.messages import HumanMessage, AIMessage

from schemas import (
    WorkflowRequest,
    WorkflowResponse,
    WorkflowMessage,
    WorkflowStatus,
    WorkflowCreate,
    ErrorResponse,
    DataFileUploadResponse
)
from dependencies import (
    RAGManager,
    WorkflowManager,
    OutputManagerRegistry,
    settings,
)
from auth import (
    get_current_user_optional,
    get_current_active_user,
    User,
)


router = APIRouter(prefix="/workflow", tags=["Workflow"])


# Task tracking
_running_tasks: Dict[str, Dict[str, Any]] = {}


TEMP_DATA_DIR = Path(tempfile.gettempdir()) / "workflow_data"
TEMP_DATA_DIR.mkdir(exist_ok=True)

# Supported data file extensions
SUPPORTED_DATA_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".parquet", ".tsv"}

# =============================================================================
# DATA FILE UPLOAD
# =============================================================================

@router.post(
    "/upload-data",
    response_model=DataFileUploadResponse,
    summary="Upload data file for workflow analysis",
    description="Upload a data file (CSV, Excel, JSON, Parquet) and get a temporary path for use in workflow execution.",
)
async def upload_data_file(
    current_user: Annotated[User, Depends(get_current_active_user)],
    file: UploadFile = File(..., description="Data file to upload"),
) -> DataFileUploadResponse:
    """
    Upload a data file and return a temporary path for workflow use.
    
    The returned `file_path` can be used as the `data_path` argument
    in the `/workflow/run` endpoint.
    
    Supported formats:
    - CSV (.csv)
    - TSV (.tsv)
    - Excel (.xlsx, .xls)
    - JSON (.json)
    - Parquet (.parquet)
    
    Files are stored temporarily and may be cleaned up periodically.
    """
    filename = file.filename or "data_file"
    extension = Path(filename).suffix.lower()
    
    # Validate file type
    if extension not in SUPPORTED_DATA_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension}. Supported: {', '.join(sorted(SUPPORTED_DATA_EXTENSIONS))}"
        )
    
    # Create unique subdirectory for this upload
    upload_id = uuid.uuid4().hex[:12]
    upload_dir = TEMP_DATA_DIR / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Save file
    file_path = upload_dir / filename
    
    try:
        content = await file.read()
        file_size = len(content)
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info(f"📁 Data file uploaded: {file_path} ({file_size} bytes)")
        
        return DataFileUploadResponse(
            success=True,
            filename=filename,
            file_path=str(file_path),
            file_size=file_size,
            content_type=file.content_type or "application/octet-stream",
            message=f"File uploaded successfully. Use this path as data_path in /workflow/run"
        )
        
    except Exception as e:
        # Clean up on failure
        if upload_dir.exists():
            import shutil
            shutil.rmtree(upload_dir, ignore_errors=True)
        
        logger.error(f"❌ Data file upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.delete(
    "/data/{upload_id}",
    summary="Delete uploaded data file",
    description="Clean up a previously uploaded data file.",
)
async def delete_data_file(
    upload_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> dict:
    """Delete an uploaded data file by its upload ID."""
    import shutil
    
    upload_dir = TEMP_DATA_DIR / upload_id
    
    if not upload_dir.exists():
        raise HTTPException(status_code=404, detail="Upload not found")
    
    try:
        shutil.rmtree(upload_dir)
        logger.info(f"🗑️ Deleted upload: {upload_id}")
        return {"success": True, "message": f"Upload {upload_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")
    

# =============================================================================
# WORKFLOW EXECUTION
# =============================================================================


@router.post(
    "/run",
    response_model=WorkflowResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Execute workflow with query",
    description="Run the data analysis workflow with the given query and return the result.",
)
async def run_workflow(
    request: WorkflowRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> WorkflowResponse:
    """
    Execute the workflow and return the most recent message.

    This endpoint:
    1. Initializes RAG context for the workflow
    2. Runs the LangGraph workflow (gather → plan → execute/answer → summarize)
    3. Returns the summary or error message

    Requires authentication.
    """
    workflow = WorkflowManager.get_workflow()
    if not workflow:
        raise HTTPException(status_code=503, detail="Workflow not available")

    executor = WorkflowManager.get_executor()
    output_mgr = OutputManagerRegistry.get_output_manager(request.workflow_id)
    rag = await RAGManager.get_rag(request.workflow_id)

    # Build initial state
    initial_state = {
        "messages": [HumanMessage(content=request.query)],
        "data_path": request.data_path or "",
        "stage_name": request.stage_name,
        "workflow_id": request.workflow_id,
        "web_search_enabled": request.web_search_enabled,
    }

    # Build dependencies
    deps = {
        "executor": executor,
        "output_manager": output_mgr,
        "rag": rag,
        "llm": settings.DEFAULT_MODEL,
        "code_llm": settings.CODE_MODEL,
        "vision_llm": settings.VISION_MODEL,
        "base_url": settings.OLLAMA_BASE_URL,
        "max_retries": 3,
    }

    try:
        logger.info(f"🚀 Running workflow: {request.workflow_id}/{request.stage_name}")
        logger.info(f"   Query: {request.query[:100]}...")

        start_time = datetime.now()

        # Execute workflow
        result = await workflow.ainvoke(initial_state, context=deps)

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ Workflow completed in {elapsed:.2f}s")

        # Extract results
        summary = result.get("summary", "")
        action = result.get("action", "unknown")
        code = result.get("code", "")
        error = result.get("error", "")

        # Get plots if any
        plots = []
        context = result.get("context", {})
        if isinstance(context, dict):
            plots = context.get("plots", [])

        # Build response message
        if error:
            message_content = (
                f"Error: {error}\n\n{summary}" if summary else f"Error: {error}"
            )
            status = "error"
        else:
            message_content = summary or "Workflow completed"
            status = "completed"

        return WorkflowResponse(
            workflow_id=request.workflow_id,
            stage_name=request.stage_name,
            status=status,
            message=WorkflowMessage(
                role="assistant",
                content=message_content,
                timestamp=datetime.now(),
                metadata={"action": action, "elapsed_seconds": elapsed},
            ),
            action_taken=action,
            code_executed=code if code else None,
            plots=plots if plots else None,
            summary=summary if summary else None,
            error=error if error else None,
        )

    except Exception as e:
        logger.error(f"❌ Workflow failed: {e}")
        import traceback

        traceback.print_exc()

        return WorkflowResponse(
            workflow_id=request.workflow_id,
            stage_name=request.stage_name,
            status="error",
            message=WorkflowMessage(
                role="assistant",
                content=f"Workflow execution failed: {str(e)}",
                timestamp=datetime.now(),
            ),
            error=str(e),
        )


# =============================================================================
# ASYNC WORKFLOW (Background Task)
# =============================================================================


# @router.post(
#     "/run-async",
#     response_model=WorkflowStatus,
#     summary="Start workflow asynchronously",
#     description="Start workflow in background and return task ID for status polling.",
# )
# async def run_workflow_async(
#     request: WorkflowRequest,
#     background_tasks: BackgroundTasks,
#     current_user: Annotated[User, Depends(get_current_active_user)],
# ) -> WorkflowStatus:
#     """
#     Start workflow execution in background.

#     Returns task ID for polling status via /workflow/status/{task_id}

#     Requires authentication.
#     """
#     task_id = f"{request.workflow_id}_{request.stage_name}_{datetime.now().strftime('%H%M%S%f')}"

#     # Initialize task tracking
#     _running_tasks[task_id] = {
#         "workflow_id": request.workflow_id,
#         "stage_name": request.stage_name,
#         "status": "starting",
#         "current_node": None,
#         "progress": 0.0,
#         "start_time": datetime.now(),
#         "result": None,
#         "error": None,
#     }

#     async def run_in_background():
#         try:
#             _running_tasks[task_id]["status"] = "running"

#             # Run workflow (reuse logic from run_workflow)
#             workflow = WorkflowManager.get_workflow()
#             executor = WorkflowManager.get_executor()
#             output_mgr = OutputManagerRegistry.get_output_manager(request.workflow_id)
#             rag = await RAGManager.get_rag(request.workflow_id)

#             initial_state = {
#                 "messages": [HumanMessage(content=request.query)],
#                 "data_path": request.data_path or "",
#                 "stage_name": request.stage_name,
#                 "workflow_id": request.workflow_id,
#                 "web_search_enabled": request.web_search_enabled,
#             }

#             deps = {
#                 "executor": executor,
#                 "output_manager": output_mgr,
#                 "rag": rag,
#                 "llm": settings.DEFAULT_MODEL,
#                 "base_url": settings.OLLAMA_BASE_URL,
#             }

#             result = await workflow.ainvoke(initial_state, context=deps)

#             _running_tasks[task_id]["status"] = "completed"
#             _running_tasks[task_id]["result"] = result
#             _running_tasks[task_id]["progress"] = 1.0

#         except Exception as e:
#             _running_tasks[task_id]["status"] = "error"
#             _running_tasks[task_id]["error"] = str(e)
#             logger.error(f"Background task {task_id} failed: {e}")

#     # Schedule background execution
#     background_tasks.add_task(run_in_background)

#     return WorkflowStatus(
#         workflow_id=request.workflow_id,
#         stage_name=request.stage_name,
#         status="starting",
#         progress=0.0,
#     )


@router.get(
    "/status/{task_id}",
    response_model=WorkflowStatus,
    summary="Get workflow task status",
)
async def get_workflow_status(task_id: str) -> WorkflowStatus:
    """Get status of a background workflow task."""
    if task_id not in _running_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = _running_tasks[task_id]
    elapsed = (datetime.now() - task["start_time"]).total_seconds()

    return WorkflowStatus(
        workflow_id=task["workflow_id"],
        stage_name=task["stage_name"],
        status=task["status"],
        current_node=task.get("current_node"),
        progress=task["progress"],
        elapsed_seconds=elapsed,
    )


@router.get(
    "/result/{task_id}",
    response_model=WorkflowResponse,
    summary="Get completed workflow result",
)
async def get_workflow_result(task_id: str) -> WorkflowResponse:
    """Get result of a completed background workflow task."""
    if task_id not in _running_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = _running_tasks[task_id]

    if task["status"] not in ("completed", "error"):
        raise HTTPException(
            status_code=400, detail=f"Task not complete. Status: {task['status']}"
        )

    if task["status"] == "error":
        return WorkflowResponse(
            workflow_id=task["workflow_id"],
            stage_name=task["stage_name"],
            status="error",
            message=WorkflowMessage(
                role="assistant",
                content=f"Error: {task['error']}",
                timestamp=datetime.now(),
            ),
            error=task["error"],
        )

    result = task["result"]

    return WorkflowResponse(
        workflow_id=task["workflow_id"],
        stage_name=task["stage_name"],
        status="completed",
        message=WorkflowMessage(
            role="assistant",
            content=result.get("summary", "Completed"),
            timestamp=datetime.now(),
        ),
        action_taken=result.get("action"),
        code_executed=result.get("code"),
        summary=result.get("summary"),
    )


# =============================================================================
# WORKFLOW INFO
# =============================================================================


@router.get(
    "/stages/{workflow_id}",
    summary="List workflow stages",
    description="Get all stages for a workflow with file counts.",
)
async def list_workflow_stages(workflow_id: str) -> list:
    """List all stages for a workflow."""
    from pathlib import Path

    workflow_dir = Path(settings.RESULTS_DIR) / workflow_id
    if not workflow_dir.exists():
        return []

    stages = []
    for stage_dir in sorted(workflow_dir.iterdir()):
        if not stage_dir.is_dir():
            continue

        stage_info = {
            "stage_name": stage_dir.name,
            "path": str(stage_dir),
            "has_plots": False,
            "plot_count": 0,
            "has_code": False,
            "code_count": 0,
        }

        plots_dir = stage_dir / "plots"
        if plots_dir.exists():
            plots = list(plots_dir.glob("*.png")) + list(plots_dir.glob("*.jpg"))
            stage_info["has_plots"] = len(plots) > 0
            stage_info["plot_count"] = len(plots)

        code_files = list(stage_dir.glob("*.py"))
        stage_info["has_code"] = len(code_files) > 0
        stage_info["code_count"] = len(code_files)

        stages.append(stage_info)

    return stages


# @router.get(
#     "/list",
#     summary="List all workflows",
# )
# async def list_workflows() -> list:
#     """List all available workflows."""
#     from pathlib import Path

#     results_dir = Path(settings.RESULTS_DIR)
#     if not results_dir.exists():
#         return []

#     workflows = []
#     for workflow_dir in sorted(results_dir.iterdir()):
#         if workflow_dir.is_dir():
#             stage_count = len([d for d in workflow_dir.iterdir() if d.is_dir()])
#             workflows.append(
#                 {
#                     "workflow_id": workflow_dir.name,
#                     "stage_count": stage_count,
#                 }
#             )

#     return workflows


@router.get(
    "/list",
    response_model=list,  # Will be list[WorkflowInfo]
    summary="List all workflows",
)
async def list_workflows() -> list:
    """
    List all workflows from database.
    
    Merges database metadata with filesystem stage counts.
    """
    from database import get_workflow_db
    
    db = get_workflow_db()
    results_dir = Path(settings.RESULTS_DIR)
    
    # Sync any new filesystem workflows
    db.sync_filesystem(settings.RESULTS_DIR)
    
    workflows = []
    for record in db.list_all():
        workflow_dir = results_dir / record.workflow_id
        
        # Count stages from filesystem
        stage_count = 0
        has_results = False
        if workflow_dir.exists():
            has_results = True
            stage_count = len([d for d in workflow_dir.iterdir() if d.is_dir()])
        
        workflows.append({
            "workflow_id": record.workflow_id,
            "name": record.name,
            "description": record.description,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "stage_count": stage_count,
            "has_results": has_results,
        })
    
    return workflows

@router.post(
    "/create",
    response_model=dict,  # Will be WorkflowInfo
    status_code=status.HTTP_201_CREATED,
    summary="Create a new workflow",
)
async def create_workflow(
    request: WorkflowCreate,  # Import from schemas
    current_user: Annotated["User", Depends(get_current_active_user)],
) -> dict:
    """
    Create a new workflow.
    
    Creates database record and optional results directory.
    """
    from database import get_workflow_db
    import re
    
    
    # Validate workflow_id format
    if not re.match(r"^[a-zA-Z0-9_-]+$", request.workflow_id):
        raise HTTPException(
            status_code=400,
            detail="workflow_id must contain only letters, numbers, hyphens, and underscores",
        )
    
    db = get_workflow_db()
    
    # Check if exists
    if db.exists(request.workflow_id):
        raise HTTPException(
            status_code=409,
            detail=f"Workflow '{request.workflow_id}' already exists",
        )
    
    # Create database record
    try:
        record = db.create(
            workflow_id=request.workflow_id,
            name=request.name,
            description=request.description,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    
    # Create results directory
    results_dir = Path(settings.RESULTS_DIR) / request.workflow_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 Created workflow: {request.workflow_id} by {current_user.username}")
    
    return {
        "workflow_id": record.workflow_id,
        "name": record.name,
        "description": record.description,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "stage_count": 0,
        "has_results": False,
    }