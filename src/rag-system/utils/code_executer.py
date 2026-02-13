"""
Code Executor with Output Capture

This module extends the base code executor to:
- Capture all console output (stdout/stderr)
- Save generated files (plots, data, models)
- Track execution results
- Integrate with OutputManager for structured storage
"""

import sys
import tempfile
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel
from loguru import logger

MATPLOTLIB_STYLE_SETUP = '''
# Auto-generated: Configure matplotlib styling
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

# Professional color palette
_PALETTE = [
    "#667eea", "#48bb78", "#ed8936", "#f56565", "#38b2ac",
    "#9f7aea", "#ed64a6", "#4299e1", "#ecc94b", "#718096",
]

# Apply professional styling
_style = {{
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "figure.autolayout": True,
    "axes.facecolor": "white",
    "axes.edgecolor": "#718096",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.labelsize": 11,
    "axes.labelcolor": "#2d3748",
    "axes.labelweight": "medium",
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.titlecolor": "#2d3748",
    "axes.titlepad": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": cycler("color", _PALETTE),
    "grid.color": "#e2e8f0",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.7,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "patch.linewidth": 0,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.facecolor": "white",
    "legend.edgecolor": "#e2e8f0",
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.color": "#718096",
    "ytick.color": "#718096",
    "font.family": "sans-serif",
    "font.size": 10,
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
}}
for _k, _v in _style.items():
    try:
        mpl.rcParams[_k] = _v
    except KeyError:
        pass

import os
os.chdir('{working_dir}')

'''

class ExecutionResult(BaseModel):
    """Result of code execution with comprehensive output tracking"""
    success: bool
    stdout: str
    stderr: str
    error: Optional[str]
    execution_time_seconds: float
    generated_files: list[str] = []
    working_dir: Optional[str] = None


class OutputCapturingExecutor:
    """
    Code executor that captures all outputs for downstream use.
    
    Features:
    - Executes Python code in a sandboxed environment
    - Captures stdout and stderr
    - Tracks generated files (plots, data files, models)
    - Provides execution results for saving via OutputManager
    - Passes data between agents via pickle files
    """
    
    def __init__(
        self,
        timeout_seconds: int = 300,
        max_output_lines: int = 10000
    ):
        """
        Initialize executor.
        
        Args:
            timeout_seconds: Maximum execution time
            max_output_lines: Maximum lines of output to capture
        """
        self.timeout_seconds = timeout_seconds
        self.max_output_lines = max_output_lines
        logger.info(f"OutputCapturingExecutor initialized (timeout={timeout_seconds}s)")
    
    async def execute_code(
        self,
        code: str,
        working_dir: Optional[Path] = None,
        input_data: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Execute Python code and capture all outputs.
        
        Args:
            code: Python code to execute
            working_dir: Directory for execution (creates temp if None)
            input_data: Optional data to pass to code via pickle
            
        Returns:
            ExecutionResult with outputs and generated files
        """
        start_time = datetime.now()
        
        # Create working directory if needed
        if working_dir is None:
            working_dir = Path(tempfile.mkdtemp(prefix="agent_exec_"))
        else:
            working_dir = Path(working_dir).resolve()  # Convert to absolute path
            working_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"🔧 Executing code in: {working_dir}")
        
        # Track files before execution
        files_before = set(working_dir.rglob("*"))
        
        try:
            # Prepare code with data loading if needed
            if input_data:
                code = self._wrap_code_with_data(code, input_data, working_dir)
            
            # Add output directory setup to code
            code = self._add_output_setup(code, working_dir)
            
            # Save code to temporary file
            code_file = working_dir / "_agent_code.py"
            with open(code_file, 'w') as f:
                f.write(code)
            
            # Execute in subprocess
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(code_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(working_dir)
            )
            
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout_seconds
                )
                
                stdout = stdout_bytes.decode('utf-8', errors='replace')
                stderr = stderr_bytes.decode('utf-8', errors='replace')
                
                # Truncate if too long
                stdout = self._truncate_output(stdout)
                stderr = self._truncate_output(stderr)
                
                success = process.returncode == 0
                error = None if success else f"Exit code: {process.returncode}"
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                
                stdout = ""
                stderr = ""
                success = False
                error = f"Execution timeout after {self.timeout_seconds}s"
            
            # Track generated files
            files_after = set(working_dir.rglob("*"))
            generated_files = [
                str(f.relative_to(working_dir))
                for f in (files_after - files_before)
                if f.is_file() and not f.name.startswith('_agent_')
            ]
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = ExecutionResult(
                success=success,
                stdout=stdout,
                stderr=stderr,
                error=error,
                execution_time_seconds=execution_time,
                generated_files=generated_files,
                working_dir=str(working_dir)
            )
            
            if success:
                logger.info(f"✅ Execution successful ({execution_time:.2f}s)")
                logger.info(f"   Generated {len(generated_files)} files")
            else:
                logger.error(f"❌ Execution failed: {error}")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ Execution error: {str(e)}")
            logger.debug(e, exc_info=True)
            
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                error=f"Execution exception: {str(e)}",
                execution_time_seconds=execution_time,
                generated_files=[],
                working_dir=str(working_dir)
            )
    
    def _wrap_code_with_data(
        self,
        code: str,
        input_data: Dict[str, Any],
        working_dir: Path
    ) -> str:
        """
        Wrap code to load input data from pickle file.
        
        This enables passing data between agents.
        """
        import pickle
        
        # Ensure absolute path
        working_dir = Path(working_dir).resolve()
        
        # Save data to pickle
        data_file = working_dir / "_input_data.pkl"
        with open(data_file, 'wb') as f:
            pickle.dump(input_data, f)
        
        wrapped_code = f"""
# Auto-generated: Load input data
import pickle
import sys

try:
    with open('{data_file}', 'rb') as f:
        __INPUT_DATA__ = pickle.load(f)
    
    # Make variables available in global scope
    for __key__, __value__ in __INPUT_DATA__.items():
        globals()[__key__] = __value__
    
    print(f"✓ Loaded input data: {{list(__INPUT_DATA__.keys())}}")
except Exception as e:
    print(f"Warning: Could not load input data: {{e}}", file=sys.stderr)

# User code begins here
{code}
"""
        return wrapped_code
    
    def _add_output_setup(self, code: str, working_dir: Path) -> str:
        """
        Add matplotlib configuration to save plots automatically.
        """
        # Ensure absolute path
        from pathlib import Path
        working_dir = Path(working_dir).resolve()
        
        setup_code = MATPLOTLIB_STYLE_SETUP.format(working_dir=working_dir)
        logger.info(setup_code)
        return setup_code + code
    
    def _truncate_output(self, text: str) -> str:
        """Truncate output if too long"""
        lines = text.split('\n')
        if len(lines) > self.max_output_lines:
            kept_lines = lines[:self.max_output_lines]
            return '\n'.join(kept_lines) + f"\n\n[... truncated {len(lines) - self.max_output_lines} lines ...]"
        return text
    
    async def execute_with_output_manager(
        self,
        code: str,
        stage_name: str,
        output_manager: Any,  # OutputManager instance
        stage_order: Optional[int] = None,
        input_data: Optional[Dict[str, Any]] = None,
        code_filename: str = "code.py"
    ) -> ExecutionResult:
        """
        Execute code and automatically save all outputs via OutputManager.
        
        This is the high-level method that agents should use.
        
        Args:
            code: Python code to execute
            stage_name: Name of the workflow stage
            output_manager: OutputManager instance for saving results
            stage_order: Optional stage order number
            input_data: Optional data to pass to code
            code_filename: Name for saved code file
            
        Returns:
            ExecutionResult with execution status and outputs
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"EXECUTING: {stage_name}")
        logger.info(f"{'='*70}")
        
        # Get working directory from output manager
        stage_dir = output_manager.get_stage_dir(stage_name, stage_order)
        working_dir = (stage_dir / "execution").resolve()  # Use absolute path
        working_dir.mkdir(exist_ok=True, parents=True)
        
        # Execute code
        result = await self.execute_code(code, working_dir, input_data)
        
        # Save all outputs via output manager
        logger.info("📦 Saving outputs...")
        
        # 1. Save the code
        output_manager.save_code(stage_name, code, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{code_filename}", stage_order)
        
        # 2. Save console output
        output_manager.save_console_output(
            stage_name,
            result.stdout,
            result.stderr,
            stage_order
        )
        
        # 3. Save execution metadata
        execution_info = {
            "success": result.success,
            "execution_time_seconds": result.execution_time_seconds,
            "generated_files": result.generated_files,
            "error": result.error,
            "timestamp": datetime.now().isoformat()
        }
        output_manager.save_json(
            stage_name,
            execution_info,
            "execution_info.json",
            stage_order
        )
        
        # 4. Copy generated files to output structure
        if result.generated_files:
            logger.info(f"📁 Copying {len(result.generated_files)} generated files...")
            output_manager.copy_generated_files(
                stage_name,
                Path(result.working_dir),
                stage_order,
                ['*.png', '*.jpg', '*.csv', '*.pkl', '*.txt', '*.json', '*.html']
            )
        
        output_manager.save_manifest()
        logger.info(f"✅ All outputs saved for {stage_name}\n")
        
        return result


# ============================================================================
# STANDALONE TESTING
# ============================================================================

async def test_executor():
    """Test the output capturing executor"""
    
    from variable_importance.utils.output_manager import OutputManager

    print("\n" + "="*70)
    print("TESTING OUTPUT CAPTURING EXECUTOR")
    print("="*70)
    
    # Create output manager
    output_mgr = OutputManager(workflow_id="test_workflow")
    
    # Create executor
    executor = OutputCapturingExecutor()
    
    # Test 1: Simple code execution
    print("\n1. Testing simple execution...")
    code = """
import numpy as np
import matplotlib.pyplot as plt

print("Generating sample data...")
x = np.linspace(0, 10, 100)
y = np.sin(x)

print(f"Data shape: {x.shape}")
print(f"Mean: {y.mean():.4f}")

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title("Sine Wave")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.savefig("sine_wave.png")
plt.close()

print("✓ Plot saved")
"""
    
    result = await executor.execute_with_output_manager(
        code=code,
        stage_name="test_stage",
        output_manager=output_mgr,
        stage_order=1,
        code_filename="test_code.py"
    )
    
    print(f"Success: {result.success}")
    print(f"Generated files: {result.generated_files}")
    
    # Test 2: Code with input data
    print("\n2. Testing execution with input data...")
    code_with_input = """
print("Loaded data from previous stage:")
print(f"  - df shape: {df.shape}")
print(f"  - columns: {list(df.columns)}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
df.plot()
plt.title("Data Visualization")
plt.savefig("data_plot.png")
plt.close()

print("✓ Visualization saved")
"""
    
    # Simulate data from previous stage
    import pandas as pd
    input_data = {
        "df": pd.DataFrame({
            "x": range(10),
            "y": range(10, 20)
        })
    }
    
    result2 = await executor.execute_with_output_manager(
        code=code_with_input,
        stage_name="test_with_data",
        output_manager=output_mgr,
        stage_order=2,
        input_data=input_data,
        code_filename="code_with_data.py"
    )
    
    print(f"Success: {result2.success}")
    print(f"Generated files: {result2.generated_files}")
    
    # Save manifest
    output_mgr.save_manifest()
    output_mgr.create_summary_report()
    
    print(f"\n✅ Test complete! Results in: {output_mgr.workflow_dir}")


if __name__ == "__main__":
    asyncio.run(test_executor())