"""
Output Manager - Centralized Results Storage System

This module provides comprehensive output management for all agents:
- Creates organized results folder structure
- Captures and saves code execution outputs
- Saves generated plots and files
- Tracks console output (stdout/stderr)
- Creates execution manifests
- Ensures data flows between agents via saved outputs
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger
import sys
from contextlib import contextmanager
from io import StringIO


class OutputManager:
    """
    Manages output storage for data science agent workflows.
    
    Creates a structured results folder for each workflow execution:
    
    results/
        workflow_{id}_{timestamp}/
            00_orchestrator/
                workflow_plan.json
                execution_log.txt
            01_data_acquisition/
                console_output.txt
                data_loading_code.py
                loaded_data_info.json
                data/
                    loaded_dataset.csv
            02_eda/
                console_output.txt
                eda_code.py
                plots/
                    distribution_*.png
                    correlation_heatmap.png
                eda_summary.json
            03_modeling/
                console_output.txt
                model_code.py
                model_summary.txt
                plots/
                    residual_plots.png
                    diagnostic_plots.png
                fitted_model.pkl
            04_interpretation/
                console_output.txt
                final_report.md
                plots/
            manifest.json  # Complete record of all outputs
    """
    
    def __init__(
        self,
        base_results_dir: str = "results",
        workflow_id: Optional[str] = None
    ):
        """
        Initialize output manager.
        
        Args:
            base_results_dir: Base directory for all results
            workflow_id: Optional workflow ID; if None, creates new one
        """
        self.base_results_dir = Path(base_results_dir).resolve()  # Use absolute path
        self.base_results_dir.mkdir(exist_ok=True)
        
        # Create workflow-specific directory
        if workflow_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            workflow_id = f"workflow_{timestamp}"
        
        self.workflow_id = workflow_id
        self.workflow_dir = (self.base_results_dir / workflow_id).resolve()  # Absolute path
        self.workflow_dir.mkdir(exist_ok=True)
        
        # Track all outputs
        self.manifest: Dict[str, Any] = {
            "workflow_id": workflow_id,
            "created_at": datetime.now().isoformat(),
            "stages": {},
            "all_outputs": []
        }
        
        logger.info(f"📁 OutputManager initialized: {self.workflow_dir}")
    
    def get_stage_dir(self, stage_name: str, stage_order: Optional[int] = None) -> Path:
        """
        Get (or create) directory for a specific stage.
        
        Args:
            stage_name: Name of the stage (e.g., "data_acquisition")
            stage_order: Optional order number for sorting (e.g., 1, 2, 3)
            
        Returns:
            Path to stage directory (absolute path)
        """
        if stage_order is not None:
            dir_name = f"{stage_order:02d}_{stage_name}"
        else:
            dir_name = stage_name
        
        stage_dir = (self.workflow_dir / dir_name).resolve()  # Use absolute path
        stage_dir.mkdir(exist_ok=True)
        
        # Create standard subdirectories
        (stage_dir / "plots").mkdir(exist_ok=True)
        (stage_dir / "data").mkdir(exist_ok=True)
        (stage_dir / "models").mkdir(exist_ok=True)
        (stage_dir / 'inputs').mkdir(exist_ok=True)
        
        return stage_dir
    
    def save_code(
        self,
        stage_name: str,
        code: str,
        filename: str = "code.py",
        stage_order: Optional[int] = None
    ) -> Path:
        """
        Save executed code to stage directory.
        
        Args:
            stage_name: Name of the stage
            code: Python code that was executed
            filename: Name for the code file
            stage_order: Optional order number
            
        Returns:
            Path to saved code file
        """
        stage_dir = self.get_stage_dir(stage_name, stage_order)
        code_file = stage_dir / filename
        
        with open(code_file, 'w') as f:
            f.write(code)
        
        logger.info(f"💾 Saved code: {code_file}")
        self._add_to_manifest(stage_name, "code", str(code_file))
        
        return code_file
    
    def save_console_output(
        self,
        stage_name: str,
        stdout: str,
        stderr: str,
        stage_order: Optional[int] = None
    ) -> Path:
        """
        Save console output (stdout and stderr) to stage directory.
        
        Args:
            stage_name: Name of the stage
            stdout: Standard output text
            stderr: Standard error text
            stage_order: Optional order number
            
        Returns:
            Path to saved console output file
        """
        stage_dir = self.get_stage_dir(stage_name, stage_order)
        console_file = stage_dir / f"console_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(console_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("STANDARD OUTPUT\n")
            f.write("=" * 80 + "\n")
            f.write(stdout)
            
            if stderr:
                f.write("\n\n")
                f.write("=" * 80 + "\n")
                f.write("STANDARD ERROR\n")
                f.write("=" * 80 + "\n")
                f.write(stderr)
        
        logger.info(f"📄 Saved console output: {console_file}")
        self._add_to_manifest(stage_name, "console_output", str(console_file))
        
        return console_file
    
    def save_json(
        self,
        stage_name: str,
        data: Dict[str, Any],
        filename: str,
        stage_order: Optional[int] = None
    ) -> Path:
        """
        Save JSON data to stage directory.
        
        Args:
            stage_name: Name of the stage
            data: Dictionary to save as JSON
            filename: Name for the JSON file
            stage_order: Optional order number
            
        Returns:
            Path to saved JSON file
        """
        stage_dir = self.get_stage_dir(stage_name, stage_order)
        json_file = stage_dir / filename
        
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"📊 Saved JSON: {json_file}")
        self._add_to_manifest(stage_name, "json_data", str(json_file))
        
        return json_file
    
    def save_text(
        self,
        stage_name: str,
        text: str,
        filename: str,
        stage_order: Optional[int] = None
    ) -> Path:
        """
        Save text output to stage directory.
        
        Args:
            stage_name: Name of the stage
            text: Text content to save
            filename: Name for the text file
            stage_order: Optional order number
            
        Returns:
            Path to saved text file
        """
        stage_dir = self.get_stage_dir(stage_name, stage_order)
        text_file = stage_dir / filename
        
        with open(text_file, 'w') as f:
            f.write(text)
        
        logger.info(f"📝 Saved text: {text_file}")
        self._add_to_manifest(stage_name, "text_output", str(text_file))
        
        return text_file
    
    def stage_input_files(
            self,
            stage_name: str,
            input_files: List[Path],
            stage_order: Optional[int] = None
    ) -> List[Path]:
        """
        Stage input files for a specific stage by copying them to the stage directory.
        
        Args:
            stage_name: Name of the stage
            input_files: List of file paths to stage
            stage_order: Optional order number
        
        Returns:
            List of paths to staged files
        """
        stage_dir = self.get_stage_dir(stage_name, stage_order)
        staged_files = []
        
        for input_file in input_files:
            if not input_file.is_file():
                logger.warning(f"⚠️ Input file does not exist: {input_file}")
                continue
            
            dest_file = stage_dir / input_file.name
            shutil.copy2(input_file, dest_file)
            staged_files.append(dest_file)
            
            logger.info(f"📥 Staged input file: {dest_file}")
            self._add_to_manifest(stage_name, "staged_input", str(dest_file))
        
        return staged_files
    
    def copy_generated_files(
        self,
        stage_name: str,
        source_dir: Path,
        stage_order: Optional[int] = None,
        file_patterns: Optional[List[str]] = None
    ) -> List[Path]:
        """
        Copy generated files (plots, data files, models) to stage directory.
        
        Args:
            stage_name: Name of the stage
            source_dir: Directory where files were generated
            stage_order: Optional order number
            file_patterns: Optional list of glob patterns to match
                          (e.g., ['*.png', '*.csv', '*.pkl'])
                          If None, copies all files
            
        Returns:
            List of paths to copied files
        """
        stage_dir = self.get_stage_dir(stage_name, stage_order)
        copied_files = []
        
        if file_patterns is None:
            file_patterns = ['*']
        
        for pattern in file_patterns:
            for source_file in source_dir.glob(pattern):
                if not source_file.is_file():
                    continue
                
                # Determine destination subdirectory based on file type
                suffix = source_file.suffix.lower()
                if suffix in ['.png', '.jpg', '.jpeg', '.svg', '.pdf']:
                    dest_subdir = stage_dir / "plots"
                elif suffix in ['.csv', '.xlsx', '.parquet', '.json']:
                    dest_subdir = stage_dir / "data"
                elif suffix in ['.pkl', '.joblib', '.h5', '.pt', '.pth']:
                    dest_subdir = stage_dir / "models"
                else:
                    dest_subdir = stage_dir
                
                dest_file = dest_subdir / source_file.name
                if not suffix in ['.csv', '.xlsx', '.parquet', '.json']:
                    shutil.move(source_file, dest_file)
                else:
                    shutil.copy2(source_file, dest_file)
                copied_files.append(dest_file)
                
                logger.info(f"📦 Copied file: {dest_file}")
                self._add_to_manifest(stage_name, f"file_{suffix}", str(dest_file))
        
        return copied_files
    
    def save_dataframe(
        self,
        stage_name: str,
        df: Any,  # pandas DataFrame
        filename: str = "data.csv",
        stage_order: Optional[int] = None,
        save_format: str = "csv"
    ) -> Path:
        """
        Save a pandas DataFrame to stage directory.
        
        Args:
            stage_name: Name of the stage
            df: pandas DataFrame to save
            filename: Name for the data file
            stage_order: Optional order number
            save_format: Format to save ('csv', 'parquet', 'excel')
            
        Returns:
            Path to saved data file
        """
        stage_dir = self.get_stage_dir(stage_name, stage_order)
        data_dir = stage_dir / "data"
        
        if save_format == "csv":
            data_file = data_dir / filename
            df.to_csv(data_file, index=False)
        elif save_format == "parquet":
            data_file = data_dir / filename.replace('.csv', '.parquet')
            df.to_parquet(data_file, index=False)
        elif save_format == "excel":
            data_file = data_dir / filename.replace('.csv', '.xlsx')
            df.to_excel(data_file, index=False)
        else:
            raise ValueError(f"Unsupported format: {save_format}")
        
        logger.info(f"💾 Saved DataFrame: {data_file} ({len(df)} rows)")
        self._add_to_manifest(stage_name, "dataframe", str(data_file))
        
        # Also save summary statistics
        summary_file = data_dir / f"{filename}_summary.json"
        summary = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": df.isnull().sum().to_dict()
        }
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return data_file
    
    def _add_to_manifest(self, stage_name: str, output_type: str, file_path: str):
        """Add output to manifest tracking"""
        if stage_name not in self.manifest["stages"]:
            self.manifest["stages"][stage_name] = {
                "outputs": [],
                "created_at": datetime.now().isoformat()
            }
        
        output_entry = {
            "type": output_type,
            "path": file_path,
            "timestamp": datetime.now().isoformat()
        }
        
        self.manifest["stages"][stage_name]["outputs"].append(output_entry)
        self.manifest["all_outputs"].append(output_entry)
    
    def save_manifest(self) -> Path:
        """
        Save the complete manifest of all outputs.
        
        Returns:
            Path to manifest file
        """
        manifest_file = self.workflow_dir / "manifest.json"
        
        self.manifest["completed_at"] = datetime.now().isoformat()
        self.manifest["total_stages"] = len(self.manifest["stages"])
        self.manifest["total_outputs"] = len(self.manifest["all_outputs"])
        
        with open(manifest_file, 'w') as f:
            json.dump(self.manifest, f, indent=2, default=str)
        
        logger.info(f"✅ Saved manifest: {manifest_file}")
        logger.info(f"   Total stages: {self.manifest['total_stages']}")
        logger.info(f"   Total outputs: {self.manifest['total_outputs']}")
        
        return manifest_file
    
    def get_stage_outputs(self, stage_name: str) -> List[Dict[str, Any]]:
        """
        Get all outputs for a specific stage.
        
        Args:
            stage_name: Name of the stage
            
        Returns:
            List of output entries for the stage
        """
        if stage_name in self.manifest["stages"]:
            return self.manifest["stages"][stage_name]["outputs"]
        return []
    
    def get_plots(self, stage_name: str) -> List[str]:
        """
        Get all plot file paths for a specific stage.
        
        Args:
            stage_name: Name of the stage
        Returns:
            List of plot file paths
        """
        outputs = self.get_stage_outputs(stage_name)
        plot_files = [
            output["path"]
            for output in outputs
            if output["type"] in ["file_.png", "file_.jpg", "file_.jpeg", "file_.svg", "file_.pdf"]
        ]
        return plot_files
    
    def create_summary_report(self) -> Path:
        """
        Create a human-readable summary report of all outputs.
        
        Returns:
            Path to summary report
        """
        report_file = self.workflow_dir / "SUMMARY.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# Workflow Summary: {self.workflow_id}\n\n")
            f.write(f"**Created:** {self.manifest['created_at']}\n\n")
            f.write(f"**Total Stages:** {len(self.manifest['stages'])}\n")
            f.write(f"**Total Outputs:** {len(self.manifest['all_outputs'])}\n\n")
            
            f.write("## Stages\n\n")
            for stage_name, stage_data in self.manifest["stages"].items():
                f.write(f"### {stage_name}\n\n")
                f.write(f"- Created: {stage_data['created_at']}\n")
                f.write(f"- Outputs: {len(stage_data['outputs'])}\n\n")
                
                for output in stage_data["outputs"]:
                    output_path = Path(output['path'])
                    relative_path = output_path.relative_to(self.workflow_dir)
                    f.write(f"  - `{relative_path}` ({output['type']})\n")
                
                f.write("\n")
        
        logger.info(f"📋 Created summary report: {report_file}")
        return report_file
    
    def get_stage_summary(self, stage_name: str) -> Optional[str]:
        """
        Get a brief summary of outputs for a specific stage.
        
        Args:
            stage_name: Name of the stage"""
        summary_path = self.workflow_dir / "SUMMARY.md"

        
        self.create_summary_report()

        with open(summary_path, 'r') as f:
            lines = f.readlines()
        
        in_stage_section = False
        summary_lines = []
        for line in lines:
            if line.startswith(f"### {stage_name}"):
                in_stage_section = True
                summary_lines.append(line)
            elif in_stage_section:
                if line.startswith("### "):  # Next stage
                    break
                summary_lines.append(line)
        
        return ''.join(summary_lines).strip()
    
    @contextmanager
    def capture_execution(
        self,
        stage_name: str,
        stage_order: Optional[int] = None
    ):
        """
        Context manager to capture all outputs during code execution.
        
        Usage:
            with output_manager.capture_execution("modeling", 3):
                # Execute code here
                # All stdout/plots/files will be captured
                pass
        
        Args:
            stage_name: Name of the stage
            stage_order: Optional order number
        """
        stage_dir = self.get_stage_dir(stage_name, stage_order)
        
        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            yield stage_dir
            
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # Save captured output
            stdout_text = stdout_capture.getvalue()
            stderr_text = stderr_capture.getvalue()
            
            if stdout_text or stderr_text:
                self.save_console_output(
                    stage_name,
                    stdout_text,
                    stderr_text,
                    stage_order
                )


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_usage():
    """Example of how to use OutputManager"""
    
    # Initialize for a workflow
    output_mgr = OutputManager(workflow_id="example_workflow_001")
    
    # Stage 1: Data Acquisition
    output_mgr.save_code(
        "data_acquisition",
        "import pandas as pd\ndf = pd.read_csv('data.csv')",
        stage_order=1
    )
    output_mgr.save_console_output(
        "data_acquisition",
        "Successfully loaded 1000 rows",
        "",
        stage_order=1
    )
    
    # Stage 2: EDA with captured execution
    with output_mgr.capture_execution("eda", stage_order=2):
        print("Running EDA...")
        print("Generated correlation matrix")
    
    # Save manifest and summary
    output_mgr.save_manifest()
    output_mgr.create_summary_report()
    
    print(f"Results saved to: {output_mgr.workflow_dir}")

if __name__ == "__main__":
    example_usage()
