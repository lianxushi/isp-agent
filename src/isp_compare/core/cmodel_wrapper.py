"""
CModel ISP Wrapper
================

Wrapper for C/C++ CModel ISP executable.
Features:
- Subprocess-based execution
- Multi-threaded processing
- Batch processing support
- Configurable parameters

Author: ISP Team
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CModelResult:
    """Result from CModel processing"""
    success: bool
    output_path: str = ""
    time_ms: float = 0
    error: str = ""
    raw_path: str = ""


class CModelError(Exception):
    """Exception raised when CModel execution fails"""
    pass


class CModelISP:
    """
    CModel ISP wrapper.
    
    Wraps the C/C++ CModel ISP executable with Python interface.
    
    Features:
    - Automatic CPU thread detection
    - Multi-threaded parallel processing
    - Configurable ISP parameters
    - Batch processing support
    
    Example:
        >>> cmodel = CModelISP("/path/to/cmodel", num_threads=8)
        >>> result = cmodel.process("input.raw", "output.jpg", params={"gain": 1.2})
        >>> print(result.success)
    """
    
    def __init__(
        self,
        cmodel_path: str,
        num_threads: Optional[int] = None,
        default_params: Optional[Dict[str, Any]] = None,
        timeout: int = 60
    ):
        """
        Initialize CModel ISP wrapper.
        
        Args:
            cmodel_path: Path to CModel executable
            num_threads: Number of CPU threads (default: CPU count)
            default_params: Default ISP parameters
            timeout: Command timeout in seconds
        """
        self.cmodel_path = Path(cmodel_path)
        
        if not self.cmodel_path.exists():
            raise CModelError(f"CModel executable not found: {self.cmodel_path}")
        
        if not os.access(self.cmodel_path, os.X_OK):
            raise CModelError(f"CModel not executable: {self.cmodel_path}")
        
        self.num_threads = num_threads or os.cpu_count() or 4
        self.default_params = default_params or {}
        self.timeout = timeout
        
        logger.info(f"CModelISP initialized: {self.cmodel_path}, threads={self.num_threads}")
    
    def get_version(self) -> str:
        """
        Get CModel version.
        
        Returns:
            str: Version string
        """
        try:
            result = subprocess.run(
                [str(self.cmodel_path), "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return "unknown"
        except Exception as e:
            logger.warning(f"Failed to get CModel version: {e}")
            return "unknown"
    
    def process(
        self,
        raw_path: str,
        output_path: str,
        params: Optional[Dict[str, Any]] = None
    ) -> CModelResult:
        """
        Process single RAW file through CModel ISP.
        
        Args:
            raw_path: Input RAW file path
            output_path: Output image path (RGB 24-bit)
            params: ISP parameters
            
        Returns:
            CModelResult: Processing result
        """
        start_time = time.time()
        raw_path = Path(raw_path)
        output_path = Path(output_path)
        
        if not raw_path.exists():
            return CModelResult(
                success=False,
                error=f"Input file not found: {raw_path}",
                raw_path=str(raw_path)
            )
        
        # Merge default params with provided params
        all_params = {**self.default_params, **(params or {})}
        
        # Build command
        cmd = [
            str(self.cmodel_path),
            "-i", str(raw_path),
            "-o", str(output_path),
            "-threads", str(self.num_threads)
        ]
        
        # Add ISP parameters
        for key, value in all_params.items():
            if value is not None:
                cmd.extend([f"--{key}", str(value)])
        
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Execute
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            if result.returncode != 0:
                logger.error(f"CModel failed: {result.stderr}")
                return CModelResult(
                    success=False,
                    error=result.stderr or "Unknown error",
                    time_ms=elapsed_ms,
                    raw_path=str(raw_path)
                )
            
            logger.debug(f"CModel processed: {raw_path.name} -> {output_path.name} ({elapsed_ms:.1f}ms)")
            
            return CModelResult(
                success=True,
                output_path=str(output_path),
                time_ms=elapsed_ms,
                raw_path=str(raw_path)
            )
            
        except subprocess.TimeoutExpired:
            return CModelResult(
                success=False,
                error=f"Timeout after {self.timeout}s",
                time_ms=self.timeout * 1000,
                raw_path=str(raw_path)
            )
        except Exception as e:
            return CModelResult(
                success=False,
                error=str(e),
                time_ms=(time.time() - start_time) * 1000,
                raw_path=str(raw_path)
            )
    
    def batch_process(
        self,
        raw_files: List[str],
        output_dir: str,
        params: Optional[Dict[str, Any]] = None,
        max_workers: Optional[int] = None
    ) -> List[CModelResult]:
        """
        Batch process multiple RAW files with multi-threading.
        
        Args:
            raw_files: List of input RAW file paths
            output_dir: Output directory
            params: ISP parameters
            max_workers: Max parallel threads (default: self.num_threads)
            
        Returns:
            List[CModelResult]: Processing results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not raw_files:
            logger.warning("No files to process")
            return []
        
        max_workers = max_workers or min(self.num_threads, len(raw_files))
        
        logger.info(f"Batch processing {len(raw_files)} files with {max_workers} workers")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {}
            for i, raw_path in enumerate(raw_files):
                output_path = output_dir / f"output_{i:04d}.jpg"
                future = executor.submit(self.process, raw_path, str(output_path), params)
                future_to_idx[future] = i
            
            # Collect results
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch task {idx} failed: {e}")
                    results.append(CModelResult(
                        success=False,
                        error=str(e),
                        raw_path=raw_files[idx]
                    ))
        
        # Sort by original order
        results.sort(key=lambda r: raw_files.index(r.raw_path) if r.raw_path in raw_files else -1)
        
        success_count = sum(1 for r in results if r.success)
        logger.info(f"Batch complete: {success_count}/{len(results)} successful")
        
        return results
    
    def process_with_fallback(
        self,
        raw_path: str,
        output_path: str,
        params: Optional[Dict[str, Any]] = None,
        fallback_params: Optional[Dict[str, Any]] = None
    ) -> CModelResult:
        """
        Process with fallback parameters if first attempt fails.
        
        Args:
            raw_path: Input RAW file path
            output_path: Output image path
            params: Primary ISP parameters
            fallback_params: Fallback parameters if primary fails
            
        Returns:
            CModelResult: Processing result
        """
        # Try primary params first
        result = self.process(raw_path, output_path, params)
        
        if result.success:
            return result
        
        # Try fallback if provided
        if fallback_params:
            logger.info(f"Primary params failed, trying fallback for {Path(raw_path).name}")
            return self.process(raw_path, output_path, fallback_params)
        
        return result
