"""
Resume Loader Module for Bootstrap System.

This module handles loading resume files from configurable directories
for the bootstrap system. It supports multiple resume source directories
and filters by supported file extensions. It also supports CSV ingestion
from Resume.csv files.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LoadResult:
    """Result of resume loading operation."""
    total_files_found: int
    valid_files: int
    invalid_files: int
    skipped_files: int
    file_paths: List[str]
    errors: List[str]
    load_time_seconds: float
    csv_detected: bool = False
    csv_path: Optional[str] = None


class ResumeLoader:
    """
    Loader for resume files from configurable directories.
    
    This class handles discovering and loading resume files from multiple
    source directories. It filters by supported file extensions and provides
    detailed loading results. It also detects CSV files for batch ingestion.
    
    Supported file extensions: .pdf, .docx, .doc, .txt
    CSV support: Resume.csv for batch ingestion
    """
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt'}
    
    DEFAULT_RESUME_PATHS = [
        'Resume',
        'resumes',
        'data/resumes',
    ]
    
    def __init__(self, resume_paths: Optional[List[str]] = None):
        """
        Initialize the resume loader.
        
        Args:
            resume_paths: List of directory paths to search for resumes.
                         If None, uses default paths.
        """
        self.resume_paths = resume_paths or self.DEFAULT_RESUME_PATHS
        logger.info(f"ResumeLoader initialized with paths: {self.resume_paths}")
    
    def detect_csv_files(self, base_path: Optional[str] = None) -> List[Path]:
        """
        Detect Resume.csv files in configured directories.
        
        Args:
            base_path: Base directory to resolve relative paths from.
                      If None, uses current working directory.
        
        Returns:
            List of Path objects pointing to detected CSV files
        """
        base = Path(base_path) if base_path else Path.cwd()
        csv_files: List[Path] = []
        
        for resume_path in self.resume_paths:
            full_path = base / resume_path
            
            if not full_path.exists() or not full_path.is_dir():
                continue
            
            # Check for Resume.csv
            csv_path = full_path / "Resume.csv"
            if csv_path.exists() and csv_path.is_file():
                csv_files.append(csv_path)
                logger.info(f"Detected CSV file: {csv_path}")
        
        return csv_files
    
    def discover_resumes(self, base_path: Optional[str] = None) -> List[Path]:
        """
        Discover resume files from configured directories.
        
        Args:
            base_path: Base directory to resolve relative paths from.
                      If None, uses current working directory.
        
        Returns:
            List of Path objects pointing to discovered resume files
        """
        import time
        start_time = time.time()
        
        base = Path(base_path) if base_path else Path.cwd()
        discovered_files: List[Path] = []
        
        for resume_path in self.resume_paths:
            full_path = base / resume_path
            
            if not full_path.exists():
                logger.debug(f"Resume path does not exist: {full_path}")
                continue
            
            if not full_path.is_dir():
                logger.debug(f"Resume path is not a directory: {full_path}")
                continue
            
            # Search for resume files
            for file_path in full_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    discovered_files.append(file_path)
        
        load_time = time.time() - start_time
        logger.info(f"Discovered {len(discovered_files)} resume files in {load_time:.2f}s")
        
        return discovered_files
    
    def load_resumes(self, base_path: Optional[str] = None) -> LoadResult:
        """
        Load resume files from configured directories.
        
        Args:
            base_path: Base directory to resolve relative paths from.
                      If None, uses current working directory.
        
        Returns:
            LoadResult with loading statistics and file paths
        """
        import time
        start_time = time.time()
        
        discovered_files = self.discover_resumes(base_path)
        
        # Detect CSV files
        csv_files = self.detect_csv_files(base_path)
        csv_detected = len(csv_files) > 0
        csv_path = str(csv_files[0]) if csv_files else None
        
        valid_files: List[str] = []
        invalid_files: List[str] = []
        skipped_files: List[str] = []
        errors: List[str] = []
        
        for file_path in discovered_files:
            try:
                # Check if file is accessible
                if not file_path.exists():
                    errors.append(f"File not found: {file_path}")
                    invalid_files.append(str(file_path))
                    continue
                
                if not file_path.is_file():
                    errors.append(f"Not a file: {file_path}")
                    invalid_files.append(str(file_path))
                    continue
                
                # Check file size (skip empty files)
                if file_path.stat().st_size == 0:
                    skipped_files.append(str(file_path))
                    logger.debug(f"Skipped empty file: {file_path}")
                    continue
                
                # Check file size (skip very large files > 50MB)
                if file_path.stat().st_size > 50 * 1024 * 1024:
                    skipped_files.append(str(file_path))
                    logger.debug(f"Skipped large file: {file_path}")
                    continue
                
                valid_files.append(str(file_path))
                
            except Exception as e:
                errors.append(f"Error loading {file_path}: {str(e)}")
                invalid_files.append(str(file_path))
                logger.error(f"Error loading {file_path}: {str(e)}")
        
        load_time = time.time() - start_time
        
        result = LoadResult(
            total_files_found=len(discovered_files),
            valid_files=len(valid_files),
            invalid_files=len(invalid_files),
            skipped_files=len(skipped_files),
            file_paths=valid_files,
            errors=errors,
            load_time_seconds=load_time,
            csv_detected=csv_detected,
            csv_path=csv_path
        )
        
        logger.info(
            f"Resume loading complete: "
            f"found={result.total_files_found}, "
            f"valid={result.valid_files}, "
            f"invalid={result.invalid_files}, "
            f"skipped={result.skipped_files}, "
            f"csv_detected={csv_detected}, "
            f"time={load_time:.2f}s"
        )
        
        return result
    
    def get_resume_count(self, base_path: Optional[str] = None) -> int:
        """
        Get the count of valid resume files.
        
        Args:
            base_path: Base directory to resolve relative paths from.
        
        Returns:
            Number of valid resume files
        """
        result = self.load_resumes(base_path)
        return result.valid_files
    
    def validate_resume_paths(self, base_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate that resume directories exist and are accessible.
        
        Args:
            base_path: Base directory to resolve relative paths from.
        
        Returns:
            Dictionary with validation results
        """
        base = Path(base_path) if base_path else Path.cwd()
        
        validation_results = {
            'valid_paths': [],
            'invalid_paths': [],
            'missing_paths': [],
            'total_paths': len(self.resume_paths)
        }
        
        for resume_path in self.resume_paths:
            full_path = base / resume_path
            
            if not full_path.exists():
                validation_results['missing_paths'].append(str(full_path))
                logger.warning(f"Resume path missing: {full_path}")
            elif not full_path.is_dir():
                validation_results['invalid_paths'].append(str(full_path))
                logger.warning(f"Resume path not a directory: {full_path}")
            else:
                validation_results['valid_paths'].append(str(full_path))
                logger.debug(f"Resume path valid: {full_path}")
        
        return validation_results
