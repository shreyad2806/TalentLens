"""
Resume Ingestor module - Handles discovery and ingestion of resume files.

This module provides the ResumeIngestor class for discovering resume files
from various sources (directories, file lists, etc.) and preparing them for
indexing. It supports filtering by file type and provides batch processing
capabilities.
"""

import logging
from typing import List, Union, Optional, Callable
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """
    Result of a resume ingestion operation.
    
    Attributes:
        total_files: Total number of files discovered
        valid_files: Number of valid resume files
        invalid_files: Number of invalid files
        skipped_files: Number of skipped files
        file_paths: List of valid file paths
        errors: List of error messages
    """
    total_files: int
    valid_files: int
    invalid_files: int
    skipped_files: int
    file_paths: List[Path]
    errors: List[str]


class ResumeIngestor:
    """
    Ingestor for discovering and preparing resume files for indexing.
    
    This class provides methods to discover resume files from various sources,
    filter them by type, and prepare them for the indexing pipeline. It supports
    common resume formats: PDF, DOCX, and TXT.
    """
    
    # Supported resume file extensions
    SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt'}
    
    def __init__(self, supported_extensions: Optional[set] = None):
        """
        Initialize the resume ingestor.
        
        Args:
            supported_extensions: Set of supported file extensions.
                                If None, uses default {'.pdf', '.docx', '.doc', '.txt'}
        """
        self.supported_extensions = supported_extensions or self.SUPPORTED_EXTENSIONS
        logger.info(f"ResumeIngestor initialized with extensions: {self.supported_extensions}")
    
    def ingest_from_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        filter_fn: Optional[Callable[[Path], bool]] = None
    ) -> IngestionResult:
        """
        Discover and ingest resume files from a directory.
        
        Args:
            directory: Path to the directory to search
            recursive: Whether to search recursively (default: True)
            filter_fn: Optional function to filter files. Takes Path and returns bool.
            
        Returns:
            IngestionResult with discovered files and statistics
        """
        directory = Path(directory)
        
        if not directory.exists():
            error_msg = f"Directory does not exist: {directory}"
            logger.error(error_msg)
            return IngestionResult(
                total_files=0,
                valid_files=0,
                invalid_files=0,
                skipped_files=0,
                file_paths=[],
                errors=[error_msg]
            )
        
        if not directory.is_dir():
            error_msg = f"Path is not a directory: {directory}"
            logger.error(error_msg)
            return IngestionResult(
                total_files=0,
                valid_files=0,
                invalid_files=0,
                skipped_files=0,
                file_paths=[],
                errors=[error_msg]
            )
        
        logger.info(f"Ingesting resumes from directory: {directory} (recursive={recursive})")
        
        valid_files = []
        invalid_files = []
        skipped_files = []
        errors = []
        
        # Get all files
        if recursive:
            files = directory.rglob('*')
        else:
            files = directory.glob('*')
        
        for file_path in files:
            if not file_path.is_file():
                continue
            
            # Check extension
            if file_path.suffix.lower() not in self.supported_extensions:
                skipped_files.append(file_path)
                continue
            
            # Apply custom filter if provided
            if filter_fn and not filter_fn(file_path):
                skipped_files.append(file_path)
                continue
            
            # Check if file is readable
            try:
                # Try to read file size to verify it's accessible
                file_path.stat()
                valid_files.append(file_path)
            except Exception as e:
                error_msg = f"Cannot read file {file_path}: {str(e)}"
                errors.append(error_msg)
                invalid_files.append(file_path)
                logger.warning(error_msg)
        
        total_files = len(valid_files) + len(invalid_files) + len(skipped_files)
        
        result = IngestionResult(
            total_files=total_files,
            valid_files=len(valid_files),
            invalid_files=len(invalid_files),
            skipped_files=len(skipped_files),
            file_paths=valid_files,
            errors=errors
        )
        
        logger.info(
            f"Ingestion complete: {len(valid_files)} valid, "
            f"{len(invalid_files)} invalid, {len(skipped_files)} skipped"
        )
        
        return result
    
    def ingest_from_list(
        self,
        file_paths: List[Union[str, Path]],
        filter_fn: Optional[Callable[[Path], bool]] = None
    ) -> IngestionResult:
        """
        Ingest resume files from a list of file paths.
        
        Args:
            file_paths: List of file paths to ingest
            filter_fn: Optional function to filter files. Takes Path and returns bool.
            
        Returns:
            IngestionResult with validated files and statistics
        """
        logger.info(f"Ingesting resumes from list of {len(file_paths)} files")
        
        valid_files = []
        invalid_files = []
        skipped_files = []
        errors = []
        
        for file_path in file_paths:
            file_path = Path(file_path)
            
            # Check if file exists
            if not file_path.exists():
                error_msg = f"File does not exist: {file_path}"
                errors.append(error_msg)
                invalid_files.append(file_path)
                logger.warning(error_msg)
                continue
            
            # Check if it's a file
            if not file_path.is_file():
                error_msg = f"Path is not a file: {file_path}"
                errors.append(error_msg)
                invalid_files.append(file_path)
                logger.warning(error_msg)
                continue
            
            # Check extension
            if file_path.suffix.lower() not in self.supported_extensions:
                skipped_files.append(file_path)
                continue
            
            # Apply custom filter if provided
            if filter_fn and not filter_fn(file_path):
                skipped_files.append(file_path)
                continue
            
            # Check if file is readable
            try:
                file_path.stat()
                valid_files.append(file_path)
            except Exception as e:
                error_msg = f"Cannot read file {file_path}: {str(e)}"
                errors.append(error_msg)
                invalid_files.append(file_path)
                logger.warning(error_msg)
        
        total_files = len(valid_files) + len(invalid_files) + len(skipped_files)
        
        result = IngestionResult(
            total_files=total_files,
            valid_files=len(valid_files),
            invalid_files=len(invalid_files),
            skipped_files=len(skipped_files),
            file_paths=valid_files,
            errors=errors
        )
        
        logger.info(
            f"Ingestion complete: {len(valid_files)} valid, "
            f"{len(invalid_files)} invalid, {len(skipped_files)} skipped"
        )
        
        return result
    
    def ingest_single_file(self, file_path: Union[str, Path]) -> IngestionResult:
        """
        Ingest a single resume file.
        
        Args:
            file_path: Path to the resume file
            
        Returns:
            IngestionResult with validation result
        """
        return self.ingest_from_list([file_path])
    
    def filter_by_extension(
        self,
        file_paths: List[Union[str, Path]],
        extensions: set
    ) -> List[Path]:
        """
        Filter file paths by extension.
        
        Args:
            file_paths: List of file paths to filter
            extensions: Set of extensions to keep (e.g., {'.pdf', '.docx'})
            
        Returns:
            List of file paths with matching extensions
        """
        filtered = []
        for file_path in file_paths:
            file_path = Path(file_path)
            if file_path.suffix.lower() in extensions:
                filtered.append(file_path)
        
        logger.info(f"Filtered {len(file_paths)} files to {len(filtered)} by extension")
        return filtered
    
    def filter_by_size(
        self,
        file_paths: List[Union[str, Path]],
        min_size_bytes: int = 0,
        max_size_bytes: Optional[int] = None
    ) -> List[Path]:
        """
        Filter file paths by size.
        
        Args:
            file_paths: List of file paths to filter
            min_size_bytes: Minimum file size in bytes (default: 0)
            max_size_bytes: Maximum file size in bytes (default: None for no limit)
            
        Returns:
            List of file paths within size range
        """
        filtered = []
        for file_path in file_paths:
            file_path = Path(file_path)
            try:
                size = file_path.stat().st_size
                if size >= min_size_bytes:
                    if max_size_bytes is None or size <= max_size_bytes:
                        filtered.append(file_path)
            except Exception as e:
                logger.warning(f"Cannot get size for {file_path}: {e}")
        
        logger.info(f"Filtered {len(file_paths)} files to {len(filtered)} by size")
        return filtered
