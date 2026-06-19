"""
Text Extractor module - Extracts text from various document formats.

This module provides a unified interface for extracting text from PDF, DOCX,
and TXT files. It handles file I/O, format detection, and error handling.
"""

import io
from typing import Optional, Union
from pathlib import Path

# PDF extraction
from PyPDF2 import PdfReader

# DOCX extraction
import docx

# File handling
from typing import BinaryIO


class TextExtractor:
    """
    Unified text extractor for multiple document formats.
    
    This class provides methods to extract text from PDF, DOCX, and TXT files.
    It handles file I/O, format detection, and provides error handling.
    
    Supported formats:
        - PDF (.pdf)
        - DOCX (.docx)
        - TXT (.txt)
    """
    
    @staticmethod
    def extract_from_file(file_path: Union[str, Path]) -> str:
        """
        Extract text from a file given its path.
        
        Args:
            file_path: Path to the file (string or Path object)
            
        Returns:
            Extracted text as a string
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported
            Exception: For other extraction errors
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type from extension
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.pdf':
            return TextExtractor.extract_from_pdf(file_path)
        elif file_extension == '.docx':
            return TextExtractor.extract_from_docx(file_path)
        elif file_extension == '.txt':
            return TextExtractor.extract_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    @staticmethod
    def extract_from_file_object(file_object: BinaryIO) -> str:
        """
        Extract text from a file object (e.g., from Streamlit upload).
        
        Args:
            file_object: File-like object (must have read() and name attributes)
            
        Returns:
            Extracted text as a string
            
        Raises:
            ValueError: If the file format is not supported
            Exception: For extraction errors
        """
        # Reset file pointer to beginning
        file_object.seek(0)
        
        # Get filename from file object
        filename = getattr(file_object, 'name', '')
        if not filename:
            raise ValueError("File object must have a 'name' attribute")
        
        # Determine file type from extension
        file_extension = Path(filename).suffix.lower()
        
        if file_extension == '.pdf':
            return TextExtractor._extract_pdf_from_object(file_object)
        elif file_extension == '.docx':
            return TextExtractor._extract_docx_from_object(file_object)
        elif file_extension == '.txt':
            return TextExtractor._extract_txt_from_object(file_object)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    @staticmethod
    def extract_from_pdf(file_path: Union[str, Path]) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
            
        Raises:
            Exception: If PDF extraction fails
        """
        try:
            with open(file_path, 'rb') as file:
                result = TextExtractor._extract_pdf_from_object(file)
                return result
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    @staticmethod
    def _extract_pdf_from_object(file_object: BinaryIO) -> str:
        """
        Internal method to extract text from a PDF file object.
        
        Args:
            file_object: File-like object containing PDF data
            
        Returns:
            Extracted text as a string
        """
        try:
            # Reset file pointer
            file_object.seek(0)
            
            # Read PDF bytes
            pdf_bytes = file_object.read()
            pdf_stream = io.BytesIO(pdf_bytes)
            
            # Create PDF reader
            reader = PdfReader(pdf_stream)
            
            # Extract text from all pages
            text = ""
            for i, page in enumerate(reader.pages):
                content = page.extract_text()
                if content:
                    text += content + "\n"
            
            return text.strip()
            
        except Exception as e:
            raise Exception(f"PDF extraction failed: {str(e)}")
    
    @staticmethod
    def extract_from_docx(file_path: Union[str, Path]) -> str:
        """
        Extract text from a DOCX file.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Extracted text as a string
            
        Raises:
            Exception: If DOCX extraction fails
        """
        try:
            with open(file_path, 'rb') as file:
                return TextExtractor._extract_docx_from_object(file)
        except Exception as e:
            raise Exception(f"Failed to extract text from DOCX: {str(e)}")
    
    @staticmethod
    def _extract_docx_from_object(file_object: BinaryIO) -> str:
        """
        Internal method to extract text from a DOCX file object.
        
        Args:
            file_object: File-like object containing DOCX data
            
        Returns:
            Extracted text as a string
        """
        try:
            # Reset file pointer
            file_object.seek(0)
            
            # Load DOCX document
            doc = docx.Document(file_object)
            
            # Extract text from all paragraphs
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            return text.strip()
            
        except Exception as e:
            raise Exception(f"DOCX extraction failed: {str(e)}")
    
    @staticmethod
    def extract_from_txt(file_path: Union[str, Path]) -> str:
        """
        Extract text from a TXT file.
        
        Args:
            file_path: Path to the TXT file
            
        Returns:
            Extracted text as a string
            
        Raises:
            Exception: If TXT extraction fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read().strip()
            except Exception as e:
                raise Exception(f"TXT extraction failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to extract text from TXT: {str(e)}")
    
    @staticmethod
    def _extract_txt_from_object(file_object: BinaryIO) -> str:
        """
        Internal method to extract text from a TXT file object.
        
        Args:
            file_object: File-like object containing TXT data
            
        Returns:
            Extracted text as a string
        """
        try:
            # Reset file pointer
            file_object.seek(0)
            
            # Read text with UTF-8 encoding
            text = file_object.read().decode('utf-8')
            
            return text.strip()
            
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                file_object.seek(0)
                text = file_object.read().decode('latin-1')
                return text.strip()
            except Exception as e:
                raise Exception(f"TXT extraction failed: {str(e)}")
        except Exception as e:
            raise Exception(f"TXT extraction failed: {str(e)}")
    
    @staticmethod
    def extract_from_text(text: str) -> str:
        """
        Return text as-is (for already extracted text).
        
        This method is provided for consistency when the text is already
        extracted (e.g., from a database or API).
        
        Args:
            text: Already extracted text
            
        Returns:
            The same text, stripped of leading/trailing whitespace
        """
        return text.strip() if text else ""
