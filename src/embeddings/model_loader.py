"""
Model Loader module - Singleton model loading with lazy initialization.

This module provides a singleton pattern for loading the BAAI/bge-m3 embedding model.
The model is loaded only once and reused across the application to avoid memory overhead
and loading time.

The model is loaded lazily - only when first requested, not at module import time.
"""

from typing import Optional
from threading import Lock


class ModelLoader:
    """
    Singleton model loader for BAAI/bge-m3 embedding model.
    
    This class implements the singleton pattern to ensure that the embedding model
    is loaded only once and reused across the application. The model is loaded
    lazily on first access to avoid unnecessary memory usage during import.
    
    The model loader is thread-safe and can be used in multi-threaded environments.
    """
    
    _instance: Optional['ModelLoader'] = None
    _lock: Lock = Lock()
    _model = None
    _model_name: str = "BAAI/bge-m3"
    _is_loaded: bool = False
    
    def __new__(cls) -> 'ModelLoader':
        """
        Create or return the singleton instance.
        
        Returns:
            The singleton ModelLoader instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """
        Initialize the model loader (no-op if already initialized).
        
        The actual model loading happens on first access via get_model().
        """
        pass
    
    def load_model(self) -> None:
        """
        Load the BAAI/bge-m3 embedding model.
        
        This method loads the model from Hugging Face and caches it for future use.
        The model is loaded only once, subsequent calls return the cached model.
        
        Raises:
            Exception: If model loading fails
        """
        if self._is_loaded:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            self._is_loaded = True
        except Exception as e:
            raise Exception(f"Failed to load model {self._model_name}: {str(e)}")
    
    def get_model(self):
        """
        Get the loaded embedding model.
        
        This method loads the model if not already loaded and returns it.
        
        Returns:
            The loaded SentenceTransformer model
        """
        if not self._is_loaded:
            self.load_model()
        return self._model
    
    def get_model_name(self) -> str:
        """
        Get the name of the model being used.
        
        Returns:
            The model name
        """
        return self._model_name
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            The embedding dimension (1024 for BAAI/bge-m3)
        """
        return 1024
    
    def is_loaded(self) -> bool:
        """
        Check if the model is loaded.
        
        Returns:
            True if the model is loaded, False otherwise
        """
        return self._is_loaded
    
    def unload_model(self) -> None:
        """
        Unload the model from memory.
        
        This method removes the model from memory and resets the loaded state.
        Use this to free memory when the model is no longer needed.
        """
        self._model = None
        self._is_loaded = False


# Global singleton instance
_model_loader: Optional[ModelLoader] = None
_model_loader_lock: Lock = Lock()


def get_model_loader() -> ModelLoader:
    """
    Get the singleton ModelLoader instance.
    
    This function provides thread-safe access to the singleton ModelLoader instance.
    
    Returns:
        The singleton ModelLoader instance
    """
    global _model_loader
    if _model_loader is None:
        with _model_loader_lock:
            if _model_loader is None:
                _model_loader = ModelLoader()
    return _model_loader
