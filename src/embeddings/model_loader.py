"""
Model Loader module - Singleton model loading with lazy initialization and local caching.

This module provides a singleton pattern for loading the BAAI/bge-m3 embedding model.
The model is loaded only once and reused across the application to avoid memory overhead
and loading time.

Features:
- Lazy loading: Model loads only when first requested
- Singleton pattern: Only one model instance exists
- Local caching: Checks local cache before downloading
- Configurable path: Uses BGE_MODEL_PATH env var, falls back to ./models/bge-m3
- Offline mode: If OFFLINE_MODE=true, never attempts internet download
- Retry mechanism: Retries download 3 times before failing
- Comprehensive logging: Logs all loading operations
"""

from typing import Optional
from threading import Lock
import os
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Singleton model loader for BAAI/bge-m3 embedding model.
    
    This class implements the singleton pattern to ensure that the embedding model
    is loaded only once and reused across the application. The model is loaded
    lazily on first access to avoid unnecessary memory usage during import.
    
    The model loader is thread-safe and can be used in multi-threaded environments.
    
    Features:
    - Checks local cache before downloading from HuggingFace
    - Supports offline mode (OFFLINE_MODE environment variable)
    - Configurable model path (BGE_MODEL_PATH environment variable)
    - Retry mechanism for failed downloads
    - Comprehensive logging
    """
    
    _instance: Optional['ModelLoader'] = None
    _lock: Lock = Lock()
    _model = None
    _model_name: str = "BAAI/bge-m3"
    _is_loaded: bool = False
    _offline_mode: bool = False
    _model_path: str = "./models/bge-m3"
    _max_retries: int = 3
    
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
        # Load configuration from environment variables
        self._offline_mode = os.getenv('OFFLINE_MODE', 'false').lower() == 'true'
        self._model_path = os.getenv('BGE_MODEL_PATH', './models/bge-m3')
        self._max_retries = int(os.getenv('MODEL_DOWNLOAD_RETRIES', '3'))
    
    def _check_local_cache(self) -> bool:
        """
        Check if the model exists in the local cache.
        
        Returns:
            True if model exists in local cache, False otherwise
        """
        model_path = Path(self._model_path)
        return model_path.exists() and model_path.is_dir()
    
    def _load_from_local(self) -> bool:
        """
        Attempt to load the model from local cache.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading cached model from {self._model_path}...")
            self._model = SentenceTransformer(self._model_path)
            self._is_loaded = True
            logger.info("Model loaded successfully from local cache")
            return True
        except Exception as e:
            logger.warning(f"Failed to load model from local cache: {str(e)}")
            return False
    
    def _download_model(self) -> bool:
        """
        Download the model from HuggingFace with retry mechanism.
        
        Returns:
            True if successful, False otherwise
        """
        if self._offline_mode:
            logger.info("Offline mode enabled. Skipping download.")
            return False
        
        for attempt in range(self._max_retries):
            try:
                logger.info(f"Downloading model {self._model_name} (attempt {attempt + 1}/{self._max_retries})...")
                from sentence_transformers import SentenceTransformer
                
                # Download to local cache path
                self._model = SentenceTransformer(
                    self._model_name,
                    cache_folder=self._model_path
                )
                self._is_loaded = True
                logger.info("Model downloaded and loaded successfully")
                return True
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {str(e)}")
                if attempt < self._max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error("All download attempts failed")
                    return False
        return False
    
    def load_model(self) -> None:
        """
        Load the BAAI/bge-m3 embedding model.
        
        This method attempts to load the model from local cache first.
        If not available and offline mode is disabled, it downloads from HuggingFace.
        
        Raises:
            Exception: If model loading fails and no local cache is available
        """
        if self._is_loaded:
            logger.info("Model already loaded")
            return
        
        # Try local cache first
        if self._check_local_cache():
            logger.info("Local cache found")
            if self._load_from_local():
                return
            else:
                logger.warning("Local cache exists but failed to load")
        
        # Try downloading if not in offline mode
        if not self._offline_mode:
            if self._download_model():
                return
        
        # If we get here, all attempts failed
        if self._offline_mode:
            raise Exception(
                f"Offline mode enabled and model not found at {self._model_path}. "
                "Please run 'python preload_model.py' to download the model locally."
            )
        else:
            raise Exception(
                f"Failed to load model {self._model_name}. "
                f"Please check your internet connection or run 'python preload_model.py' "
                f"to download the model locally."
            )
    
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
    
    def get_model_path(self) -> str:
        """
        Get the local path where the model is cached.
        
        Returns:
            The model cache path
        """
        return self._model_path
    
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
    
    def is_offline_mode(self) -> bool:
        """
        Check if offline mode is enabled.
        
        Returns:
            True if offline mode is enabled, False otherwise
        """
        return self._offline_mode
    
    def unload_model(self) -> None:
        """
        Unload the model from memory.
        
        This method removes the model from memory and resets the loaded state.
        Use this to free memory when the model is no longer needed.
        """
        self._model = None
        self._is_loaded = False
        logger.info("Model unloaded from memory")


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
