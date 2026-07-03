"""
Model Loader module - Singleton model loading with lazy initialization and local caching.

This module provides a singleton pattern for loading embedding models.
The model is loaded only once and reused across the application to avoid memory overhead
and loading time.

Features:
- Lazy loading: Model loads only when first requested
- Singleton pattern: Only one model instance exists
- Local caching: Checks local cache before downloading
- Configurable model: Supports dev/prod models via EMBEDDING_MODEL env var
- Streamlit support: @st.cache_resource decorator for Streamlit apps
- Diagnostics: Model Name, Load Time, Device, Cache Path, Memory Usage
- Offline mode: If OFFLINE_MODE=true, never attempts internet download
- Retry mechanism: Retries download 3 times before failing
- Comprehensive logging: Logs all loading operations
"""

from typing import Optional, Dict, Any
from threading import Lock
import os
import logging
from pathlib import Path
import time
import psutil

logger = logging.getLogger(__name__)

# Optional streamlit import for Streamlit-specific caching
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


class ModelLoader:
    """
    Singleton model loader for embedding models.
    
    This class implements the singleton pattern to ensure that the embedding model
    is loaded only once and reused across the application. The model is loaded
    lazily on first access to avoid unnecessary memory usage during import.
    
    The model loader is thread-safe and can be used in multi-threaded environments.
    
    Features:
    - Checks local cache before downloading from HuggingFace
    - Supports offline mode (OFFLINE_MODE environment variable)
    - Configurable model path (BGE_MODEL_PATH environment variable)
    - Configurable model name (EMBEDDING_MODEL environment variable)
    - Retry mechanism for failed downloads
    - Comprehensive logging
    - Diagnostics: Model Name, Load Time, Device, Cache Path, Memory Usage
    """
    
    _instance: Optional['ModelLoader'] = None
    _lock: Lock = Lock()
    _model = None
    _model_name: str = "BAAI/bge-small-en-v1.5"
    _is_loaded: bool = False
    _offline_mode: bool = False
    _model_path: str = "./models"
    _max_retries: int = 3
    
    # Diagnostics
    _load_time: float = 0.0
    _device: str = "cpu"
    _memory_usage_mb: float = 0.0
    
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
        from ..config import EMBEDDING_MODEL
        
        self._model_name = os.getenv('EMBEDDING_MODEL', EMBEDDING_MODEL)
        self._offline_mode = os.getenv('OFFLINE_MODE', 'false').lower() == 'true'
        self._model_path = os.getenv('BGE_MODEL_PATH', './models')
        self._max_retries = int(os.getenv('MODEL_DOWNLOAD_RETRIES', '3'))
        
        # Build full model path
        self._full_model_path = os.path.join(self._model_path, self._model_name.replace('/', '-'))
        
        logger.info(f"ModelLoader initialized with model: {self._model_name}")
        logger.info(f"Model cache path: {self._full_model_path}")
        logger.info(f"Offline mode: {self._offline_mode}")
    
    def _check_local_cache(self) -> bool:
        """
        Check if the model exists in the local cache.
        
        Returns:
            True if model exists in local cache, False otherwise
        """
        model_path = Path(self._full_model_path)
        return model_path.exists() and model_path.is_dir()
    
    def _load_from_local(self) -> bool:
        """
        Attempt to load the model from local cache.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            start_time = time.time()
            logger.info(f"Loading cached model from {self._full_model_path}...")
            
            self._model = SentenceTransformer(self._full_model_path)
            
            # Track load time
            self._load_time = time.time() - start_time
            
            # Get device
            self._device = str(self._model.device)
            
            # Track memory usage
            process = psutil.Process()
            self._memory_usage_mb = process.memory_info().rss / (1024 * 1024)
            
            self._is_loaded = True
            logger.info(f"Model loaded successfully from local cache in {self._load_time:.2f}s")
            logger.info(f"Device: {self._device}")
            logger.info(f"Memory usage: {self._memory_usage_mb:.2f} MB")
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
                start_time = time.time()
                logger.info(f"Downloading model {self._model_name} (attempt {attempt + 1}/{self._max_retries})...")
                from sentence_transformers import SentenceTransformer
                
                # Download to local cache path
                self._model = SentenceTransformer(
                    self._model_name,
                    cache_folder=self._model_path
                )
                
                # Track load time
                self._load_time = time.time() - start_time
                
                # Get device
                self._device = str(self._model.device)
                
                # Track memory usage
                process = psutil.Process()
                self._memory_usage_mb = process.memory_info().rss / (1024 * 1024)
                
                self._is_loaded = True
                logger.info(f"Model downloaded and loaded successfully in {self._load_time:.2f}s")
                logger.info(f"Device: {self._device}")
                logger.info(f"Memory usage: {self._memory_usage_mb:.2f} MB")
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
        Load the embedding model.
        
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
                f"Offline mode enabled and model not found at {self._full_model_path}. "
                "Please run 'python scripts/preload_models.py' to download the model locally."
            )
        else:
            raise Exception(
                f"Failed to load model {self._model_name}. "
                f"Please check your internet connection or run 'python scripts/preload_models.py' "
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
        return self._full_model_path
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information about the model loader.
        
        Returns:
            Dictionary with diagnostic information including:
                - model_name: Name of the model
                - is_loaded: Whether the model is loaded
                - load_time: Time taken to load the model (seconds)
                - device: Device the model is running on
                - cache_path: Path to the model cache
                - memory_usage_mb: Memory usage in MB
                - offline_mode: Whether offline mode is enabled
        """
        return {
            'model_name': self._model_name,
            'is_loaded': self._is_loaded,
            'load_time': self._load_time,
            'device': self._device,
            'cache_path': self._full_model_path,
            'memory_usage_mb': self._memory_usage_mb,
            'offline_mode': self._offline_mode
        }
    
    def print_diagnostics(self) -> None:
        """
        Print diagnostic information to console.
        
        This method prints a formatted summary of the model loader's state.
        """
        diagnostics = self.get_diagnostics()
        
        print("\n" + "="*60)
        print("📊 Embedding Model Diagnostics")
        print("="*60)
        print(f"Model Name: {diagnostics['model_name']}")
        print(f"Status: {'✅ Loaded' if diagnostics['is_loaded'] else '❌ Not Loaded'}")
        print(f"Load Time: {diagnostics['load_time']:.2f}s")
        print(f"Device: {diagnostics['device']}")
        print(f"Cache Path: {diagnostics['cache_path']}")
        print(f"Memory Usage: {diagnostics['memory_usage_mb']:.2f} MB")
        print(f"Offline Mode: {'✅ Enabled' if diagnostics['offline_mode'] else '❌ Disabled'}")
        print("="*60)
        
        if diagnostics['is_loaded']:
            print("🚀 Embedding Model Ready")
        else:
            print("⚠️  Model not loaded - call load_model() or get_model()")
        print("="*60 + "\n")
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            The embedding dimension based on the model
        """
        from ..config import EMBEDDING_DIM
        return EMBEDDING_DIM
    
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


def get_streamlit_model_loader() -> ModelLoader:
    """
    Get the ModelLoader instance with Streamlit caching.
    
    This function provides Streamlit-specific caching using @st.cache_resource
    to ensure the model is loaded only once per Streamlit session.
    
    Returns:
        The ModelLoader instance (cached by Streamlit)
    """
    if not STREAMLIT_AVAILABLE:
        logger.warning("Streamlit not available, falling back to regular singleton")
        return get_model_loader()
    
    @st.cache_resource
    def _get_cached_loader():
        return ModelLoader()
    
    return _get_cached_loader()
