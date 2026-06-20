"""
Model Loader module for Cross-Encoder Reranker.

This module provides a singleton model loader for cross-encoder models with
lazy loading, offline caching, and configurable model support.

Architecture Notes:
- Singleton Pattern: Ensures only one model instance per model name
- Lazy Loading: Model loaded only when first accessed
- Offline Cache: Supports offline mode with cached models
- Thread-Safe: Uses thread-safe singleton implementation

Cross-Encoder Models:
Cross-encoders are neural network models that take a query-document pair as input
and output a relevance score. Unlike bi-encoders (which encode queries and documents
separately), cross-encoders jointly process the pair, allowing for more accurate
relevance assessment at the cost of higher computational overhead.

The cross-encoder architecture:
1. Input: [CLS] query [SEP] document [SEP]
2. BERT-based encoder processes the sequence
3. Classification head outputs relevance score
4. Score typically in range [-10, 10] or [0, 1] depending on model

SOLID Principles Applied:
- Single Responsibility: Only handles model loading
- Open/Closed: Can be extended with new model types
- Dependency Inversion: Depends on model abstraction
- Interface Segregation: Focused loading interface
"""

import logging
import os
import time
import threading
from typing import Optional, Dict, Any
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class RerankerModel(str, Enum):
    """
    Enumeration for supported reranker models.
    
    This enum defines the supported cross-encoder models for reranking.
    Models are categorized by their intended use case.
    """
    
    # Default model: Lightweight, fast inference
    MINILM_V2 = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Production model: High accuracy, larger model
    BGE_RERANKER_V2_M3 = "BAAI/bge-reranker-v2-m3"
    
    # Alternative models
    BGE_RERANKER_BASE = "BAAI/bge-reranker-base"
    MS_MARCO_BERT = "cross-encoder/ms-marco-bert-base-uncased"


class ModelLoader:
    """
    Singleton model loader for cross-encoder models.
    
    This class implements a thread-safe singleton pattern for loading
    cross-encoder models. It supports lazy loading (model loaded only when
    first accessed), offline caching (models cached locally), and configurable
    model selection.
    
    Architecture Pattern: Singleton Pattern
    - Ensures only one instance per model name
    - Thread-safe initialization
    - Lazy loading for efficiency
    - Offline mode support
    
    Thread Safety:
    The singleton implementation uses double-checked locking to ensure
    thread-safe initialization while avoiding the overhead of synchronization
    after the instance is created.
    
    Offline Mode:
    When offline mode is enabled, the loader attempts to load models from
    a local cache directory. If the model is not found in the cache, it
    raises an exception rather than attempting to download.
    
    Attributes:
        model_name: Name of the model to load
        offline_mode: Whether to operate in offline mode
        cache_dir: Directory for caching models
        _instance: Singleton instance (class-level)
        _lock: Thread lock for singleton initialization
        _model: Loaded model instance
        _model_loaded: Whether model has been loaded
    """
    
    # Class-level singleton instances and locks
    _instances: Dict[str, 'ModelLoader'] = {}
    _locks: Dict[str, threading.Lock] = {}
    
    def __new__(
        cls,
        model_name: str = RerankerModel.MINILM_V2.value,
        offline_mode: bool = False,
        cache_dir: Optional[str] = None
    ):
        """
        Create or get singleton instance for the given model name.
        
        This implements the singleton pattern with thread-safe initialization
        using double-checked locking. Each model name has its own singleton
        instance to support multiple models simultaneously.
        
        Args:
            model_name: Name of the model to load
            offline_mode: Whether to operate in offline mode
            cache_dir: Directory for caching models
            
        Returns:
            Singleton instance for the model
        """
        # Get or create lock for this model name
        if model_name not in cls._locks:
            cls._locks[model_name] = threading.Lock()
        
        lock = cls._locks[model_name]
        
        # Double-checked locking for thread-safe singleton
        if model_name not in cls._instances:
            with lock:
                # Check again in case another thread created it
                if model_name not in cls._instances:
                    instance = super().__new__(cls)
                    instance.model_name = model_name
                    instance.offline_mode = offline_mode
                    instance.cache_dir = cache_dir or os.path.join(
                        os.getcwd(),
                        "models",
                        model_name.replace("/", "-")
                    )
                    instance._model = None
                    instance._model_loaded = False
                    cls._instances[model_name] = instance
        
        return cls._instances[model_name]
    
    def __init__(
        self,
        model_name: str = RerankerModel.MINILM_V2.value,
        offline_mode: bool = False,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the model loader.
        
        Note: Actual initialization happens in __new__ for singleton pattern.
        This method is kept for compatibility and documentation.
        
        Args:
            model_name: Name of the model to load
            offline_mode: Whether to operate in offline mode
            cache_dir: Directory for caching models
        """
        pass  # Initialization handled in __new__
    
    def get_model(self):
        """
        Get the loaded cross-encoder model.
        
        This method implements lazy loading - the model is loaded only when
        first accessed. Subsequent calls return the cached model instance.
        
        The loading process:
        1. Check if model is already loaded
        2. If not, attempt to load from cache
        3. If not in cache and offline mode, raise exception
        4. If not in cache and online mode, download and cache
        5. Return loaded model
        
        Returns:
            Loaded cross-encoder model instance
            
        Raises:
            Exception: If model cannot be loaded in offline mode
        """
        if self._model_loaded and self._model is not None:
            logger.debug(f"Model {self.model_name} already loaded, returning cached instance")
            return self._model
        
        logger.info(f"Loading cross-encoder model: {self.model_name}")
        start_time = time.time()
        
        try:
            # Check if model exists in cache
            cache_path = Path(self.cache_dir)
            if cache_path.exists():
                logger.info(f"Local cache found at {self.cache_dir}")
                try:
                    from sentence_transformers import CrossEncoder
                    self._model = CrossEncoder(self.cache_dir)
                    self._model_loaded = True
                    load_time = time.time() - start_time
                    logger.info(f"Loaded model from local cache in {load_time:.2f}s")
                    return self._model
                except Exception as e:
                    logger.warning(f"Failed to load model from local cache: {e}")
            
            # If offline mode and no cache, raise exception
            if self.offline_mode:
                raise Exception(
                    f"Offline mode enabled and model not found at {self.cache_dir}. "
                    f"Please download the model first or disable offline mode."
                )
            
            # Download and cache the model
            logger.info(f"Downloading model {self.model_name}...")
            from sentence_transformers import CrossEncoder
            
            # Create cache directory
            cache_path.mkdir(parents=True, exist_ok=True)
            
            # Download and save model
            self._model = CrossEncoder(self.model_name)
            self._model.save(self.cache_dir)
            
            self._model_loaded = True
            load_time = time.time() - start_time
            logger.info(f"Downloaded and cached model in {load_time:.2f}s")
            
            return self._model
            
        except ImportError as e:
            logger.error(f"sentence-transformers not installed: {e}")
            raise Exception(
                "sentence-transformers package is required for cross-encoder models. "
                "Install it with: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def is_loaded(self) -> bool:
        """
        Check if the model is loaded.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self._model_loaded and self._model is not None
    
    def unload(self):
        """
        Unload the model from memory.
        
        This method releases the model from memory and resets the loaded state.
        The model can be reloaded by calling get_model() again.
        
        This is useful for freeing memory when the model is no longer needed.
        """
        if self._model is not None:
            logger.info(f"Unloading model {self.model_name}")
            self._model = None
            self._model_loaded = False
    
    @classmethod
    def clear_all_instances(cls):
        """
        Clear all singleton instances.
        
        This class method clears all cached singleton instances, allowing
        them to be reloaded with new configurations. This is primarily
        useful for testing and debugging.
        """
        logger.info("Clearing all model loader instances")
        cls._instances.clear()
        cls._locks.clear()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information including name, loaded status,
            cache directory, and model metadata if available
        """
        info = {
            "model_name": self.model_name,
            "loaded": self.is_loaded(),
            "cache_dir": self.cache_dir,
            "offline_mode": self.offline_mode
        }
        
        if self.is_loaded() and self._model is not None:
            try:
                # Try to get model metadata
                if hasattr(self._model, 'config'):
                    info["config"] = str(self._model.config)
                if hasattr(self._model, 'num_parameters'):
                    info["num_parameters"] = self._model.num_parameters()
            except Exception as e:
                logger.warning(f"Could not retrieve model metadata: {e}")
        
        return info
