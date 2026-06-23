"""
Tests for the Model Loading module.

This module contains tests for the ModelLoader class, including:
- Singleton behavior
- Cache loading
- Startup loading
- Model availability
- Diagnostics
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.embeddings.model_loader import ModelLoader, get_model_loader, get_streamlit_model_loader


class TestModelLoaderSingleton:
    """Tests for ModelLoader singleton behavior."""
    
    def test_singleton_pattern(self):
        """Test that ModelLoader implements singleton pattern correctly."""
        loader1 = ModelLoader()
        loader2 = ModelLoader()
        
        # Both instances should be the same
        assert loader1 is loader2
    
    def test_get_model_loader_singleton(self):
        """Test that get_model_loader returns the same instance."""
        # Reset the global singleton for testing
        import src.embeddings.model_loader as ml_module
        ml_module._model_loader = None
        
        loader1 = get_model_loader()
        loader2 = get_model_loader()
        
        assert loader1 is loader2
    
    def test_thread_safe_singleton(self):
        """Test that singleton is thread-safe."""
        import threading
        
        # Reset the global singleton for testing
        import src.embeddings.model_loader as ml_module
        ml_module._model_loader = None
        
        instances = []
        
        def create_loader():
            loader = get_model_loader()
            instances.append(loader)
        
        threads = [threading.Thread(target=create_loader) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All instances should be the same
        assert all(inst is instances[0] for inst in instances)


class TestModelLoaderInitialization:
    """Tests for ModelLoader initialization."""
    
    def test_init_default_config(self):
        """Test initialization with default configuration."""
        # Reset the global singleton for testing
        import src.embeddings.model_loader as ml_module
        ml_module._model_loader = None
        
        loader = ModelLoader()
        
        assert loader._model_name is not None
        assert loader._is_loaded is False
        assert loader._offline_mode is False
    
    def test_init_with_env_vars(self):
        """Test initialization with environment variables."""
        # Reset the global singleton for testing
        import src.embeddings.model_loader as ml_module
        ml_module._model_loader = None
        
        # Set environment variables
        os.environ['OFFLINE_MODE'] = 'true'
        os.environ['BGE_MODEL_PATH'] = '/custom/path'
        
        try:
            loader = ModelLoader()
            
            assert loader._offline_mode is True
            assert loader._model_path == '/custom/path'
        finally:
            # Clean up
            del os.environ['OFFLINE_MODE']
            del os.environ['BGE_MODEL_PATH']
    
    def test_init_with_custom_model(self):
        """Test initialization with custom model name."""
        # Reset the global singleton for testing
        import src.embeddings.model_loader as ml_module
        ml_module._model_loader = None
        
        os.environ['EMBEDDING_MODEL'] = 'BAAI/bge-small-en-v1.5'
        
        try:
            loader = ModelLoader()
            assert 'bge-small' in loader._model_name.lower()
        finally:
            del os.environ['EMBEDDING_MODEL']


class TestModelLoaderCache:
    """Tests for ModelLoader cache functionality."""
    
    def test_check_local_cache_nonexistent(self):
        """Test checking local cache when it doesn't exist."""
        # Reset the global singleton for testing
        import src.embeddings.model_loader as ml_module
        ml_module._model_loader = None
        
        loader = ModelLoader()
        loader._full_model_path = '/nonexistent/path'
        
        assert loader._check_local_cache() is False
    
    def test_check_local_cache_exists(self):
        """Test checking local cache when it exists."""
        # Reset the global singleton for testing
        import src.embeddings.model_loader as ml_module
        ml_module._model_loader = None
        
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = ModelLoader()
            loader._full_model_path = temp_dir
            
            assert loader._check_local_cache() is True


class TestModelLoaderDiagnostics:
    """Tests for ModelLoader diagnostics."""
    
    def test_get_diagnostics(self):
        """Test getting diagnostic information."""
        # Reset the global singleton for testing
        import src.embeddings.model_loader as ml_module
        ml_module._model_loader = None
        
        loader = ModelLoader()
        diagnostics = loader.get_diagnostics()
        
        assert 'model_name' in diagnostics
        assert 'is_loaded' in diagnostics
        assert 'load_time' in diagnostics
        assert 'device' in diagnostics
        assert 'cache_path' in diagnostics
        assert 'memory_usage_mb' in diagnostics
        assert 'offline_mode' in diagnostics
    
    def test_print_diagnostics(self, capsys):
        """Test printing diagnostic information."""
        # Reset the global singleton for testing
        import src.embeddings.model_loader as ml_module
        ml_module._model_loader = None
        
        loader = ModelLoader()
        loader.print_diagnostics()
        
        captured = capsys.readouterr()
        assert 'Embedding Model Diagnostics' in captured.out
        assert 'Model Name:' in captured.out
        assert 'Status:' in captured.out


class TestModelLoaderLoading:
    """Tests for ModelLoader loading functionality."""
    
    @patch('src.embeddings.model_loader.SentenceTransformer')
    def test_load_from_local_success(self, mock_transformer):
        """Test successful loading from local cache."""
        # Reset the global singleton for testing
        import src.embeddings.model_loader as ml_module
        ml_module._model_loader = None
        
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = ModelLoader()
            loader._full_model_path = temp_dir
            
            # Mock the model
            mock_model = Mock()
            mock_model.device = 'cpu'
            mock_transformer.return_value = mock_model
            
            result = loader._load_from_local()
            
            assert result is True
            assert loader._is_loaded is True
    
    @patch('src.embeddings.model_loader.SentenceTransformer')
    def test_load_from_local_failure(self, mock_transformer):
        """Test failed loading from local cache."""
        # Reset the global singleton for testing
        import src.embeddings.model_loader as ml_module
        ml_module._model_loader = None
        
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = ModelLoader()
            loader._full_model_path = temp_dir
            
            # Mock failure
            mock_transformer.side_effect = Exception("Load failed")
            
            result = loader._load_from_local()
            
            assert result is False
            assert loader._is_loaded is False
    
    def test_download_model_offline_mode(self):
        """Test download in offline mode."""
        # Reset the global singleton for testing
        import src.embeddings.model_loader as ml_module
        ml_module._model_loader = None
        
        os.environ['OFFLINE_MODE'] = 'true'
        
        try:
            loader = ModelLoader()
            result = loader._download_model()
            
            assert result is False
        finally:
            del os.environ['OFFLINE_MODE']


class TestModelLoaderMethods:
    """Tests for ModelLoader methods."""
    
    def test_get_model_name(self):
        """Test getting model name."""
        # Reset the global singleton for testing
        import src.embeddings.model_loader as ml_module
        ml_module._model_loader = None
        
        loader = ModelLoader()
        model_name = loader.get_model_name()
        
        assert model_name is not None
        assert isinstance(model_name, str)
    
    def test_get_model_path(self):
        """Test getting model path."""
        # Reset the global singleton for testing
        import src.embeddings.model_loader as ml_module
        ml_module._model_loader = None
        
        loader = ModelLoader()
        model_path = loader.get_model_path()
        
        assert model_path is not None
        assert isinstance(model_path, str)
    
    def test_get_embedding_dimension(self):
        """Test getting embedding dimension."""
        # Reset the global singleton for testing
        import src.embeddings.model_loader as ml_module
        ml_module._model_loader = None
        
        loader = ModelLoader()
        dim = loader.get_embedding_dimension()
        
        assert dim is not None
        assert isinstance(dim, int)
        assert dim > 0
    
    def test_is_loaded(self):
        """Test checking if model is loaded."""
        # Reset the global singleton for testing
        import src.embeddings.model_loader as ml_module
        ml_module._model_loader = None
        
        loader = ModelLoader()
        assert loader.is_loaded() is False
    
    def test_is_offline_mode(self):
        """Test checking offline mode."""
        # Reset the global singleton for testing
        import src.embeddings.model_loader as ml_module
        ml_module._model_loader = None
        
        loader = ModelLoader()
        assert isinstance(loader.is_offline_mode(), bool)
    
    def test_unload_model(self):
        """Test unloading the model."""
        # Reset the global singleton for testing
        import src.embeddings.model_loader as ml_module
        ml_module._model_loader = None
        
        loader = ModelLoader()
        loader._is_loaded = True
        loader._model = Mock()
        
        loader.unload_model()
        
        assert loader._is_loaded is False
        assert loader._model is None


class TestStreamlitSupport:
    """Tests for Streamlit support."""
    
    def test_get_streamlit_model_loader_no_streamlit(self):
        """Test get_streamlit_model_loader when Streamlit is not available."""
        # Reset the global singleton for testing
        import src.embeddings.model_loader as ml_module
        ml_module._model_loader = None
        
        # Temporarily disable Streamlit
        original_available = ml_module.STREAMLIT_AVAILABLE
        ml_module.STREAMLIT_AVAILABLE = False
        
        try:
            loader = get_streamlit_model_loader()
            # Should fall back to regular singleton
            assert loader is not None
        finally:
            ml_module.STREAMLIT_AVAILABLE = original_available
    
    @patch('src.embeddings.model_loader.STREAMLIT_AVAILABLE', True)
    @patch('src.embeddings.model_loader.st')
    def test_get_streamlit_model_loader_with_streamlit(self, mock_st):
        """Test get_streamlit_model_loader when Streamlit is available."""
        # Reset the global singleton for testing
        import src.embeddings.model_loader as ml_module
        ml_module._model_loader = None
        
        # Mock Streamlit cache_resource
        mock_cache_resource = Mock()
        mock_st.cache_resource = mock_cache_resource
        
        loader = get_streamlit_model_loader()
        
        # Should use Streamlit caching
        assert loader is not None


class TestModelAvailability:
    """Tests for model availability and functionality."""
    
    @patch('src.embeddings.model_loader.SentenceTransformer')
    def test_get_model_lazy_loading(self, mock_transformer):
        """Test that get_model triggers lazy loading."""
        # Reset the global singleton for testing
        import src.embeddings.model_loader as ml_module
        ml_module._model_loader = None
        
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = ModelLoader()
            loader._full_model_path = temp_dir
            
            # Mock the model
            mock_model = Mock()
            mock_model.device = 'cpu'
            mock_transformer.return_value = mock_model
            
            # Model should not be loaded initially
            assert loader._is_loaded is False
            
            # get_model should trigger loading
            model = loader.get_model()
            
            # Now model should be loaded
            assert loader._is_loaded is True
            assert model is not None


class TestIntegration:
    """Integration tests for complete model loading workflow."""
    
    def test_complete_workflow_structure(self):
        """Test that the complete workflow structure is correct."""
        # Reset the global singleton for testing
        import src.embeddings.model_loader as ml_module
        ml_module._model_loader = None
        
        loader = ModelLoader()
        
        # Verify all methods exist
        assert hasattr(loader, 'load_model')
        assert hasattr(loader, 'get_model')
        assert hasattr(loader, 'get_model_name')
        assert hasattr(loader, 'get_model_path')
        assert hasattr(loader, 'get_embedding_dimension')
        assert hasattr(loader, 'is_loaded')
        assert hasattr(loader, 'is_offline_mode')
        assert hasattr(loader, 'unload_model')
        assert hasattr(loader, 'get_diagnostics')
        assert hasattr(loader, 'print_diagnostics')
    
    def test_diagnostics_before_and_after_load(self):
        """Test diagnostics before and after model load."""
        # Reset the global singleton for testing
        import src.embeddings.model_loader as ml_module
        ml_module._model_loader = None
        
        loader = ModelLoader()
        
        # Diagnostics before load
        diag_before = loader.get_diagnostics()
        assert diag_before['is_loaded'] is False
        assert diag_before['load_time'] == 0.0
        
        # Simulate load
        loader._is_loaded = True
        loader._load_time = 1.5
        loader._device = 'cpu'
        loader._memory_usage_mb = 500.0
        
        # Diagnostics after load
        diag_after = loader.get_diagnostics()
        assert diag_after['is_loaded'] is True
        assert diag_after['load_time'] == 1.5
        assert diag_after['device'] == 'cpu'
        assert diag_after['memory_usage_mb'] == 500.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
