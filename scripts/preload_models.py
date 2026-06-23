"""
Model Preloading Utility

This script downloads and verifies embedding models for local caching.
It supports both development and production models and provides diagnostics
about the download and verification process.

Usage:
    python scripts/preload_models.py                    # Download default model
    python scripts/preload_models.py --model BAAI/bge-small-en-v1.5  # Download specific model
    python scripts/preload_models.py --verify-only     # Verify existing cache
    python scripts/preload_models.py --force           # Force re-download
"""

import os
import sys
import argparse
import time
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import EMBEDDING_MODEL, EMBEDDING_DIM
from src.embeddings.model_loader import ModelLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_model_size(model_path: Path) -> int:
    """
    Calculate the size of the model directory in MB.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Size in MB
    """
    if not model_path.exists():
        return 0
    
    total_size = 0
    for file_path in model_path.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    
    return total_size / (1024 * 1024)  # Convert to MB


def verify_model(model_path: Path, model_name: str) -> bool:
    """
    Verify that the model is correctly downloaded and can be loaded.
    
    Args:
        model_path: Path to the model directory
        model_name: Name of the model
        
    Returns:
        True if verification successful, False otherwise
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"Verifying model at {model_path}...")
        
        # Try to load the model
        model = SentenceTransformer(str(model_path))
        
        # Test encoding
        test_text = "This is a test sentence for verification."
        embedding = model.encode(test_text)
        
        # Verify embedding dimension
        expected_dim = 1024 if "bge-m3" in model_name else 384
        if len(embedding) != expected_dim:
            logger.warning(f"Embedding dimension mismatch: expected {expected_dim}, got {len(embedding)}")
            return False
        
        logger.info(f"✅ Model verification successful")
        logger.info(f"   Embedding dimension: {len(embedding)}")
        logger.info(f"   Test encoding successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model verification failed: {str(e)}")
        return False


def preload_model(model_name: str, force: bool = False, verify_only: bool = False) -> dict:
    """
    Preload the embedding model to local cache.
    
    Args:
        model_name: Name of the model to preload
        force: Force re-download even if cache exists
        verify_only: Only verify existing cache, don't download
        
    Returns:
        Dictionary with preload results
    """
    result = {
        'model_name': model_name,
        'success': False,
        'cached': False,
        'verified': False,
        'cache_path': None,
        'model_size_mb': 0,
        'load_time': 0,
        'error': None
    }
    
    # Set environment variables for model loader
    os.environ['EMBEDDING_MODEL'] = model_name
    
    # Initialize model loader
    logger.info(f"Initializing model loader for: {model_name}")
    loader = ModelLoader()
    
    # Get cache path
    cache_path = Path(loader.get_model_path())
    result['cache_path'] = str(cache_path)
    
    # Check if cache exists
    cache_exists = loader._check_local_cache()
    result['cached'] = cache_exists
    
    if cache_exists:
        model_size = get_model_size(cache_path)
        result['model_size_mb'] = model_size
        logger.info(f"Model cache exists at: {cache_path}")
        logger.info(f"Model size: {model_size:.2f} MB")
    
    # If verify_only, just verify and return
    if verify_only:
        if not cache_exists:
            logger.error("Cache does not exist, cannot verify")
            result['error'] = "Cache does not exist"
            return result
        
        logger.info("Verify-only mode, skipping download")
        verified = verify_model(cache_path, model_name)
        result['verified'] = verified
        result['success'] = verified
        return result
    
    # If cache exists and not force, verify and return
    if cache_exists and not force:
        logger.info("Cache exists, verifying...")
        verified = verify_model(cache_path, model_name)
        result['verified'] = verified
        
        if verified:
            logger.info("✅ Model already cached and verified")
            result['success'] = True
        else:
            logger.warning("Cache exists but verification failed, re-downloading...")
            force = True
        
        if not force:
            return result
    
    # Download the model
    logger.info("Downloading model...")
    start_time = time.time()
    
    try:
        loader.load_model()
        load_time = time.time() - start_time
        result['load_time'] = load_time
        
        logger.info(f"✅ Model downloaded and loaded successfully in {load_time:.2f}s")
        
        # Get updated cache path and size
        cache_path = Path(loader.get_model_path())
        result['cache_path'] = str(cache_path)
        model_size = get_model_size(cache_path)
        result['model_size_mb'] = model_size
        
        # Verify the downloaded model
        verified = verify_model(cache_path, model_name)
        result['verified'] = verified
        result['success'] = verified
        
        if verified:
            logger.info(f"✅ Model preloaded successfully")
            logger.info(f"   Cache path: {cache_path}")
            logger.info(f"   Model size: {model_size:.2f} MB")
            logger.info(f"   Load time: {load_time:.2f}s")
        else:
            logger.error("❌ Model download failed verification")
            result['error'] = "Verification failed after download"
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to preload model: {str(e)}")
        result['error'] = str(e)
        return result


def print_summary(result: dict) -> None:
    """
    Print a summary of the preload operation.
    
    Args:
        result: Dictionary with preload results
    """
    print("\n" + "="*70)
    print("📊 Model Preload Summary")
    print("="*70)
    print(f"Model Name: {result['model_name']}")
    print(f"Status: {'✅ Success' if result['success'] else '❌ Failed'}")
    print(f"Cache Path: {result['cache_path']}")
    print(f"Model Size: {result['model_size_mb']:.2f} MB")
    print(f"Load Time: {result['load_time']:.2f}s")
    print(f"Cached: {'✅ Yes' if result['cached'] else '❌ No'}")
    print(f"Verified: {'✅ Yes' if result['verified'] else '❌ No'}")
    
    if result['error']:
        print(f"Error: {result['error']}")
    
    print("="*70)
    
    if result['success']:
        print("🚀 Model Ready for Production Use")
    else:
        print("⚠️  Model Preload Failed")
    print("="*70 + "\n")


def main():
    """Main entry point for the preload script."""
    parser = argparse.ArgumentParser(
        description='Preload embedding models for local caching'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=EMBEDDING_MODEL,
        help='Model name to preload (default: from config or BAAI/bge-m3)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if cache exists'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing cache, do not download'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🔄 Embedding Model Preload Utility")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Force: {args.force}")
    print(f"Verify Only: {args.verify_only}")
    print("="*70 + "\n")
    
    # Preload the model
    result = preload_model(
        model_name=args.model,
        force=args.force,
        verify_only=args.verify_only
    )
    
    # Print summary
    print_summary(result)
    
    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()
