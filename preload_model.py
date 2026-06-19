"""
Preload Model Script - Download and cache BAAI/bge-m3 embedding model locally.

This script downloads the BAAI/bge-m3 model from HuggingFace and caches it locally
for offline use. This ensures that future tests and production runs never require
internet connectivity for model loading.

Usage:
    python preload_model.py

Environment Variables:
    BGE_MODEL_PATH: Path where to cache the model (default: ./models/bge-m3)
    MODEL_DOWNLOAD_RETRIES: Number of download retry attempts (default: 3)
"""

import os
import sys
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_model(model_name: str = "BAAI/bge-m3", 
                  cache_path: str = "./models/bge-m3",
                  max_retries: int = 3) -> bool:
    """
    Download the embedding model from HuggingFace and cache it locally.
    
    Args:
        model_name: Name of the model to download
        cache_path: Local path where to cache the model
        max_retries: Number of download retry attempts
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("sentence-transformers is not installed. Please run: pip install sentence-transformers")
        return False
    
    # Create cache directory if it doesn't exist
    cache_dir = Path(cache_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading model {model_name} to {cache_path}...")
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Download attempt {attempt + 1}/{max_retries}...")
            
            # Download and cache the model
            model = SentenceTransformer(
                model_name,
                cache_folder=cache_path
            )
            
            # Test the model to ensure it works
            logger.info("Testing model...")
            test_embedding = model.encode("test")
            logger.info(f"Model test successful. Embedding dimension: {len(test_embedding)}")
            
            logger.info(f"✅ Model successfully downloaded and cached to {cache_path}")
            logger.info(f"Model size: {sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file()) / (1024*1024):.2f} MB")
            return True
            
        except Exception as e:
            logger.warning(f"Download attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error("All download attempts failed")
                return False
    
    return False


def main():
    """Main entry point for the preload_model script."""
    # Get configuration from environment variables
    model_name = os.getenv('BGE_MODEL_NAME', 'BAAI/bge-m3')
    cache_path = os.getenv('BGE_MODEL_PATH', './models/bge-m3')
    max_retries = int(os.getenv('MODEL_DOWNLOAD_RETRIES', '3'))
    
    logger.info("=" * 80)
    logger.info("BAAI/bge-m3 Model Preload Script")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Cache Path: {cache_path}")
    logger.info(f"Max Retries: {max_retries}")
    logger.info("=" * 80)
    
    # Download the model
    success = download_model(model_name, cache_path, max_retries)
    
    if success:
        logger.info("=" * 80)
        logger.info("✅ Model preload completed successfully")
        logger.info("=" * 80)
        logger.info("You can now run tests and production code in offline mode by setting:")
        logger.info("  export OFFLINE_MODE=true")
        logger.info("  # or on Windows:")
        logger.info("  set OFFLINE_MODE=true")
        logger.info("=" * 80)
        sys.exit(0)
    else:
        logger.error("=" * 80)
        logger.error("❌ Model preload failed")
        logger.error("=" * 80)
        logger.error("Please check your internet connection and try again.")
        logger.error("If the issue persists, the HuggingFace service may be temporarily unavailable.")
        logger.error("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
