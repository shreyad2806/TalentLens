import os
from dotenv import load_dotenv


load_dotenv()


# Embedding model configuration
# Production model: BAAI/bge-m3 (larger, higher quality)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# Model dimensions based on model selection
MODEL_DIMENSIONS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-m3": 1024,
    "all-MiniLM-L6-v2": 384,
}

EMBEDDING_DIM = MODEL_DIMENSIONS.get(EMBEDDING_MODEL, 1024)


CATEGORIES = [
    "HR",
    "DESIGNER",
    "INFORMATION-TECHNOLOGY",
    "TEACHER",
    "ADVOCATE",
    "BUSINESS-DEVELOPMENT",
    "HEALTHCARE",
    "FITNESS",
    "AGRICULTURE",
    "BPO",
    "SALES",
    "CONSULTANT",
    "DIGITAL-MEDIA",
    "AUTOMOBILE",
    "CHEF",
    "FINANCE",
    "APPAREL",
    "ENGINEERING",
    "ACCOUNTANT",
    "CONSTRUCTION",
    "PUBLIC-RELATIONS",
    "BANKING",
    "ARTS",
    "AVIATION",
]


def get_env(key: str, default: str = None, required: bool = False) -> str:
    value = os.getenv(key, default)
    if required and not value:
        # Fallback for testing - remove this in production
        if key == "PINECONE_API_KEY":
            value = "pcsk_4ZMhL6_BStBEJw4wYUdhYtSFHbVFidAs3jaWjEVXLAPYiHYouuAvknZKAPPuEF8nPZjTjx"
        else:
            raise RuntimeError(f"Missing required env var: {key}")
    return value


PINECONE_API_KEY = get_env("PINECONE_API_KEY", required=True)
PINECONE_INDEX = get_env("PINECONE_INDEX", default="resumes-index")
PINECONE_CLOUD = get_env("PINECONE_CLOUD", default="aws")
PINECONE_REGION = get_env("PINECONE_REGION", default="us-east-1")


