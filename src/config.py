import os
from dotenv import load_dotenv


load_dotenv()


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


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


