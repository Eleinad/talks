import os
from dotenv import load_dotenv

load_dotenv()

# CLIP model
CLIP_MODEL_NAME: str = os.getenv("CLIP_MODEL_NAME", "clip-ViT-B-32")
CLIP_CACHE_DIR: str = os.getenv("CLIP_CACHE_DIR", "./clip_model")

# BLIP model
BLIP_MODEL_NAME: str = os.getenv("BLIP_MODEL_NAME", "Salesforce/blip-image-captioning-base")

# Server
API_URL: str = os.getenv("API_URL", "http://localhost:8000")
STREAMLIT_ORIGIN: str = os.getenv("STREAMLIT_ORIGIN", "http://localhost:8501")

# Paths
DEFAULT_IMAGES_DIR: str = os.getenv("DEFAULT_IMAGES_DIR", "")
FAISS_ROOT: str = os.getenv("FAISS_ROOT", "./faiss")
BERTOPIC_ROOT: str = os.getenv("BERTOPIC_ROOT", "./bertTopic")

# Optuna
OPTUNA_N_TRIALS: int = int(os.getenv("OPTUNA_N_TRIALS", "500"))
OPTUNA_TIMEOUT: int = int(os.getenv("OPTUNA_TIMEOUT", "300"))
