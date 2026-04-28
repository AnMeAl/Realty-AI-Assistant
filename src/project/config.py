import os
from dotenv import load_dotenv

load_dotenv()

# MinIO
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://localhost:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
BUCKET_NAME = os.getenv("BUCKET_NAME", "realty-images")

# OpenAI через ProxyAPI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.proxyapi.ru/openai/v1")

# Пути
DATA_PATH = "project/data/final"
EMBEDDINGS_PATH = "project/data/embeddings"
INDEX_PATH = "project/data/multimodal_index"

# Модели
TEXT_MODEL_NAME = "cointegrated/rubert-tiny2"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Параметры поиска
DEFAULT_TOP_K = 5
MAX_IMAGES_PER_APARTMENT = 5