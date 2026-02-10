import os
from dotenv import load_dotenv

load_dotenv()

# LLM API設定（プライマリ）
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://new.lemonapi.site/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "[V]gemini-3-pro-preview")

# LLM API設定（フォールバック）
LLM_FALLBACK_API_KEY = os.environ.get("LLM_FALLBACK_API_KEY", "REDACTED_FALLBACK_KEY")
LLM_FALLBACK_BASE_URL = os.environ.get("LLM_FALLBACK_BASE_URL", "https://api.12ai.org/v1")
LLM_FALLBACK_MODEL = os.environ.get("LLM_FALLBACK_MODEL", "gemini-3-pro-preview")

# Embedding API設定
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "REDACTED_OLD_EMBEDDING_KEY")
EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")

# リトライ設定
LLM_MAX_RETRIES = 2  # プライマリAPIのリトライ回数

# パラメータ
TAU = 1.0
TOP_K_TESTS = 10

# パス
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DISEASES_JSONL = os.path.join(DATA_DIR, "diseases.jsonl")
TESTS_JSONL = os.path.join(DATA_DIR, "tests.jsonl")
FINDINGS_JSONL = os.path.join(DATA_DIR, "findings.jsonl")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
DISEASE_LIST_FILE = os.path.join(PROJECT_DIR, "疾患リスト.txt")
TEST_LIST_FILE = os.path.join(PROJECT_DIR, "検査リスト.txt")
