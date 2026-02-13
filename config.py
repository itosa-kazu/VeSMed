import os
from dotenv import load_dotenv

load_dotenv()

# LLM API設定（プライマリ）
LLM_API_KEY = os.environ.get("LLM_API_KEY", "REDACTED_LLM_KEY")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://new.lemonapi.site/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "[V]gemini-3-pro-preview")

# LLM API設定（メタデータ生成用 — pro）
GENERATE_LLM_API_KEY = os.environ.get("GENERATE_LLM_API_KEY", "REDACTED_GENERATE_KEY")
GENERATE_LLM_BASE_URL = os.environ.get("GENERATE_LLM_BASE_URL", "https://new.lemonapi.site/v1")
GENERATE_LLM_MODEL = os.environ.get("GENERATE_LLM_MODEL", "[V]gemini-3-pro-preview")

# LLM API設定（フォールバック）
LLM_FALLBACK_API_KEY = os.environ.get("LLM_FALLBACK_API_KEY", "REDACTED_FALLBACK_KEY")
LLM_FALLBACK_BASE_URL = os.environ.get("LLM_FALLBACK_BASE_URL", "https://api.12ai.org/v1")
LLM_FALLBACK_MODEL = os.environ.get("LLM_FALLBACK_MODEL", "gemini-3-pro-preview")

# Claude API設定（高品質メタデータ生成用）
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "REDACTED_CLAUDE_KEY")
CLAUDE_BASE_URL = os.environ.get("CLAUDE_BASE_URL", "https://gobuild.club/v1")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-4-6-thinking")

# Embedding API設定
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "REDACTED_EMBEDDING_KEY")
EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "https://openrouter.ai/api/v1")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")

# Vertex AI設定（プライマリLLM — 直接Google API、低遅延）
VERTEX_SA_KEY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thermal-outlet-483512-m4-8ec9647654b6.json")
VERTEX_PROJECT = "thermal-outlet-483512-m4"
VERTEX_LOCATION = "global"
VERTEX_MODEL = "gemini-3-flash-preview"

# リトライ設定
LLM_MAX_RETRIES = 2  # フォールバックAPIのリトライ回数

# パラメータ
TAU = 1.0
TOP_K_TESTS = 10

# パス
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DISEASES_JSONL = os.path.join(DATA_DIR, "diseases.jsonl")
TESTS_JSONL = os.path.join(DATA_DIR, "tests.jsonl")
FINDINGS_JSONL = os.path.join(DATA_DIR, "findings.jsonl")
HPE_ITEMS_JSONL = os.path.join(DATA_DIR, "hpe_items.jsonl")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
DISEASE_LIST_FILE = os.path.join(PROJECT_DIR, "疾患リスト.txt")
TEST_LIST_FILE = os.path.join(PROJECT_DIR, "検査リスト.txt")
