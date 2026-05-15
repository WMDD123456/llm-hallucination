import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
QUESTIONS_FILE = DATA_DIR / "questions.json"
KB_DIR = DATA_DIR / "knowledge_base"
KB_INDEX_DIR = DATA_DIR / "kb_index"
RESULTS_DIR = DATA_DIR / "results"

TEMPERATURE = 0.1
MAX_TOKENS = 512

MODEL_CONFIGS = {
    "deepseek-chat": {
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com/v1",
        "display_name": "DeepSeek-Chat",
    },
    "qwen-plus": {
        "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "display_name": "Qwen-Plus",
    },
    "glm-4-flash": {
        "api_key": os.getenv("ZHIPU_API_KEY", ""),
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "display_name": "GLM-4-Flash",
    },
}

SYSTEM_PROMPT = "你是一个严谨的问答助手。请根据你的知识回答问题。如果你不确定答案，请回答'无法确定'。"

RAG_CONFIG = {
    "embedding_model": "BAAI/bge-large-zh-v1.5",
    "top_k": 3,
    "chunk_size": 500,
    "chunk_overlap": 50,
}


def get_available_models():
    """返回已配置 API key 的模型列表"""
    return [k for k, v in MODEL_CONFIGS.items() if v["api_key"]]


def get_model_config(model_name):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"未知模型: {model_name}")
    cfg = MODEL_CONFIGS[model_name]
    if not cfg["api_key"]:
        raise ValueError(f"模型 {model_name} 未配置 API key，请在 .env 文件中设置")
    return cfg
