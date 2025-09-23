#!/usr/bin/env python3

import os
from pathlib import Path

# 项目根目�?
PROJECT_ROOT = Path(__file__).parent

# Neo4j 配置
NEO4J_CONFIG = {
    "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    "user": os.getenv("NEO4J_USER", "neo4j"),
    "password": os.getenv("NEO4J_PASSWORD", "90879449Drq"),
    "database": os.getenv("NEO4J_DATABASE", "neo4j")
}

# AI 配置
AI_CONFIG = {
    "model": os.getenv("AI_MODEL", "gpt-3.5-turbo"),
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "max_tokens": int(os.getenv("AI_MAX_TOKENS", "1000")),
    "temperature": float(os.getenv("AI_TEMPERATURE", "0.1"))
}

# 向量嵌入配置
EMBEDDING_CONFIG = {
    "model_name": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    "model_type": os.getenv("EMBEDDING_TYPE", "sentence_transformers"),
    "batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
    "cache_dir": os.getenv("EMBEDDING_CACHE_DIR", "./embedding_cache")
}

# 文件路径配置
PATHS = {
    "extract_output": PROJECT_ROOT / "output" / "extract_output",
    "description_output": PROJECT_ROOT / "output" / "description_output", 
    "vector_output": PROJECT_ROOT / "output" / "vector_embedding_output",
    "ingest_output": PROJECT_ROOT / "output" / "ingest_output",
    "cache": PROJECT_ROOT / "cache",
    "logs": PROJECT_ROOT / "logs"
}

# 创建必要的目�?
for path in PATHS.values():
    path.mkdir(exist_ok=True)

# 支持的文件扩展名
SUPPORTED_EXTENSIONS = {
    '.py': 'python',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.java': 'java',
    '.cpp': 'cpp',
    '.c': 'c',
    '.rs': 'rust',
    '.go': 'go',
    '.rb': 'ruby',
    '.cs': 'csharp'
}

# 日志配置
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": PATHS["logs"] / "osscompass.log"
}
