import os
from typing import Optional

class SearchConfig:
    """搜索配置�?""
    
    def __init__(self):
        # Neo4j连接配置
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "90879449Drq")
        self.neo4j_database = os.getenv("NEO4J_DATABASE")
        
        # 搜索配置
        self.default_limit = 10
        self.max_depth = 5
        self.similarity_threshold = 0.3
        
        # 嵌入模型配置
        self.embedding_model = "all-MiniLM-L6-v2"
        self.embedding_device = "cpu"
        
        # OpenAI配置（可选）
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = "gpt-3.5-turbo"
        
        # 图分析配�?
        self.centrality_algorithms = ["pagerank", "betweenness", "closeness", "eigenvector"]
        self.community_algorithms = ["louvain", "leiden", "label_propagation"]
        
        # 缓存配置
        self.enable_cache = True
        self.cache_size = 1000
        self.cache_ttl = 3600  # 1小时

# 全局配置实例
config = SearchConfig()
