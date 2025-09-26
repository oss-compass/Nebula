#!/usr/bin/env python3
"""
语义搜索模块
基于 sync 库和代码向量化的语义搜索功能
"""

from .vector_embedding import (
    CodeVectorizer, CodeEmbeddingManager, EmbeddingConfig, CodeEmbedding,
    create_default_config, create_openai_config
)

from .sync_indexer import (
    SyncSemanticIndexer, SyncIndexerConfig, IndexedFile,
    create_default_sync_config
)

from .semantic_search import (
    SemanticSearcher, AdvancedSemanticSearcher, SearchQuery, SearchResult, SearchConfig,
    create_semantic_searcher, create_advanced_searcher
)

# from .search_api import (
#     SearchAPI, CommandLineInterface,
#     create_search_api, create_cli
# )

from .hybrid_search import (
    HybridSearcher, GraphSearcher, HybridSearchResult, HybridSearchConfig,
    create_hybrid_searcher, create_default_hybrid_config
)

__version__ = "1.0.0"
__author__ = "Semantic Search Team"

__all__ = [
    # 向量嵌入
    "CodeVectorizer",
    "CodeEmbeddingManager", 
    "EmbeddingConfig",
    "CodeEmbedding",
    "create_default_config",
    "create_openai_config",
    
    # Sync 索引器
    "SyncSemanticIndexer",
    "SyncIndexerConfig",
    "IndexedFile",
    "create_default_sync_config",
    
    # 语义搜索
    "SemanticSearcher",
    "AdvancedSemanticSearcher",
    "SearchQuery",
    "SearchResult",
    "SearchConfig",
    "create_semantic_searcher",
    "create_advanced_searcher",
    
    # 搜索API (暂时不可用)
    # "SearchAPI",
    # "CommandLineInterface",
    # "create_search_api",
    # "create_cli",
    
    # 混合搜索
    "HybridSearcher",
    "GraphSearcher",
    "HybridSearchResult",
    "HybridSearchConfig",
    "create_hybrid_searcher",
    "create_default_hybrid_config"
]
