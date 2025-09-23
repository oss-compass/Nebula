#!/usr/bin/env python3

# 基础组件
from .vector_embedding import (
    CodeVectorizer, CodeEmbeddingManager, EmbeddingConfig, CodeEmbedding,
    create_default_config, create_openai_config
)

from .sync_indexer import (
    SyncSemanticIndexer, SyncIndexerConfig, IndexedFile,
    create_default_sync_config
)

# 核心搜索功能
from .single_repo import (
    SingleRepoSearch, SearchQuery, SearchResult, SearchConfig,
    create_single_repo_search
)

from .multi_repo import (
    MultiRepoSearch, MultiRepoSearchConfig, RepoConfig, CrossRepoResult, MultiRepoSearchResult,
    create_multi_repo_search
)

from .similarity_analysis import (
    APISimilarityAnalyzer, APISimilarityConfig, APISignature, SimilarityResult, APIAnalysisResult,
    create_api_similarity_analyzer
)

__version__ = "2.0.0"
__author__ = "OSSCompass Team"

__all__ = [
    # 向量嵌入
    "CodeVectorizer",
    "CodeEmbeddingManager", 
    "EmbeddingConfig",
    "CodeEmbedding",
    "create_default_config",
    "create_openai_config",
    
    # Sync 索引�?
    "SyncSemanticIndexer",
    "SyncIndexerConfig",
    "IndexedFile",
    "create_default_sync_config",
    
    # 单库搜索
    "SingleRepoSearch",
    "SearchQuery",
    "SearchResult",
    "SearchConfig",
    "create_single_repo_search",
    
    # 多库搜索
    "MultiRepoSearch",
    "MultiRepoSearchConfig",
    "RepoConfig",
    "CrossRepoResult",
    "MultiRepoSearchResult",
    "create_multi_repo_search",
    
    # 相似性分�?
    "APISimilarityAnalyzer",
    "APISimilarityConfig",
    "APISignature",
    "SimilarityResult",
    "APIAnalysisResult",
    "create_api_similarity_analyzer"
]
