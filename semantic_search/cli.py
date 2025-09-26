#!/usr/bin/env python3
"""
语义搜索命令行界面
提供纯命令行搜索功能，不包含Web界面
"""

import os
import json
import argparse
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from pathlib import Path

from .semantic_search import (
    SemanticSearcher, AdvancedSemanticSearcher, SearchQuery, SearchResult, SearchConfig
)
from .sync_indexer import SyncSemanticIndexer, SyncIndexerConfig, create_default_sync_config

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CommandLineInterface:
    """命令行界面"""
    
    def __init__(self, indexer: SyncSemanticIndexer, searcher: SemanticSearcher):
        self.indexer = indexer
        self.searcher = searcher
    
    def interactive_search(self):
        """交互式搜索"""
        print("🔍 Semantic Code Search - Interactive Mode")
        print("=" * 50)
        print("Commands:")
        print("  search <query>     - Search for code")
        print("  suggest <partial>  - Get search suggestions")
        print("  stats              - Show index statistics")
        print("  analytics          - Show search analytics")
        print("  quit               - Exit")
        print()
        
        while True:
            try:
                command = input("Search> ").strip()
                
                if command.lower() in ['quit', 'exit', 'q']:
                    break
                elif command.lower() == 'stats':
                    self._show_stats()
                elif command.lower() == 'analytics':
                    self._show_analytics()
                elif command.startswith('search '):
                    query_text = command[7:].strip()
                    if query_text:
                        self._perform_search(query_text)
                elif command.startswith('suggest '):
                    partial = command[8:].strip()
                    if partial:
                        self._show_suggestions(partial)
                else:
                    print("Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _perform_search(self, query_text: str):
        """执行搜索"""
        query = SearchQuery(
            query=query_text,
            query_type="semantic",
            top_k=10,
            threshold=0.0,
            include_context=True,
            include_highlights=True
        )
        
        results = self.searcher.search(query)
        
        if not results:
            print("No results found.")
            return
        
        print(f"\nFound {len(results)} results:")
        print("-" * 80)
        
        for result in results:
            context = result.context or {}
            highlights = result.highlights or []
            
            print(f"Rank {result.rank}: {context.get('function_name', 'Unknown')} "
                  f"(similarity: {result.similarity:.4f})")
            print(f"  File: {context.get('file_path', 'Unknown')}")
            print(f"  Lines: {context.get('start_line', '?')}-{context.get('end_line', '?')}")
            print(f"  Language: {context.get('language', 'unknown')}")
            print(f"  Complexity: {context.get('complexity_score', 0)}")
            
            if context.get('docstring'):
                print(f"  Docstring: {context['docstring'][:100]}...")
            
            if highlights:
                print(f"  Code preview:")
                for highlight in highlights[:2]:  # 最多显示2个高亮
                    print(f"    {highlight[:200]}...")
            
            print()
    
    def _show_suggestions(self, partial: str):
        """显示搜索建议"""
        suggestions = self.searcher.get_search_suggestions(partial, 10)
        
        if not suggestions:
            print("No suggestions found.")
            return
        
        print(f"\nSuggestions for '{partial}':")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")
        print()
    
    def _show_stats(self):
        """显示统计信息"""
        stats = self.indexer.get_stats()
        
        print("\nIndex Statistics:")
        print("-" * 40)
        print(f"Total functions: {stats.get('total_functions', 0)}")
        print(f"Total files: {stats.get('total_files', 0)}")
        print(f"Model: {stats.get('model_name', 'Unknown')}")
        print(f"Dimension: {stats.get('dimension', 0)}")
        
        if 'language_distribution' in stats:
            print("\nLanguages:")
            for lang, count in stats['language_distribution'].items():
                print(f"  {lang}: {count}")
        
        if 'function_type_distribution' in stats:
            print("\nFunction Types:")
            for func_type, count in stats['function_type_distribution'].items():
                print(f"  {func_type}: {count}")
        
        print()
    
    def _show_analytics(self):
        """显示搜索分析"""
        analytics = self.searcher.get_search_analytics()
        
        print("\nSearch Analytics:")
        print("-" * 40)
        print(f"Total searches: {analytics.get('total_searches', 0)}")
        print(f"Average result count: {analytics.get('average_result_count', 0):.1f}")
        print(f"Average top similarity: {analytics.get('average_top_similarity', 0):.4f}")
        
        if 'query_type_distribution' in analytics:
            print("\nQuery Types:")
            for query_type, count in analytics['query_type_distribution'].items():
                print(f"  {query_type}: {count}")
        
        if 'popular_queries' in analytics:
            print("\nPopular Queries:")
            for query, count in analytics['popular_queries'][:5]:
                print(f"  '{query}': {count} times")
        
        print()


def create_cli(indexer: SyncSemanticIndexer, 
              searcher: SemanticSearcher) -> CommandLineInterface:
    """创建命令行界面"""
    return CommandLineInterface(indexer, searcher)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Semantic Code Search CLI")
    parser.add_argument("--repo-path", required=True, help="Path to repository to index")
    parser.add_argument("--index-path", default="./semantic_index", help="Path to store index")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--model-type", default="sentence_transformers", 
                       choices=["sentence_transformers", "openai", "transformers", "tfidf"],
                       help="Embedding model type")
    parser.add_argument("--load-index", help="Load existing index from file")
    parser.add_argument("--save-index", help="Save index to file after indexing")
    
    args = parser.parse_args()
    
    try:
        # 创建配置
        config = create_default_sync_config(args.repo_path)
        config.index_path = args.index_path
        config.embedding_config.model_name = args.model
        config.embedding_config.model_type = args.model_type
        
        # 创建索引器
        indexer = SyncSemanticIndexer(config)
        
        # 加载或创建索引
        if args.load_index and os.path.exists(args.load_index):
            logger.info(f"Loading index from {args.load_index}")
            indexer.load_index(args.load_index)
        else:
            logger.info("Creating new index...")
            indexer.index_repository()
            
            if args.save_index:
                logger.info(f"Saving index to {args.save_index}")
                indexer.save_index(args.save_index)
        
        # 创建搜索器
        from .semantic_search import create_advanced_searcher
        searcher = create_advanced_searcher(indexer)
        
        # 运行命令行界面
        cli = create_cli(indexer, searcher)
        cli.interactive_search()
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    finally:
        if 'indexer' in locals():
            indexer.close()
    
    return 0


if __name__ == "__main__":
    exit(main())
