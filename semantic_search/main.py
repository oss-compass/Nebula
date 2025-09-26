#!/usr/bin/env python3

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from .vector_embedding import create_default_config, create_openai_config
from .sync_indexer import create_default_sync_config, SyncSemanticIndexer
from .semantic_search import create_advanced_searcher
from .cli import create_cli
from .hybrid_search import create_hybrid_searcher, create_default_hybrid_config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_embedding_config(args) -> dict:
    """设置嵌入配置"""
    if args.model_type == "openai":
        if not args.openai_api_key:
            raise ValueError("OpenAI API key required for OpenAI model")
        return create_openai_config(args.openai_api_key)
    else:
        config = create_default_config()
        config.model_name = args.model
        config.model_type = args.model_type
        return config


def setup_sync_config(args) -> dict:
    """设置Sync配置"""
    config = create_default_sync_config(args.repo_path)
    config.index_path = args.index_path
    config.embedding_config = setup_embedding_config(args)
    
    # 设置文件过滤（如果参数存在）
    if hasattr(args, 'include_extensions') and args.include_extensions:
        config.include_extensions = args.include_extensions.split(',')
    if hasattr(args, 'exclude_patterns') and args.exclude_patterns:
        config.exclude_patterns = args.exclude_patterns.split(',')
    if hasattr(args, 'max_file_size') and args.max_file_size:
        config.max_file_size = args.max_file_size
    
    return config


def setup_hybrid_config(args) -> dict:
    """设置混合搜索配置"""
    config = create_default_hybrid_config()
    
    if args.neo4j_uri:
        config.neo4j_uri = args.neo4j_uri
    if args.neo4j_user:
        config.neo4j_user = args.neo4j_user
    if args.neo4j_password:
        config.neo4j_password = args.neo4j_password
    if args.neo4j_database:
        config.neo4j_database = args.neo4j_database
    
    if args.vector_weight is not None:
        config.vector_weight = args.vector_weight
    if args.graph_weight is not None:
        config.graph_weight = args.graph_weight
    
    return config


def index_repository(args):
    """索引代码库"""
    logger.info(f"Starting repository indexing: {args.repo_path}")
    
    # 创建配置
    sync_config = setup_sync_config(args)
    
    # 创建索引器
    indexer = SyncSemanticIndexer(sync_config)
    
    try:
        # 执行索引
        stats = indexer.index_repository()
        
        logger.info("Indexing completed successfully!")
        logger.info(f"Stats: {json.dumps(stats, indent=2)}")
        
        # 保存索引
        if args.save_index:
            logger.info(f"Saving index to {args.save_index}")
            indexer.save_index(args.save_index)
        
        return indexer
        
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise
    finally:
        indexer.close()


def run_search_server(args):
    """运行搜索服务器"""
    logger.info("Starting semantic search server...")
    
    # 创建配置
    sync_config = setup_sync_config(args)
    hybrid_config = setup_hybrid_config(args)
    
    # 创建索引器
    indexer = SyncSemanticIndexer(sync_config)
    
    try:
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
        if args.enable_hybrid_search:
            logger.info("Creating hybrid searcher...")
            semantic_searcher = create_advanced_searcher(indexer)
            searcher = create_hybrid_searcher(semantic_searcher, hybrid_config)
        else:
            logger.info("Creating semantic searcher...")
            searcher = create_advanced_searcher(indexer)
        
        # 运行命令行界面
        logger.info("Starting CLI interface...")
        cli = create_cli(indexer, searcher)
        cli.interactive_search()
        
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        raise
    finally:
        if 'searcher' in locals() and hasattr(searcher, 'close'):
            searcher.close()
        indexer.close()


def run_search_command(args):
    """运行搜索命令"""
    logger.info(f"Running search command: {args.query}")
    
    # 创建配置
    sync_config = setup_sync_config(args)
    hybrid_config = setup_hybrid_config(args)
    
    # 创建索引器
    indexer = SyncSemanticIndexer(sync_config)
    
    try:
        # 加载索引
        if args.load_index and os.path.exists(args.load_index):
            logger.info(f"Loading index from {args.load_index}")
            indexer.load_index(args.load_index)
        else:
            logger.error("No index file specified or index file not found")
            return 1
        
        # 创建搜索器
        if args.enable_hybrid_search:
            semantic_searcher = create_advanced_searcher(indexer)
            searcher = create_hybrid_searcher(semantic_searcher, hybrid_config)
        else:
            searcher = create_advanced_searcher(indexer)
        
        # 执行搜索
        from .semantic_search import SearchQuery
        
        query = SearchQuery(
            query=args.query,
            query_type=args.query_type,
            top_k=args.top_k,
            threshold=args.threshold,
            filters=json.loads(args.filters) if args.filters else {}
        )
        
        results = searcher.search(query)
        
        # 输出结果
        if args.output_format == "json":
            output_data = []
            for result in results:
                output_data.append({
                    "rank": result.rank,
                    "similarity": result.similarity,
                    "function_name": result.embedding.metadata.get("name"),
                    "file_path": result.embedding.metadata.get("filepath"),
                    "context": result.context,
                    "highlights": result.highlights
                })
            print(json.dumps(output_data, indent=2, ensure_ascii=False))
        else:
            # 文本输出
            print(f"Found {len(results)} results for query: '{args.query}'")
            print("=" * 80)
            
            for result in results:
                context = result.context or {}
                print(f"Rank {result.rank}: {context.get('function_name', 'Unknown')} "
                      f"(similarity: {result.similarity:.4f})")
                print(f"  File: {context.get('file_path', 'Unknown')}")
                print(f"  Lines: {context.get('start_line', '?')}-{context.get('end_line', '?')}")
                print(f"  Language: {context.get('language', 'unknown')}")
                print(f"  Complexity: {context.get('complexity_score', 0)}")
                
                if context.get('docstring'):
                    print(f"  Docstring: {context['docstring'][:100]}...")
                
                if result.highlights:
                    print(f"  Code preview:")
                    for highlight in result.highlights[:2]:
                        print(f"    {highlight[:200]}...")
                print()
        
        return 0
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return 1
    finally:
        if 'searcher' in locals() and hasattr(searcher, 'close'):
            searcher.close()
        indexer.close()


def run_clustering_command(args):
    """运行聚类分析命令"""
    logger.info("Running clustering analysis...")
    
    # 创建配置
    sync_config = setup_sync_config(args)
    
    # 创建索引器
    indexer = SyncSemanticIndexer(sync_config)
    
    try:
        # 加载索引
        if args.load_index and os.path.exists(args.load_index):
            logger.info(f"Loading index from {args.load_index}")
            indexer.load_index(args.load_index)
        else:
            logger.error("No index file specified or index file not found")
            return 1
        
        # 创建高级搜索器
        searcher = create_advanced_searcher(indexer)
        
        # 执行聚类分析
        if args.analysis_type == "clustering":
            logger.info(f"Performing function clustering with {args.n_clusters} clusters...")
            clusters = searcher.cluster_similar_functions(n_clusters=args.n_clusters)
            
            if args.output_format == "json":
                output_data = {}
                for cluster_id, functions in clusters.items():
                    output_data[f"cluster_{cluster_id}"] = []
                    for func in functions:
                        context = func.context or {}
                        output_data[f"cluster_{cluster_id}"].append({
                            "function_name": context.get('function_name', 'Unknown'),
                            "file_path": context.get('file_path', 'Unknown'),
                            "complexity_score": context.get('complexity_score', 0),
                            "function_type": context.get('function_type', 'unknown')
                        })
                print(json.dumps(output_data, indent=2, ensure_ascii=False))
            else:
                print(f"Function Clustering Results ({len(clusters)} clusters)")
                print("=" * 80)
                for cluster_id, functions in clusters.items():
                    print(f"Cluster {cluster_id}: {len(functions)} functions")
                    for i, func in enumerate(functions[:args.max_functions_per_cluster], 1):
                        context = func.context or {}
                        func_name = context.get('function_name', 'Unknown')
                        file_name = os.path.basename(context.get('file_path', 'Unknown'))
                        print(f"  {i}. {func_name} ({file_name})")
                    if len(functions) > args.max_functions_per_cluster:
                        print(f"  ... and {len(functions) - args.max_functions_per_cluster} more")
                    print()
        
        elif args.analysis_type == "similarity":
            logger.info("Performing similarity analysis...")
            # 获取一个函数作为示例
            embeddings_list = list(indexer.embedding_manager.embeddings_index.values())
            if not embeddings_list:
                logger.error("No functions found in index")
                return 1
            
            example_function = embeddings_list[0]
            recommendations = searcher.get_recommendations(example_function, top_k=args.top_k)
            
            context = example_function.metadata
            func_name = context.get('name', 'Unknown')
            
            if args.output_format == "json":
                output_data = {
                    "example_function": {
                        "name": func_name,
                        "file_path": context.get('filepath', 'Unknown'),
                        "complexity_score": context.get('complexity_score', 0)
                    },
                    "similar_functions": []
                }
                for rec in recommendations:
                    rec_context = rec.context or {}
                    output_data["similar_functions"].append({
                        "function_name": rec_context.get('function_name', 'Unknown'),
                        "file_path": rec_context.get('file_path', 'Unknown'),
                        "similarity": rec.similarity,
                        "complexity_score": rec_context.get('complexity_score', 0)
                    })
                print(json.dumps(output_data, indent=2, ensure_ascii=False))
            else:
                print(f"Similarity Analysis for: {func_name}")
                print("=" * 80)
                print(f"Example function: {func_name} ({context.get('filepath', 'Unknown')})")
                print(f"Similar functions ({len(recommendations)} found):")
                for i, rec in enumerate(recommendations, 1):
                    rec_context = rec.context or {}
                    rec_name = rec_context.get('function_name', 'Unknown')
                    rec_file = os.path.basename(rec_context.get('file_path', 'Unknown'))
                    print(f"  {i}. {rec_name} ({rec_file}) - similarity: {rec.similarity:.4f}")
                print()
        
        elif args.analysis_type == "duplicates":
            logger.info("Performing duplicate code detection...")
            duplicates = searcher.find_duplicate_code(threshold=args.duplicate_threshold)
            
            if args.output_format == "json":
                output_data = {}
                for i, group in enumerate(duplicates):
                    output_data[f"duplicate_group_{i}"] = []
                    for func in group:
                        context = func.context or {}
                        output_data[f"duplicate_group_{i}"].append({
                            "function_name": context.get('function_name', 'Unknown'),
                            "file_path": context.get('file_path', 'Unknown'),
                            "similarity": func.similarity,
                            "complexity_score": context.get('complexity_score', 0)
                        })
                print(json.dumps(output_data, indent=2, ensure_ascii=False))
            else:
                print(f"Duplicate Code Detection Results ({len(duplicates)} groups)")
                print("=" * 80)
                if duplicates:
                    for i, group in enumerate(duplicates, 1):
                        print(f"Duplicate Group {i}: {len(group)} functions")
                        for func in group:
                            context = func.context or {}
                            func_name = context.get('function_name', 'Unknown')
                            file_name = os.path.basename(context.get('file_path', 'Unknown'))
                            print(f"  - {func_name} ({file_name}) - similarity: {func.similarity:.4f}")
                        print()
                else:
                    print("No significant duplicates found (good code quality!)")
        
        elif args.analysis_type == "complexity":
            logger.info("Performing complexity analysis...")
            complexity_analysis = searcher.get_code_complexity_analysis()
            
            if args.output_format == "json":
                print(json.dumps(complexity_analysis, indent=2, ensure_ascii=False))
            else:
                print("Code Complexity Analysis")
                print("=" * 80)
                print(f"Total functions: {complexity_analysis.get('total_functions', 0)}")
                print(f"Average complexity: {complexity_analysis.get('average_complexity', 0):.2f}")
                print(f"Max complexity: {complexity_analysis.get('max_complexity', 0)}")
                print(f"Min complexity: {complexity_analysis.get('min_complexity', 0)}")
                print(f"Std complexity: {complexity_analysis.get('std_complexity', 0):.2f}")
                
                # 显示复杂度分布
                distribution = complexity_analysis.get('complexity_distribution', {})
                print("\nComplexity distribution:")
                print(f"  Low (0-2): {distribution.get('low', 0)} functions")
                print(f"  Medium (2-5): {distribution.get('medium', 0)} functions")
                print(f"  High (5-10): {distribution.get('high', 0)} functions")
                print(f"  Very High (10+): {distribution.get('very_high', 0)} functions")
                
                # 显示函数类型分布
                func_types = complexity_analysis.get('function_type_distribution', {})
                if func_types:
                    print("\nFunction types:")
                    for func_type, count in func_types.items():
                        print(f"  {func_type}: {count} functions")
        
        return 0
        
    except Exception as e:
        logger.error(f"Clustering analysis failed: {e}")
        return 1
    finally:
        indexer.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Semantic Code Search - Advanced code search using vector embeddings and graph databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index a repository
  python main.py index --repo-path /path/to/repo --save-index ./index.json
  
  # Run CLI interface
  python main.py cli --repo-path /path/to/repo --load-index ./index.json
  
  # Search from command line
  python main.py search --query "find element in array" --load-index ./index.json
  
  # Enable hybrid search with Neo4j
  python main.py cli --repo-path /path/to/repo --load-index ./index.json --enable-hybrid-search
  
  # Perform function clustering analysis
  python main.py cluster --analysis-type clustering --load-index ./index.json --repo-path /path/to/repo
  
  # Find similar functions
  python main.py cluster --analysis-type similarity --load-index ./index.json --repo-path /path/to/repo
  
  # Detect duplicate code
  python main.py cluster --analysis-type duplicates --load-index ./index.json --repo-path /path/to/repo
  
  # Analyze code complexity
  python main.py cluster --analysis-type complexity --load-index ./index.json --repo-path /path/to/repo
        """
    )
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # 索引命令
    index_parser = subparsers.add_parser('index', help='Index a repository')
    index_parser.add_argument('--repo-path', required=True, help='Path to repository to index')
    index_parser.add_argument('--index-path', default='./semantic_index', help='Path to store index')
    index_parser.add_argument('--save-index', help='Save index to file after indexing')
    index_parser.add_argument('--model', default='all-MiniLM-L6-v2', help='Embedding model name')
    index_parser.add_argument('--model-type', default='sentence_transformers',
                             choices=['sentence_transformers', 'openai', 'transformers', 'tfidf'],
                             help='Embedding model type')
    index_parser.add_argument('--openai-api-key', help='OpenAI API key (required for OpenAI model)')
    index_parser.add_argument('--include-extensions', help='Comma-separated list of file extensions to include')
    index_parser.add_argument('--exclude-patterns', help='Comma-separated list of patterns to exclude')
    index_parser.add_argument('--max-file-size', type=int, help='Maximum file size in bytes')
    
    # CLI命令
    cli_parser = subparsers.add_parser('cli', help='Start CLI interface')
    cli_parser.add_argument('--repo-path', required=True, help='Path to repository')
    cli_parser.add_argument('--index-path', default='./semantic_index', help='Path to index')
    cli_parser.add_argument('--load-index', help='Load existing index from file')
    cli_parser.add_argument('--save-index', help='Save index to file after indexing')
    cli_parser.add_argument('--model', default='all-MiniLM-L6-v2', help='Embedding model name')
    cli_parser.add_argument('--model-type', default='sentence_transformers',
                              choices=['sentence_transformers', 'openai', 'transformers', 'tfidf'],
                              help='Embedding model type')
    cli_parser.add_argument('--openai-api-key', help='OpenAI API key (required for OpenAI model)')
    cli_parser.add_argument('--enable-hybrid-search', action='store_true', help='Enable hybrid search with Neo4j')
    cli_parser.add_argument('--neo4j-uri', default='bolt://localhost:7687', help='Neo4j URI')
    cli_parser.add_argument('--neo4j-user', default='neo4j', help='Neo4j username')
    cli_parser.add_argument('--neo4j-password', default='90879449Drq', help='Neo4j password')
    cli_parser.add_argument('--neo4j-database', help='Neo4j database name')
    cli_parser.add_argument('--vector-weight', type=float, help='Vector search weight (0.0-1.0)')
    cli_parser.add_argument('--graph-weight', type=float, help='Graph search weight (0.0-1.0)')
    
    # 搜索命令
    search_parser = subparsers.add_parser('search', help='Search from command line')
    search_parser.add_argument('--query', required=True, help='Search query')
    search_parser.add_argument('--load-index', required=True, help='Load index from file')
    search_parser.add_argument('--repo-path', required=True, help='Path to repository')
    search_parser.add_argument('--index-path', default='./semantic_index', help='Path to index')
    search_parser.add_argument('--model', default='all-MiniLM-L6-v2', help='Embedding model name')
    search_parser.add_argument('--model-type', default='sentence_transformers',
                              choices=['sentence_transformers', 'openai', 'transformers', 'tfidf'],
                              help='Embedding model type')
    search_parser.add_argument('--openai-api-key', help='OpenAI API key (required for OpenAI model)')
    search_parser.add_argument('--query-type', default='semantic', choices=['semantic', 'keyword', 'hybrid'],
                              help='Search type')
    search_parser.add_argument('--top-k', type=int, default=10, help='Number of results to return')
    search_parser.add_argument('--threshold', type=float, default=0.0, help='Minimum similarity threshold')
    search_parser.add_argument('--filters', help='JSON string of filters to apply')
    search_parser.add_argument('--output-format', choices=['text', 'json'], default='text', help='Output format')
    search_parser.add_argument('--enable-hybrid-search', action='store_true', help='Enable hybrid search with Neo4j')
    search_parser.add_argument('--neo4j-uri', default='bolt://localhost:7687', help='Neo4j URI')
    search_parser.add_argument('--neo4j-user', default='neo4j', help='Neo4j username')
    search_parser.add_argument('--neo4j-password', default='90879449Drq', help='Neo4j password')
    search_parser.add_argument('--neo4j-database', help='Neo4j database name')
    search_parser.add_argument('--vector-weight', type=float, help='Vector search weight (0.0-1.0)')
    search_parser.add_argument('--graph-weight', type=float, help='Graph search weight (0.0-1.0)')
    
    # 聚类分析命令
    cluster_parser = subparsers.add_parser('cluster', help='Perform clustering and similarity analysis')
    cluster_parser.add_argument('--analysis-type', required=True, 
                               choices=['clustering', 'similarity', 'duplicates', 'complexity'],
                               help='Type of analysis to perform')
    cluster_parser.add_argument('--load-index', required=True, help='Load index from file')
    cluster_parser.add_argument('--repo-path', required=True, help='Path to repository')
    cluster_parser.add_argument('--index-path', default='./semantic_index', help='Path to index')
    cluster_parser.add_argument('--model', default='all-MiniLM-L6-v2', help='Embedding model name')
    cluster_parser.add_argument('--model-type', default='sentence_transformers',
                               choices=['sentence_transformers', 'openai', 'transformers', 'tfidf'],
                               help='Embedding model type')
    cluster_parser.add_argument('--openai-api-key', help='OpenAI API key (required for OpenAI model)')
    cluster_parser.add_argument('--output-format', choices=['text', 'json'], default='text', help='Output format')
    
    # 聚类特定参数
    cluster_parser.add_argument('--n-clusters', type=int, default=10, 
                               help='Number of clusters for clustering analysis')
    cluster_parser.add_argument('--max-functions-per-cluster', type=int, default=5,
                               help='Maximum number of functions to show per cluster')
    cluster_parser.add_argument('--top-k', type=int, default=5,
                               help='Number of similar functions to return for similarity analysis')
    cluster_parser.add_argument('--duplicate-threshold', type=float, default=0.9,
                               help='Similarity threshold for duplicate detection (0.0-1.0)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'index':
            index_repository(args)
        elif args.command == 'cli':
            run_search_server(args)
        elif args.command == 'search':
            return run_search_command(args)
        elif args.command == 'cluster':
            return run_clustering_command(args)
        else:
            parser.print_help()
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        if hasattr(args, 'debug') and args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
