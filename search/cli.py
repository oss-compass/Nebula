import argparse
import json
import sys
import os
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search import Neo4jSearchEngine

def cmd_search(args):
    """搜索命令"""
    with Neo4jSearchEngine() as engine:
        results = engine.search_by_natural_language(
            args.query,
            limit=args.limit,
            search_type=args.search_type,
            similarity_threshold=args.threshold
        )
        
        print(f"找到 {len(results)} 个结�?")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['name']}")
            print(f"   文件: {result.get('filepath', 'N/A')}")
            print(f"   描述: {result.get('docstring_description', 'N/A')[:100]}...")
            if result.get('similarity_score'):
                print(f"   相似�? {result['similarity_score']:.3f}")
        
        if args.output:
            engine.export_results(results, args.output, args.format)

def cmd_callers(args):
    """查找调用者命�?""
    with Neo4jSearchEngine() as engine:
        result = engine.find_api_callers(
            args.function,
            max_depth=args.depth,
            include_external=args.include_external
        )
        
        print(f"函数 '{args.function}' 的调用�?")
        print(f"总调用者数: {result['total_callers']}")
        print(f"直接调用�? {len(result['direct_callers'])}")
        print(f"间接调用�? {len(result['indirect_callers'])}")
        
        if result['direct_callers']:
            print("\n直接调用�?")
            for caller in result['direct_callers'][:args.limit]:
                print(f"  - {caller['caller_name']} ({caller['caller_file']})")
        
        if args.output:
            engine.export_results(result, args.output, args.format)

def cmd_callees(args):
    """查找被调用者命�?""
    with Neo4jSearchEngine() as engine:
        result = engine.find_api_callees(
            args.function,
            max_depth=args.depth,
            include_external=args.include_external
        )
        
        print(f"函数 '{args.function}' 的被调用�?")
        print(f"总被调用者数: {result['total_callees']}")
        print(f"直接被调用�? {len(result['direct_callees'])}")
        print(f"间接被调用�? {len(result['indirect_callees'])}")
        
        if result['direct_callees']:
            print("\n直接被调用�?")
            for callee in result['direct_callees'][:args.limit]:
                print(f"  - {callee['callee_name']} ({callee['callee_file']})")
        
        if args.output:
            engine.export_results(result, args.output, args.format)

def cmd_dependencies(args):
    """依赖分析命令"""
    with Neo4jSearchEngine() as engine:
        result = engine.get_dependency_list(
            args.function,
            include_transitive=args.include_transitive,
            max_depth=args.depth
        )
        
        print(f"函数 '{args.function}' 的依赖关�?")
        print(f"总依赖数: {result['total_dependencies']}")
        print(f"直接依赖: {len(result['direct_dependencies'])}")
        print(f"传递依�? {len(result['transitive_dependencies'])}")
        print(f"外部依赖: {len(result['external_dependencies'])}")
        
        if result['direct_dependencies']:
            print("\n直接依赖:")
            for dep in result['direct_dependencies'][:args.limit]:
                print(f"  - {dep['dep_name']} ({dep['dep_file']})")
        
        if args.output:
            engine.export_results(result, args.output, args.format)

def cmd_centrality(args):
    """中心性分析命�?""
    with Neo4jSearchEngine() as engine:
        result = engine.calculate_centrality(
            algorithm=args.algorithm,
            top_k=args.top_k,
            include_weights=args.include_weights
        )
        
        print(f"{args.algorithm.upper()} 中心性分�?")
        print(f"分析节点�? {result['total_nodes']}")
        print(f"总边�? {result['total_edges']}")
        
        print(f"\n�?{args.top_k} 个最重要的函�?")
        for item in result['results'][:args.limit]:
            print(f"  {item['rank']}. {item['node_name']} (分数: {item['centrality_score']:.4f})")
        
        if args.output:
            engine.export_results(result, args.output, args.format)

def cmd_communities(args):
    """社区发现命令"""
    with Neo4jSearchEngine() as engine:
        result = engine.find_communities(
            algorithm=args.algorithm,
            min_community_size=args.min_size,
            resolution=args.resolution
        )
        
        print(f"{args.algorithm.upper()} 社区发现:")
        print(f"发现社区�? {result['total_communities']}")
        print(f"总节点数: {result['total_nodes']}")
        
        print(f"\n�?{args.limit} 个社�?")
        for community in result['communities'][:args.limit]:
            print(f"  社区 {community['community_id']}: {community['size']} 个函�?)
            for node in community['nodes'][:3]:
                print(f"    - {node['node_name']}")
            if len(community['nodes']) > 3:
                print(f"    ... 还有 {len(community['nodes']) - 3} 个函�?)
        
        if args.output:
            engine.export_results(result, args.output, args.format)

def cmd_analyze(args):
    """函数重要性分析命�?""
    with Neo4jSearchEngine() as engine:
        result = engine.analyze_function_importance(
            args.function,
            include_centrality=args.include_centrality,
            include_community=args.include_community,
            include_dependencies=args.include_dependencies
        )
        
        if "error" in result:
            print(f"分析失败: {result['error']}")
            return
        
        print(f"函数 '{args.function}' 的重要性分�?")
        print(f"分析组件: {', '.join(result.get('analysis_components', []))}")
        
        # 中心性分�?
        if 'centrality_analysis' in result:
            centrality = result['centrality_analysis']
            if 'function_rank' in centrality:
                print(f"中心性排�? {centrality['function_rank']}/{centrality['total_functions']}")
        
        # 社区分析
        if 'community_analysis' in result:
            community = result['community_analysis']
            if 'function_community' in community and community['function_community']:
                func_community = community['function_community']
                print(f"所属社�? 社区 {func_community['community_id']} (大小: {func_community['size']})")
        
        # 依赖分析
        if 'dependency_analysis' in result:
            deps = result['dependency_analysis']
            print(f"总依赖数: {deps['total_dependencies']}")
        
        if args.output:
            engine.export_results(result, args.output, args.format)

def cmd_info(args):
    """数据库信息命�?""
    with Neo4jSearchEngine() as engine:
        info = engine.get_database_info()
        stats = engine.get_graph_statistics()
        
        print("数据库信�?")
        print(f"  函数�? {info.get('total_functions', 0)}")
        print(f"  文件�? {info.get('total_files', 0)}")
        print(f"  仓库�? {info.get('total_repositories', 0)}")
        print(f"  关系�? {info.get('total_relationships', 0)}")
        
        basic_stats = stats.get('basic_statistics', {})
        if basic_stats:
            print(f"\n图统�?")
            print(f"  节点�? {basic_stats.get('total_nodes', 0)}")
            print(f"  边数: {basic_stats.get('total_edges', 0)}")
            print(f"  平均度数: {basic_stats.get('avg_degree', 0):.2f}")

def main():
    """主函�?""
    parser = argparse.ArgumentParser(description="Neo4j搜索命令行工�?)
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 通用参数
    def add_common_args(parser):
        parser.add_argument('--output', '-o', help='输出文件路径')
        parser.add_argument('--format', '-f', choices=['json', 'markdown'], default='json', help='输出格式')
        parser.add_argument('--limit', '-l', type=int, default=10, help='结果数量限制')
    
    # 搜索命令
    search_parser = subparsers.add_parser('search', help='搜索函数')
    search_parser.add_argument('query', help='搜索查询')
    search_parser.add_argument('--search-type', choices=['semantic', 'keyword', 'hybrid', 'ai'], default='hybrid', help='搜索类型')
    search_parser.add_argument('--threshold', type=float, help='相似度阈�?)
    add_common_args(search_parser)
    search_parser.set_defaults(func=cmd_search)
    
    # 调用者命�?
    callers_parser = subparsers.add_parser('callers', help='查找函数调用�?)
    callers_parser.add_argument('function', help='函数�?)
    callers_parser.add_argument('--depth', type=int, default=3, help='搜索深度')
    callers_parser.add_argument('--include-external', action='store_true', help='包含外部调用')
    add_common_args(callers_parser)
    callers_parser.set_defaults(func=cmd_callers)
    
    # 被调用者命�?
    callees_parser = subparsers.add_parser('callees', help='查找函数被调用�?)
    callees_parser.add_argument('function', help='函数�?)
    callees_parser.add_argument('--depth', type=int, default=3, help='搜索深度')
    callees_parser.add_argument('--include-external', action='store_true', help='包含外部调用')
    add_common_args(callees_parser)
    callees_parser.set_defaults(func=cmd_callees)
    
    # 依赖命令
    deps_parser = subparsers.add_parser('dependencies', help='分析函数依赖')
    deps_parser.add_argument('function', help='函数�?)
    deps_parser.add_argument('--depth', type=int, default=5, help='搜索深度')
    deps_parser.add_argument('--include-transitive', action='store_true', help='包含传递依�?)
    add_common_args(deps_parser)
    deps_parser.set_defaults(func=cmd_dependencies)
    
    # 中心性命�?
    centrality_parser = subparsers.add_parser('centrality', help='中心性分�?)
    centrality_parser.add_argument('--algorithm', choices=['pagerank', 'betweenness', 'closeness', 'eigenvector'], default='pagerank', help='中心性算�?)
    centrality_parser.add_argument('--top-k', type=int, default=20, help='返回前k个结�?)
    centrality_parser.add_argument('--include-weights', action='store_true', help='考虑边权�?)
    add_common_args(centrality_parser)
    centrality_parser.set_defaults(func=cmd_centrality)
    
    # 社区命令
    communities_parser = subparsers.add_parser('communities', help='社区发现')
    communities_parser.add_argument('--algorithm', choices=['louvain', 'leiden', 'label_propagation'], default='louvain', help='社区发现算法')
    communities_parser.add_argument('--min-size', type=int, default=2, help='最小社区大�?)
    communities_parser.add_argument('--resolution', type=float, default=1.0, help='分辨率参�?)
    add_common_args(communities_parser)
    communities_parser.set_defaults(func=cmd_communities)
    
    # 分析命令
    analyze_parser = subparsers.add_parser('analyze', help='函数重要性分�?)
    analyze_parser.add_argument('function', help='函数�?)
    analyze_parser.add_argument('--include-centrality', action='store_true', help='包含中心性分�?)
    analyze_parser.add_argument('--include-community', action='store_true', help='包含社区分析')
    analyze_parser.add_argument('--include-dependencies', action='store_true', help='包含依赖分析')
    add_common_args(analyze_parser)
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # 信息命令
    info_parser = subparsers.add_parser('info', help='显示数据库信�?)
    info_parser.set_defaults(func=cmd_info)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        args.func(args)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
