import logging
from typing import Dict, Any
from .main_interface import Neo4jSearchEngine

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_basic_queries():
    """基础查询示例"""
    print("=" * 50)
    print("基础查询示例")
    print("=" * 50)
    
    with Neo4jSearchEngine() as engine:
        # 1. 查找API调用�?
        print("\n1. 查找API调用�?")
        callers = engine.find_api_callers("process_data", max_depth=2)
        print(f"找到 {callers['total_callers']} 个调用�?)
        for caller in callers['direct_callers'][:3]:
            print(f"  - {caller['caller_name']} ({caller['caller_file']})")
        
        # 2. 查找API被调用�?
        print("\n2. 查找API被调用�?")
        callees = engine.find_api_callees("process_data", max_depth=2)
        print(f"找到 {callees['total_callees']} 个被调用�?)
        for callee in callees['direct_callees'][:3]:
            print(f"  - {callee['callee_name']} ({callee['callee_file']})")
        
        # 3. 获取依赖清单
        print("\n3. 获取依赖清单:")
        deps = engine.get_dependency_list("process_data", include_transitive=True, max_depth=3)
        print(f"总依赖数: {deps['total_dependencies']}")
        print(f"直接依赖: {len(deps['direct_dependencies'])}")
        print(f"传递依�? {len(deps['transitive_dependencies'])}")
        
        # 4. 根据函数名查�?
        print("\n4. 根据函数名查�?")
        functions = engine.find_function_by_name("process", exact_match=False)
        print(f"找到 {len(functions)} 个相关函�?)
        for func in functions[:3]:
            print(f"  - {func['name']} ({func['filepath']})")

def example_semantic_search():
    """语义搜索示例"""
    print("=" * 50)
    print("语义搜索示例")
    print("=" * 50)
    
    with Neo4jSearchEngine() as engine:
        # 1. 自然语言搜索
        print("\n1. 自然语言搜索:")
        results = engine.search_by_natural_language(
            "处理数据的函�?, 
            limit=5, 
            search_type="hybrid"
        )
        print(f"找到 {len(results)} 个结�?)
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['name']} - {result.get('similarity_score', 0):.3f}")
            if result.get('docstring_description'):
                print(f"     描述: {result['docstring_description'][:100]}...")
        
        # 2. 按复杂度搜索
        print("\n2. 按复杂度搜索:")
        complex_funcs = engine.search_by_complexity(
            complexity_level="complex",
            min_complexity=10,
            limit=5
        )
        print(f"找到 {len(complex_funcs)} 个复杂函�?)
        for func in complex_funcs[:3]:
            print(f"  - {func['name']} (复杂�? {func.get('cyclomatic_complexity', 'N/A')})")
        
        # 3. 查找相似函数
        print("\n3. 查找相似函数:")
        similar_funcs = engine.search_similar_functions("process_data", limit=5)
        print(f"找到 {len(similar_funcs)} 个相似函�?)
        for func in similar_funcs[:3]:
            print(f"  - {func['name']} (相似�? {func.get('similarity_score', 0):.3f})")

def example_graph_analysis():
    """图分析示�?""
    print("=" * 50)
    print("图分析示�?)
    print("=" * 50)
    
    with Neo4jSearchEngine() as engine:
        # 1. 中心性分�?
        print("\n1. 中心性分�?(PageRank):")
        centrality = engine.calculate_centrality("pagerank", top_k=10)
        print(f"分析 {centrality['total_nodes']} 个节�?)
        print("�?0个最重要的函�?")
        for result in centrality['results'][:5]:
            print(f"  {result['rank']}. {result['node_name']} (分数: {result['centrality_score']:.4f})")
        
        # 2. 社区发现
        print("\n2. 社区发现:")
        communities = engine.find_communities("louvain", min_community_size=3)
        print(f"发现 {communities['total_communities']} 个社�?)
        for i, community in enumerate(communities['communities'][:3], 1):
            print(f"  社区 {i}: {community['size']} 个函�?)
            for node in community['nodes'][:3]:
                print(f"    - {node['node_name']}")
        
        # 3. 相似度矩�?
        print("\n3. 相似度矩�?")
        function_names = ["process_data", "validate_input", "format_output"]
        similarity_matrix = engine.calculate_similarity_matrix(function_names, "structural")
        print("函数相似度矩�?")
        for i, func1 in enumerate(function_names):
            for j, func2 in enumerate(function_names):
                score = similarity_matrix['similarity_matrix'][i][j]
                print(f"  {func1} vs {func2}: {score:.3f}")
        
        # 4. 图统计信�?
        print("\n4. 图统计信�?")
        stats = engine.get_graph_statistics()
        basic_stats = stats.get('basic_statistics', {})
        print(f"总节点数: {basic_stats.get('total_nodes', 0)}")
        print(f"总边�? {basic_stats.get('total_edges', 0)}")
        print(f"平均度数: {basic_stats.get('avg_degree', 0):.2f}")

def example_comprehensive_analysis():
    """综合分析示例"""
    print("=" * 50)
    print("综合分析示例")
    print("=" * 50)
    
    with Neo4jSearchEngine() as engine:
        # 1. 函数重要性分�?
        print("\n1. 函数重要性分�?")
        importance = engine.analyze_function_importance(
            "process_data",
            include_centrality=True,
            include_community=True,
            include_dependencies=True
        )
        
        if "error" not in importance:
            print(f"分析函数: {importance['function_name']}")
            
            # 中心性排�?
            centrality = importance.get('centrality_analysis', {})
            if centrality and "function_rank" in centrality:
                print(f"中心性排�? {centrality['function_rank']}/{centrality['total_functions']}")
            
            # 社区信息
            community = importance.get('community_analysis', {})
            if community and "function_community" in community:
                func_community = community['function_community']
                if func_community:
                    print(f"所属社�? 社区 {func_community['community_id']} (大小: {func_community['size']})")
            
            # 依赖信息
            deps = importance.get('dependency_analysis', {})
            if deps:
                print(f"总依赖数: {deps['total_dependencies']}")
        else:
            print(f"分析失败: {importance['error']}")
        
        # 2. 综合搜索
        print("\n2. 综合搜索:")
        search_result = engine.comprehensive_search(
            "数据处理相关函数",
            search_type="hybrid",
            include_analysis=True,
            limit=5
        )
        
        print(f"搜索查询: {search_result['query']}")
        print(f"语义搜索结果: {search_result['total_semantic_results']} �?)
        
        if search_result.get('importance_analysis'):
            print("重要性分析结�?")
            for analysis in search_result['importance_analysis']:
                func_name = analysis['function_name']
                components = analysis.get('analysis_components', [])
                print(f"  - {func_name}: {', '.join(components)}")

def example_export_results():
    """结果导出示例"""
    print("=" * 50)
    print("结果导出示例")
    print("=" * 50)
    
    with Neo4jSearchEngine() as engine:
        # 执行搜索
        results = engine.search_by_natural_language("数据处理", limit=5)
        
        # 导出为JSON
        engine.export_results(results, "search_results.json", "json")
        print("结果已导出为JSON格式")
        
        # 导出为Markdown
        engine.export_results(results, "search_results.md", "markdown")
        print("结果已导出为Markdown格式")

def main():
    """主函�?- 运行所有示�?""
    print("Neo4j搜索功能示例")
    print("=" * 60)
    
    try:
        # 基础查询示例
        example_basic_queries()
        
        # 语义搜索示例
        example_semantic_search()
        
        # 图分析示�?
        example_graph_analysis()
        
        # 综合分析示例
        example_comprehensive_analysis()
        
        # 结果导出示例
        example_export_results()
        
        print("\n" + "=" * 60)
        print("所有示例执行完成！")
        
    except Exception as e:
        logger.error(f"示例执行失败: {e}")
        print(f"错误: {e}")

if __name__ == "__main__":
    main()
