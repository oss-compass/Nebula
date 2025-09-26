
import sys
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional

from .utils import clone_repo, SetEncoder
from .info_extractor import process_file
from .call_graph_analyzer import (
    build_api_call_graph, 
    analyze_api_dependencies, 
    analyze_transitive_calls,
    analyze_api_metrics
)
from .importance_calculator import (
    calculate_function_importance,
    update_function_importance
)
from .config import LARGE_FUNCTION_THRESHOLD
# from .neo4j_writer import write_to_neo4j  # 已禁用直接Neo4j写入功能


def process_repo(repo_path: Path, repo_name: str, filter_large_functions: bool = False) -> dict:
    print(f"\n 开始分析仓库: {repo_name}")
    
    all_classes = []
    all_functions = []
    file_count = 0
    
    # 第一遍：收集所有函数名
    print("  第一遍扫描：收集所有函数名...")
    all_function_names = set()
    temp_functions = []
    
    for file_path in repo_path.rglob("*"):
        if not file_path.is_file():
            continue
        
        file_count += 1
        result = process_file(file_path, repo_path)
        temp_functions.extend(result["functions"])
        
        # 收集函数名
        for func in result["functions"]:
            func_name = func["basic_info"]["function_name"]
            if func_name:
                all_function_names.add(func_name)
    
    print(f"  收集到 {len(all_function_names)} 个函数名")
    
    # 第二遍：使用完整的函数名集合重新处理
    print("  第二遍扫描：分析函数调用关系...")
    all_classes = []
    all_functions = []
    
    for file_path in repo_path.rglob("*"):
        if not file_path.is_file():
            continue
        
        result = process_file(file_path, repo_path, all_function_names)
        all_classes.extend(result["classes"])
        all_functions.extend(result["functions"])
        
        # 只打印函数信息，不打印类信息
        if result["functions"]:
            print(f"  在 {file_path.relative_to(repo_path)} 中找到 {len(result['functions'])} 个函数")
    
    total_functions_before_filter = len(all_functions)
    
    # 如果启用大函数过滤，则过滤函数
    if filter_large_functions:
        filtered_functions = []
        for func in all_functions:
            lines_of_code = func["complexity"]["semantic_complexity"]["lines_of_code"]
            if lines_of_code > LARGE_FUNCTION_THRESHOLD:
                filtered_functions.append(func)
        
        all_functions = filtered_functions
        print(f"\n 启用大函数过滤：从 {total_functions_before_filter} 个函数中筛选出 {len(all_functions)} 个代码行数>{LARGE_FUNCTION_THRESHOLD}的函数")
    
    print(f"\n 完成分析 {file_count} 个文件，共找到 {len(all_functions)} 个函数")

    # 提取所有函数名
    function_names = [func["basic_info"]["function_name"] for func in all_functions]

    # 构建API调用关系图
    call_graph = build_api_call_graph(all_functions, repo_path)
    
    # 分析API依赖关系
    dependencies = analyze_api_dependencies(all_functions, repo_path)
    
    # 分析API指标
    metrics = analyze_api_metrics(all_functions, call_graph)
    
    # 分析传递调用逻辑
    transitive_calls_result = analyze_transitive_calls(all_functions, call_graph)
    
    # 计算函数重要度
    print("  计算函数重要度...")
    direct_calls = transitive_calls_result.get("direct_calls", {})
    transitive_calls = transitive_calls_result.get("transitive_calls", {})
    transitive_paths = transitive_calls_result.get("transitive_paths", {})
    
    # 创建函数映射
    function_map = {}
    for func in all_functions:
        func_name = func["basic_info"]["function_name"]
        if func_name:
            # 确保func_name是字符串类型
            if isinstance(func_name, bytes):
                func_name = func_name.decode('utf-8')
            function_map[func_name] = func
    
    importance_scores = calculate_function_importance(
        function_map, direct_calls, transitive_calls, transitive_paths
    )
    
    # 更新每个函数的重要度信息
    update_function_importance(all_functions, importance_scores)
    
    # 将重要度信息添加到传递调用分析结果中
    transitive_calls_result["impact_analysis"]["importance_scores"] = importance_scores

    result = {
        "repository": repo_name,
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_functions": len(all_functions),
        "function_names": function_names,
        "functions": all_functions,
        "class_function_relationship": {
            "description": "类和函数的关联关系分析",
            "classes_with_methods": len([c for c in all_classes if c["methods"]]),
            "standalone_functions": len([f for f in all_functions if not f["context"]["parent_class"]])
        },
        # API调用关系分析
        "api_call_relationships": {
            "call_graph": call_graph,
            "dependencies": dependencies,
            "metrics": metrics
        },
        "transitive_calls": transitive_calls_result
    }
    
    # 如果启用了过滤，添加过滤信息
    if filter_large_functions:
        result["filter_info"] = {
            "filter_enabled": True,
            "filter_condition": f"lines_of_code > {LARGE_FUNCTION_THRESHOLD}",
            "total_functions_before_filter": total_functions_before_filter,
            "total_functions_after_filter": len(all_functions),
            "filtered_out_count": total_functions_before_filter - len(all_functions)
        }
    
    return result


def print_analysis_summary(result: dict) -> None:
    print(f"\n  分析结果:")
    print(f"  - 总函数数: {result['total_functions']}")
    print(f"  - 独立函数: {result['class_function_relationship']['standalone_functions']}")
    
    # 显示API调用关系分析结果
    api_relationships = result.get("api_call_relationships", {})
    if api_relationships:
        call_graph = api_relationships.get("call_graph", {})
        dependencies = api_relationships.get("dependencies", {})
        metrics = api_relationships.get("metrics", {})
        
        print(f"\n  API调用关系分析结果:")
        print(f"  - 调用图节点数: {call_graph.get('statistics', {}).get('total_nodes', 0)}")
        print(f"  - 内部API调用边数: {call_graph.get('statistics', {}).get('total_edges', 0)}")
        print(f"  - 孤立函数数: {call_graph.get('statistics', {}).get('isolated_functions', 0)}")
        
        # 统计外部调用
        total_external_calls = 0
        for func in result.get('functions', []):
            total_external_calls += len(func["context"]["function_calls"]["external_calls"])
        print(f"  - 外部函数调用总数: {total_external_calls}")
        
        # 显示被调用最多的函数
        most_called = call_graph.get('statistics', {}).get('most_called_functions', [])
        if most_called:
            print(f"  - 被调用最多的函数:")
            for i, func in enumerate(most_called[:5]):
                print(f"    {i+1}. {func['function_name']} (被调用 {func['call_count']} 次)")
        
        # 显示循环依赖
        circular_deps = dependencies.get('circular_dependencies', {})
        if any(circular_deps.values()):
            print(f"  - 发现循环依赖:")
            for level, cycles in circular_deps.items():
                if cycles:
                    print(f"    {level}: {len(cycles)} 个循环")
        
        # 显示最复杂的函数
        most_complex = metrics.get('overall_metrics', {}).get('most_complex_functions', [])
        if most_complex:
            print(f"  - 最复杂的函数:")
            for i, func in enumerate(most_complex[:5]):
                print(f"    {i+1}. {func['function_name']} (复杂度: {func['complexity']})")
    
    # 显示传递调用关系分析结果
    transitive_calls_result = result.get("transitive_calls", {})
    if transitive_calls_result:
        transitive_graph = transitive_calls_result.get("transitive_graph", {})
        impact_analysis = transitive_calls_result.get("impact_analysis", {})
        
        print(f"\n 传递调用关系分析结果:")
        print(f"  - 传递调用图节点数: {transitive_graph.get('statistics', {}).get('total_nodes', 0)}")
        print(f"  - 传递调用边数: {len(transitive_graph.get('edges', []))}")
        print(f"  - 最大传递调用深度: {transitive_graph.get('statistics', {}).get('max_transitive_depth', 0)}")
        print(f"  - 有传递调用的函数数: {transitive_graph.get('statistics', {}).get('functions_with_transitive_calls', 0)}")

        print(f"  - 传递调用最多的函数:")
        for i, func in enumerate(impact_analysis.get('most_transitive_functions', [])[:5]):
            print(f"    {i+1}. {func['function_name']} (传递调用 {func['transitive_call_count']} 次)")

        print(f"  - 最长传递调用链 (前10):")
        for i, chain in enumerate(impact_analysis.get('transitive_call_chains', [])[:10]):
            print(f"    {i+1}. {chain['description']} (长度: {chain['length']})")

        print(f"  - 高影响函数 (被很多函数间接调用的函数):")
        for i, func in enumerate(impact_analysis.get('high_impact_functions', [])[:10]):
            print(f"    {i+1}. {func['function_name']} (影响函数数: {func['impact_count']})")

        print(f"  - 孤立函数 (没有传递调用的函数):")
        for i, func_name in enumerate(impact_analysis.get('isolated_functions', [])[:10]):
            print(f"    {i+1}. {func_name}")

        print(f"  - 枢纽函数 (既有直接调用又有传递调用的函数):")
        for i, func in enumerate(impact_analysis.get('hub_functions', [])[:10]):
            print(f"    {i+1}. {func['function_name']} (总影响: {func['total_influence']})")

        print(f"  - 叶子函数 (只被调用，不调用其他函数的函数):")
        for i, func_name in enumerate(impact_analysis.get('leaf_functions', [])[:20]):
            print(f"    {i+1}. {func_name}")
        
        # 显示重要度分析结果
        importance_scores = impact_analysis.get('importance_scores', {})
        if importance_scores:
            print(f"\n  === 函数重要度分析结果 ===")
            
            # 显示重要度分布
            distribution = importance_scores.get('importance_distribution', {})
            print(f"  - 重要度分布:")
            for level, count in distribution.items():
                print(f"    {level}: {count} 个函数")
            
            # 显示前10名重要函数
            top_functions = importance_scores.get('top_functions', [])
            if top_functions:
                print(f"  - 前10名重要函数:")
                for i, func in enumerate(top_functions[:10]):
                    print(f"    {i+1}. {func['function_name']} (重要度: {func['total_score']}, 等级: {func['importance_level']})")
            
            # 显示Critical级别函数的详细信息
            critical_functions = []
            for func_name, score_info in importance_scores.get('scores', {}).items():
                if score_info['importance_level'] == 'Critical':
                    critical_functions.append((func_name, score_info))
            
            if critical_functions:
                print(f"  - Critical级别函数详细信息:")
                for i, (func_name, score_info) in enumerate(critical_functions[:5]):
                    breakdown = score_info['breakdown']
                    metrics = score_info['metrics']
                    print(f"    {i+1}. {func_name}:")
                    print(f"      总分数: {score_info['total_score']}")
                    
                    # 根据算法类型显示不同的breakdown信息
                    if 'direct_call_weight' in breakdown:
                        # 传统算法
                        print(f"      直接调用权重: {breakdown.get('direct_call_weight', 'N/A')}")
                        print(f"      传递调用权重: {breakdown.get('transitive_call_weight', 'N/A')}")
                        print(f"      深度影响: {breakdown.get('depth_impact', 'N/A')}")
                        print(f"      枢纽效应: {breakdown.get('hub_effect', 'N/A')}")
                        print(f"      被调用频率: {breakdown.get('incoming_call_frequency', 'N/A')}")
                        print(f"      特殊权重: {breakdown.get('special_weight', 'N/A')}")
                        print(f"      复杂度调整: {breakdown.get('complexity_adjustment', 'N/A')}")
                    elif 'pagerank_score' in breakdown and 'traditional_score' in breakdown:
                        # 混合算法
                        print(f"      PageRank分数: {breakdown.get('pagerank_score', 'N/A')}")
                        print(f"      传统算法分数: {breakdown.get('traditional_score', 'N/A')}")
                        print(f"      复杂度调整: {breakdown.get('complexity_adjustment', 'N/A')}")
                        print(f"      PageRank权重: {breakdown.get('pagerank_weight', 'N/A')}")
                        print(f"      传统权重: {breakdown.get('traditional_weight', 'N/A')}")
                        print(f"      复杂度权重: {breakdown.get('complexity_weight', 'N/A')}")
                    elif 'pagerank_score' in breakdown:
                        # PageRank算法
                        print(f"      PageRank分数: {breakdown.get('pagerank_score', 'N/A')}")
                        print(f"      复杂度调整: {breakdown.get('complexity_adjustment', 'N/A')}")
                    else:
                        # 未知算法类型，显示所有可用的breakdown信息
                        print(f"      算法类型: 未知")
                        for key, value in breakdown.items():
                            print(f"      {key}: {value}")
                    
                    # 显示metrics信息（如果存在）
                    if 'is_hub' in metrics:
                        print(f"      是否枢纽: {metrics['is_hub']}")
                    if 'is_leaf' in metrics:
                        print(f"      是否叶子: {metrics['is_leaf']}")
                    if 'is_coordinator' in metrics:
                        print(f"      是否协调者: {metrics['is_coordinator']}")
                    if 'is_foundation' in metrics:
                        print(f"      是否基础函数: {metrics['is_foundation']}")
                    if 'out_degree' in metrics:
                        print(f"      出度: {metrics['out_degree']}")
                    if 'in_degree' in metrics:
                        print(f"      入度: {metrics['in_degree']}")


def main():
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python -m extract.main <GitHub仓库链接> [选项]")
        print("  python -m extract.main <GitHub仓库链接> <稀疏检出路径> [选项]")
        print("\n选项说明:")
        print("  --filter-large-functions: 只保留代码行数>200的函数")
        print("  --neo4j: 直接写入Neo4j数据库（不生成JSON文件）")
        print("  --neo4j-uri <URI>: Neo4j数据库URI (默认: bolt://localhost:7687)")
        print("  --neo4j-user <用户名>: Neo4j用户名 (默认: neo4j)")
        print("  --neo4j-password <密码>: Neo4j密码 (默认: 90879449Drq)")
        print("  --neo4j-database <数据库>: Neo4j数据库名 (默认: neo4j)")
        print("  --clear-db: 清空Neo4j数据库后写入")
        print("  <稀疏检出路径>: 指定要检出的子目录路径（如: src/, lib/, core/）")
        print("\n示例:")
        print("  # 完整克隆 + 过滤大函数")
        print("  python -m extract.main https://github.com/user/repo --filter-large-functions")
        print("  # 稀疏检出src目录 + 过滤大函数")
        print("  python -m extract.main https://github.com/user/repo src --filter-large-functions")
        print("  # 直接写入Neo4j数据库")
        print("  python -m extract.main https://github.com/user/repo --neo4j")
        print("  # 写入指定Neo4j数据库")
        print("  python -m extract.main https://github.com/user/repo --neo4j --neo4j-uri bolt://localhost:7687 --neo4j-user myuser --neo4j-password mypass")
        print("  # 清空数据库后写入")
        print("  python -m extract.main https://github.com/user/repo --neo4j --clear-db")
        return

    repo_url = sys.argv[1]
    
    # 解析命令行参数
    filter_large_functions = "--filter-large-functions" in sys.argv
    # 已移除Neo4j直接写入功能，只保留JSON文件生成
    
    # 检查是否有稀疏检出路径
    sparse_checkout_paths = None
    for arg in sys.argv[2:]:
        if arg not in ["--filter-large-functions"]:
            # 分割多个路径（用逗号分隔）
            sparse_checkout_paths = [path.strip() for path in arg.split(',')]
            break
    
    if filter_large_functions:
        print("启用大函数过滤模式：只保留代码行数>200的函数")
    
    if sparse_checkout_paths:
        print(f"启用稀疏检出模式：只检出路径 {', '.join(sparse_checkout_paths)}")
    
    repo_path, repo_name = clone_repo(repo_url, sparse_checkout_paths)

    result = process_repo(repo_path, repo_name, filter_large_functions)

    # 保存结果为JSON文件
    output_dir = os.path.join("output", "extract_output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{repo_name}_api_extraction.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, cls=SetEncoder)
    print(f"\n  增强版API提取结果已保存到: {output_file}")
    
    # 打印分析摘要
    print_analysis_summary(result)
    
    # 打印过滤信息
    if filter_large_functions:
        print(f"  - 过滤条件: 只保留代码行数>{LARGE_FUNCTION_THRESHOLD}的函数")
        print(f"  - 过滤前总函数数: {result.get('filter_info', {}).get('total_functions_before_filter', 'N/A')}")
        print(f"  - 过滤后函数数: {result['total_functions']}")
    
    if sparse_checkout_paths:
        print(f"  - 稀疏检出路径: {', '.join(sparse_checkout_paths)}")
    
    print(f"\n  函数列表预览 (前10个):")
    function_names = result.get('function_names', [])
    for i, name in enumerate(function_names[:10]):
        print(f"    {i+1}. {name}")
    if len(function_names) > 10:
        print(f" ... 还有 {len(function_names) - 10} 个函数")
    
    print(f"\n  分析包含以下信息:")
    print(f"  (1) 基础信息: 原代码、代码定义位置、注释、函数名")
    print(f"  (2) 复杂度: 语义复杂度(lizard)、语法复杂度(Tree-sitter)、结构复杂度")
    print(f"  (3) 上下文信息: 所在类/模块、import列表、函数调用、返回值使用")
    print(f"  (4) API调用关系: 调用图、依赖关系、循环依赖检测、API指标分析")
    print(f"  (5) 传递调用关系: 传递调用图、影响范围分析、循环依赖检测")
    print(f"  (6) 函数重要度: 多维度重要度计算、重要度等级分类")


if __name__ == "__main__":
    main()

