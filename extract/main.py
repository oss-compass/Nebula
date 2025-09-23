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
    """
    处理整个仓库
    
    Args:
        repo_path: 仓库路径
        repo_name: 仓库名称
        filter_large_functions: 是否过滤大函数
        
    Returns:
        包含分析结果的字典
    """
    print(f"开始处理仓库: {repo_name}")
    print(f"仓库路径: {repo_path}")
    
    # 收集所有函数名用于调用图分析
    all_function_names = set()
    temp_functions = []
    file_count = 0
    
    # 第一遍：收集所有函数名
    for file_path in repo_path.rglob("*.py"):
        if file_path.is_file():
            file_count += 1
            result = process_file(file_path, repo_path)
            temp_functions.extend(result["functions"])
            
            # 收集函数名
            for func in result["functions"]:
                func_name = func["basic_info"]["function_name"]
                if func_name:
                    all_function_names.add(func_name)
    
    print(f"  收集了{len(all_function_names)} 个函数名")
    
    # 第二遍：使用完整的函数名集合进行详细分析
    all_classes = []
    all_functions = []
    
    for file_path in repo_path.rglob("*.py"):
        if file_path.is_file():
            result = process_file(file_path, repo_path, all_function_names)
            all_classes.extend(result["classes"])
            all_functions.extend(result["functions"])
            
            # 打印函数信息，不打印类信息
            if result["functions"]:
                print(f"  在{file_path.relative_to(repo_path)} 中找到{len(result['functions'])} 个函数")
    
    total_functions_before_filter = len(all_functions)
    
    # 如果启用大函数过滤，则过滤函数
    if filter_large_functions:
        filtered_functions = []
        for func in all_functions:
            lines_of_code = func["complexity"]["semantic_complexity"]["lines_of_code"]
            if lines_of_code > LARGE_FUNCTION_THRESHOLD:
                filtered_functions.append(func)
        
        all_functions = filtered_functions
        print(f"\n 启用大函数过滤：从{total_functions_before_filter} 个函数中筛选出 {len(all_functions)} 个代码行数>{LARGE_FUNCTION_THRESHOLD}的函数")
    
    # 构建调用图
    print("构建API调用图...")
    call_graph = build_api_call_graph(all_functions, all_classes)
    
    # 分析API依赖关系
    print("分析API依赖关系...")
    api_dependencies = analyze_api_dependencies(call_graph, all_functions)
    
    # 分析传递调用
    print("分析传递调用...")
    transitive_calls = analyze_transitive_calls(call_graph, all_functions)
    
    # 计算函数重要性
    print("计算函数重要性...")
    importance_scores = calculate_function_importance(
        all_functions, 
        call_graph, 
        api_dependencies
    )
    
    # 更新函数重要性分数
    all_functions = update_function_importance(all_functions, importance_scores)
    
    # 分析API指标
    print("分析API指标...")
    api_metrics = analyze_api_metrics(all_functions, call_graph, api_dependencies)
    
    # 构建最终结果
    result = {
        "repo_name": repo_name,
        "repo_path": str(repo_path),
        "analysis_time": datetime.now(timezone.utc).isoformat(),
        "file_count": file_count,
        "total_functions": len(all_functions),
        "total_classes": len(all_classes),
        "functions": all_functions,
        "classes": all_classes,
        "call_graph": call_graph,
        "api_dependencies": api_dependencies,
        "transitive_calls": transitive_calls,
        "importance_scores": importance_scores,
        "api_metrics": api_metrics,
        "filter_large_functions": filter_large_functions,
        "large_function_threshold": LARGE_FUNCTION_THRESHOLD
    }
    
    print(f"分析完成！")
    print(f"  文件数量: {file_count}")
    print(f"  函数数量: {len(all_functions)}")
    print(f"  类数量: {len(all_classes)}")
    print(f"  API依赖关系: {len(api_dependencies)}")
    print(f"  传递调用: {len(transitive_calls)}")
    
    return result


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="代码分析工具")
    parser.add_argument("--repo-path", required=True, help="仓库路径")
    parser.add_argument("--repo-name", help="仓库名称（默认为路径名）")
    parser.add_argument("--output", help="输出文件路径")
    parser.add_argument("--filter-large-functions", action="store_true", 
                       help="过滤大函数")
    
    args = parser.parse_args()
    
    repo_path = Path(args.repo_path)
    if not repo_path.exists():
        print(f"错误：路径不存在: {repo_path}")
        sys.exit(1)
    
    repo_name = args.repo_name or repo_path.name
    
    try:
        # 处理仓库
        result = process_repo(
            repo_path=repo_path,
            repo_name=repo_name,
            filter_large_functions=args.filter_large_functions
        )
        
        # 输出结果
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2, cls=SetEncoder)
            
            print(f"结果已保存到: {output_path}")
        else:
            # 输出到标准输出
            print(json.dumps(result, ensure_ascii=False, indent=2, cls=SetEncoder))
            
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
