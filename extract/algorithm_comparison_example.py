#!/usr/bin/env python3
"""
重要度算法比较示例

展示如何使用不同的重要度计算算法，包括：
1. 传统统计算法
2. PageRank算法  
3. 混合算法

使用方法：
python extract/algorithm_comparison_example.py
"""

import json
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extract.importance_calculator import (
    calculate_function_importance, 
    compare_algorithms,
    get_importance_summary
)


def load_sample_data():
    """加载示例数据"""
    # 这里应该加载实际的函数数据
    # 为了演示，我们创建一些模拟数据
    
    function_map = {
        "main": {
            "complexity": {
                "semantic_complexity": {
                    "complexity_score": 5.0,
                    "lines_of_code": 50,
                    "cyclomatic_complexity": 3
                }
            }
        },
        "process_data": {
            "complexity": {
                "semantic_complexity": {
                    "complexity_score": 8.0,
                    "lines_of_code": 100,
                    "cyclomatic_complexity": 5
                }
            }
        },
        "validate_input": {
            "complexity": {
                "semantic_complexity": {
                    "complexity_score": 3.0,
                    "lines_of_code": 30,
                    "cyclomatic_complexity": 2
                }
            }
        },
        "format_output": {
            "complexity": {
                "semantic_complexity": {
                    "complexity_score": 2.0,
                    "lines_of_code": 20,
                    "cyclomatic_complexity": 1
                }
            }
        },
        "helper_function": {
            "complexity": {
                "semantic_complexity": {
                    "complexity_score": 1.0,
                    "lines_of_code": 10,
                    "cyclomatic_complexity": 1
                }
            }
        }
    }
    
    # 模拟调用关系
    direct_calls = {
        "main": {"process_data", "validate_input"},
        "process_data": {"format_output", "helper_function"},
        "validate_input": {"helper_function"},
        "format_output": set(),
        "helper_function": set()
    }
    
    # 模拟传递调用
    transitive_calls = {
        "main": {"format_output", "helper_function"},
        "process_data": {"helper_function"},
        "validate_input": set(),
        "format_output": set(),
        "helper_function": set()
    }
    
    # 模拟传递路径
    transitive_paths = {
        "main": ["main", "process_data", "format_output"],
        "main": ["main", "validate_input", "helper_function"]
    }
    
    return function_map, direct_calls, transitive_calls, transitive_paths


def demonstrate_traditional_algorithm():
    """演示传统算法"""
    print("=" * 60)
    print("传统统计算法演示")
    print("=" * 60)
    
    function_map, direct_calls, transitive_calls, transitive_paths = load_sample_data()
    
    result = calculate_function_importance(
        function_map, direct_calls, transitive_calls, transitive_paths,
        algorithm="traditional"
    )
    
    print(f"算法类型: {result['algorithm']}")
    print(f"总函数数: {len(result['scores'])}")
    print("\n前5名重要函数:")
    for i, func in enumerate(result['top_functions'][:5], 1):
        print(f"{i}. {func['function_name']}: {func['total_score']:.2f} ({func['importance_level']})")
    
    print(f"\n重要度分布: {result['importance_distribution']}")
    return result


def demonstrate_pagerank_algorithm():
    """演示PageRank算法"""
    print("\n" + "=" * 60)
    print("PageRank算法演示")
    print("=" * 60)
    
    function_map, direct_calls, transitive_calls, transitive_paths = load_sample_data()
    
    result = calculate_function_importance(
        function_map, direct_calls, transitive_calls, transitive_paths,
        algorithm="pagerank"
    )
    
    print(f"算法类型: {result['algorithm']}")
    print(f"总函数数: {len(result['scores'])}")
    
    if 'graph_stats' in result:
        stats = result['graph_stats']
        print(f"图统计: {stats['total_nodes']}个节点, {stats['total_edges']}条边")
        print(f"图连通性: {'连通' if stats['is_connected'] else '不连通'}")
    
    print("\n前5名重要函数:")
    for i, func in enumerate(result['top_functions'][:5], 1):
        print(f"{i}. {func['function_name']}: {func['total_score']:.2f} ({func['importance_level']})")
    
    print(f"\n重要度分布: {result['importance_distribution']}")
    return result


def demonstrate_hybrid_algorithm():
    """演示混合算法"""
    print("\n" + "=" * 60)
    print("混合算法演示")
    print("=" * 60)
    
    function_map, direct_calls, transitive_calls, transitive_paths = load_sample_data()
    
    result = calculate_function_importance(
        function_map, direct_calls, transitive_calls, transitive_paths,
        algorithm="hybrid",
        pagerank_weight=0.4,
        traditional_weight=0.4,
        complexity_weight=0.2
    )
    
    print(f"算法类型: {result['algorithm']}")
    print(f"总函数数: {len(result['scores'])}")
    
    if 'algorithm_weights' in result:
        weights = result['algorithm_weights']
        print(f"算法权重: PageRank={weights['pagerank_weight']}, "
              f"传统={weights['traditional_weight']}, "
              f"复杂度={weights['complexity_weight']}")
    
    print("\n前5名重要函数:")
    for i, func in enumerate(result['top_functions'][:5], 1):
        print(f"{i}. {func['function_name']}: {func['total_score']:.2f} ({func['importance_level']})")
    
    print(f"\n重要度分布: {result['importance_distribution']}")
    return result


def demonstrate_algorithm_comparison():
    """演示算法比较"""
    print("\n" + "=" * 60)
    print("算法比较分析")
    print("=" * 60)
    
    function_map, direct_calls, transitive_calls, transitive_paths = load_sample_data()
    
    comparison = compare_algorithms(function_map, direct_calls, transitive_calls, transitive_paths)
    
    print("算法特征对比:")
    for algorithm, characteristics in comparison['algorithm_characteristics'].items():
        print(f"\n{algorithm.upper()}算法:")
        print(f"  描述: {characteristics['description']}")
        print(f"  优点: {', '.join(characteristics['strengths'])}")
        print(f"  缺点: {', '.join(characteristics['weaknesses'])}")
    
    print("\n前5名函数对比:")
    for algorithm in comparison['algorithms']:
        print(f"\n{algorithm.upper()}算法前5名:")
        for func in comparison['top_functions_comparison'][algorithm][:5]:
            print(f"  {func['rank']}. {func['function_name']}: {func['score']:.2f}")
    
    if comparison['correlation_analysis']:
        print("\n算法相关性分析:")
        for comparison_name, analysis in comparison['correlation_analysis'].items():
            print(f"  {comparison_name}: 相关性={analysis['correlation']}, "
                  f"共同函数数={analysis['common_functions']}")


def main():
    """主函数"""
    print("重要度算法比较演示")
    print("本演示将展示三种不同的重要度计算算法")
    
    try:
        # 演示各种算法
        traditional_result = demonstrate_traditional_algorithm()
        pagerank_result = demonstrate_pagerank_algorithm()
        hybrid_result = demonstrate_hybrid_algorithm()
        
        # 算法比较
        demonstrate_algorithm_comparison()
        
        print("\n" + "=" * 60)
        print("总结和建议")
        print("=" * 60)
        print("""
算法选择建议:

1. 传统算法 (traditional):
   - 适用于: 需要快速计算，重视代码复杂度的场景
   - 优点: 计算简单，结果可解释性强
   - 缺点: 忽略全局影响，可能遗漏重要的枢纽函数

2. PageRank算法 (pagerank):
   - 适用于: 需要识别关键枢纽函数，重视调用网络结构的场景
   - 优点: 全局视角，能识别传递性影响
   - 缺点: 忽略代码复杂度，对孤立函数不友好

3. 混合算法 (hybrid):
   - 适用于: 需要平衡多种因素的综合性分析
   - 优点: 兼顾全局和局部，结果更全面
   - 缺点: 计算复杂度高，需要调优参数

推荐使用混合算法作为默认选择，可以根据具体需求调整权重参数。
        """)
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

