import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from .config import IMPORTANCE_THRESHOLDS


def calculate_function_importance(function_map: Dict, direct_calls: Dict, transitive_calls: Dict, transitive_paths: Dict, 
                                algorithm: str = "hybrid", pagerank_weight: float = 0.4, traditional_weight: float = 0.4, 
                                complexity_weight: float = 0.2) -> Dict:
    """
    计算函数重要�?    
    支持三种算法�?    1. "traditional": 传统的多维度统计算法
    2. "pagerank": 基于PageRank的图算法
    3. "hybrid": 混合算法，结合PageRank和传统算�?    
    传统算法考虑因素�?    1. 直接调用�?(Direct Call Weight)
    2. 传递调用量 (Transitive Call Weight) 
    3. 调用深度影响 (Depth Impact)
    4. 枢纽效应 (Hub Effect)
    5. 被调用频�?(Incoming Call Frequency)
    6. 复杂度调�?(Complexity Adjustment)
    
    PageRank算法�?    - 基于调用图计算全局重要�?    - 考虑传递性影�?    - 识别关键枢纽函数
    
    混合算法�?    最终重要度 = α × PageRank分数 + β × 传统分数 + γ × 复杂度调�?    
    Args:
        function_map: 函数名到函数信息的映�?        direct_calls: 直接调用关系
        transitive_calls: 传递调用关�?        transitive_paths: 传递调用路�?        algorithm: 算法类型 ("traditional", "pagerank", "hybrid")
        pagerank_weight: PageRank权重 (混合算法�?
        traditional_weight: 传统算法权重 (混合算法�?
        complexity_weight: 复杂度权�?(混合算法�?
        
    Returns:
        重要度分析结�?    """
    print(f"正在计算函数重要�?.. 使用算法: {algorithm}")
    
    # 根据算法类型选择计算方法
    if algorithm == "pagerank":
        return _calculate_pagerank_importance(function_map, direct_calls, transitive_calls, transitive_paths)
    elif algorithm == "traditional":
        return _calculate_traditional_importance(function_map, direct_calls, transitive_calls, transitive_paths)
    elif algorithm == "hybrid":
        return _calculate_hybrid_importance(function_map, direct_calls, transitive_calls, transitive_paths,
                                          pagerank_weight, traditional_weight, complexity_weight)
    else:
        raise ValueError(f"不支持的算法类型: {algorithm}")


def _calculate_traditional_importance(function_map: Dict, direct_calls: Dict, transitive_calls: Dict, transitive_paths: Dict) -> Dict:
    """传统的多维度统计算法"""
    print("使用传统统计算法...")
    
    importance_scores = {}
    
    # 统计每个函数被调用的次数（入度）
    incoming_calls = defaultdict(int)
    for targets in direct_calls.values():
        for target in targets:
            incoming_calls[target] += 1
    
    # 统计每个函数在传递调用中被影响的次数
    transitive_impact = defaultdict(int)
    for targets in transitive_calls.values():
        for target in targets:
            transitive_impact[target] += 1
    
    for func_name in function_map.keys():
        # 1. 直接调用权重 (DCW) - 该函数调用其他函数的数量
        direct_call_weight = len(direct_calls.get(func_name, set()))
        
        # 2. 传递调用权�?(TCW) - 该函数通过传递调用影响的其他函数数量
        transitive_call_weight = len(transitive_calls.get(func_name, set()))
        
        # 3. 调用深度影响 (DI) - 考虑调用链的深度，深度越深影响越�?        depth_impact = 0
        max_depth = 0
        for path in transitive_paths.values():
            if func_name in path:
                depth = len(path)
                max_depth = max(max_depth, depth)
                # 深度影响：深度越深，影响越大，使用对数增�?                depth_impact += math.log(depth + 1, 2)
        
        # 4. 枢纽效应 (HE) - 如果函数既是调用者又是被调用者，说明它是枢纽
        hub_effect = 0
        if func_name in direct_calls and incoming_calls[func_name] > 0:
            # 枢纽效应：既是调用者又是被调用者，权重更高
            hub_effect = min(direct_call_weight, incoming_calls[func_name]) * 0.5
        
        # 5. 被调用频�?(ICF) - 该函数被其他函数调用的次�?        incoming_call_frequency = incoming_calls[func_name]
        
        # 6. 复杂度调整因�?(CA) - 基于函数的复杂度进行调整
        func_info = function_map[func_name]
        complexity_score = func_info["complexity"]["semantic_complexity"]["complexity_score"]
        lines_of_code = func_info["complexity"]["semantic_complexity"]["lines_of_code"]
        cyclomatic_complexity = func_info["complexity"]["semantic_complexity"]["cyclomatic_complexity"]
        
        # 复杂度调整：复杂度越高，重要度越高，但过高的复杂度可能表示代码质量差
        complexity_adjustment = 1.0
        if complexity_score > 0:
            # 使用对数调整，避免过高复杂度过度影响
            complexity_adjustment = 1.0 + math.log(complexity_score + 1, 10) * 0.3
        
        # 7. 特殊情况的额外权�?        special_weight = 0
        
        # 如果函数只调用了很少的函数，但被很多函数调用，说明它是重要的基础函数
        if direct_call_weight <= 2 and incoming_call_frequency >= 5:
            special_weight = incoming_call_frequency * 0.2
        
        # 如果函数调用了很多其他函数，说明它是重要的协调函�?        if direct_call_weight >= 5:
            special_weight += direct_call_weight * 0.1
        
        # 8. 计算最终重要度
        base_importance = (
            direct_call_weight * 1.0 +           # 直接调用权重
            transitive_call_weight * 0.8 +       # 传递调用权重（稍微降低�?            depth_impact * 0.6 +                 # 深度影响
            hub_effect * 1.2 +                   # 枢纽效应（权重最高）
            incoming_call_frequency * 0.7 +      # 被调用频�?            special_weight                        # 特殊情况权重
        )
        
        final_importance = base_importance * complexity_adjustment
        
        # 9. 计算重要度等�?        importance_level = _get_importance_level(final_importance)
        
        importance_scores[func_name] = {
            "total_score": round(final_importance, 2),
            "importance_level": importance_level,
            "breakdown": {
                "direct_call_weight": direct_call_weight,
                "transitive_call_weight": transitive_call_weight,
                "depth_impact": round(depth_impact, 2),
                "hub_effect": round(hub_effect, 2),
                "incoming_call_frequency": incoming_call_frequency,
                "special_weight": round(special_weight, 2),
                "complexity_adjustment": round(complexity_adjustment, 2)
            },
            "metrics": {
                "max_call_depth": max_depth,
                "total_influence_scope": direct_call_weight + transitive_call_weight,
                "is_hub": func_name in direct_calls and incoming_calls[func_name] > 0,
                "is_leaf": func_name not in direct_calls and incoming_calls[func_name] > 0,
                "is_coordinator": direct_call_weight >= 5,
                "is_foundation": direct_call_weight <= 2 and incoming_call_frequency >= 5
            }
        }
    
    # 按重要度排序
    sorted_importance = sorted(
        importance_scores.items(), 
        key=lambda x: x[1]["total_score"], 
        reverse=True
    )
    
    # 添加排名信息
    for rank, (func_name, score_info) in enumerate(sorted_importance, 1):
        score_info["rank"] = rank
    
    return {
        "algorithm": "traditional",
        "scores": importance_scores,
        "top_functions": [
            {
                "function_name": func_name,
                "total_score": score_info["total_score"],
                "importance_level": score_info["importance_level"],
                "rank": score_info["rank"]
            }
            for func_name, score_info in sorted_importance[:20]  # �?0�?        ],
        "importance_distribution": _calculate_importance_distribution(importance_scores)
    }


def _calculate_pagerank_importance(function_map: Dict, direct_calls: Dict, transitive_calls: Dict, transitive_paths: Dict) -> Dict:
    """基于PageRank的图算法"""
    print("使用PageRank算法...")
    
    if not NETWORKX_AVAILABLE:
        print("警告: NetworkX不可用，回退到传统算�?)
        return _calculate_traditional_importance(function_map, direct_calls, transitive_calls, transitive_paths)
    
    # 构建调用�?    G = _build_call_graph(function_map, direct_calls)
    
    if G.number_of_nodes() == 0:
        print("警告: 图中没有节点，回退到传统算�?)
        return _calculate_traditional_importance(function_map, direct_calls, transitive_calls, transitive_paths)
    
    # 计算PageRank分数
    try:
        pagerank_scores = nx.pagerank(G, weight='weight', max_iter=1000, tol=1e-06)
    except Exception as e:
        print(f"PageRank计算失败: {e}，回退到传统算�?)
        return _calculate_traditional_importance(function_map, direct_calls, transitive_calls, transitive_paths)
    
    # 计算复杂度调整因�?    complexity_adjustments = {}
    for func_name, func_info in function_map.items():
        complexity_score = func_info["complexity"]["semantic_complexity"]["complexity_score"]
        if complexity_score > 0:
            complexity_adjustments[func_name] = 1.0 + math.log(complexity_score + 1, 10) * 0.3
        else:
            complexity_adjustments[func_name] = 1.0
    
    # 计算最终重要度分数
    importance_scores = {}
    for func_name in function_map.keys():
        pagerank_score = pagerank_scores.get(func_name, 0.0)
        complexity_adj = complexity_adjustments.get(func_name, 1.0)
        
        # 最终分�?= PageRank分数 * 复杂度调�?        final_score = pagerank_score * complexity_adj * 1000  # 放大到合理范�?        
        importance_level = _get_importance_level(final_score)
        
        importance_scores[func_name] = {
            "total_score": round(final_score, 2),
            "importance_level": importance_level,
            "breakdown": {
                "pagerank_score": round(pagerank_score, 6),
                "complexity_adjustment": round(complexity_adj, 2)
            },
            "metrics": {
                "is_hub": func_name in direct_calls and len(direct_calls[func_name]) > 0,
                "out_degree": len(direct_calls.get(func_name, set())),
                "in_degree": sum(1 for targets in direct_calls.values() if func_name in targets)
            }
        }
    
    # 按重要度排序
    sorted_importance = sorted(
        importance_scores.items(), 
        key=lambda x: x[1]["total_score"], 
        reverse=True
    )
    
    # 添加排名信息
    for rank, (func_name, score_info) in enumerate(sorted_importance, 1):
        score_info["rank"] = rank
    
    return {
        "algorithm": "pagerank",
        "scores": importance_scores,
        "top_functions": [
            {
                "function_name": func_name,
                "total_score": score_info["total_score"],
                "importance_level": score_info["importance_level"],
                "rank": score_info["rank"]
            }
            for func_name, score_info in sorted_importance[:20]
        ],
        "importance_distribution": _calculate_importance_distribution(importance_scores),
        "graph_stats": {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "is_connected": nx.is_weakly_connected(G) if G.number_of_nodes() > 0 else False
        }
    }


def _calculate_hybrid_importance(function_map: Dict, direct_calls: Dict, transitive_calls: Dict, transitive_paths: Dict,
                               pagerank_weight: float, traditional_weight: float, complexity_weight: float) -> Dict:
    """混合算法：结合PageRank和传统算�?""
    print(f"使用混合算法... PageRank权重: {pagerank_weight}, 传统权重: {traditional_weight}, 复杂度权�? {complexity_weight}")
    
    # 计算传统算法分数
    traditional_result = _calculate_traditional_importance(function_map, direct_calls, transitive_calls, transitive_paths)
    if traditional_result is None:
        print("警告: 传统算法计算失败，回退到传统算�?)
        return _calculate_traditional_importance(function_map, direct_calls, transitive_calls, transitive_paths)
    traditional_scores = traditional_result["scores"]
    
    # 计算PageRank分数
    pagerank_result = _calculate_pagerank_importance(function_map, direct_calls, transitive_calls, transitive_paths)
    if pagerank_result is None:
        print("警告: PageRank算法计算失败，回退到传统算�?)
        return _calculate_traditional_importance(function_map, direct_calls, transitive_calls, transitive_paths)
    pagerank_scores = pagerank_result["scores"]
    
    # 归一化分数到相同范围
    traditional_max = max(score["total_score"] for score in traditional_scores.values()) if traditional_scores else 1
    pagerank_max = max(score["total_score"] for score in pagerank_scores.values()) if pagerank_scores else 1
    
    # 计算混合分数
    importance_scores = {}
    for func_name in function_map.keys():
        # 归一化分�?        norm_traditional = (traditional_scores[func_name]["total_score"] / traditional_max) if traditional_max > 0 else 0
        norm_pagerank = (pagerank_scores[func_name]["total_score"] / pagerank_max) if pagerank_max > 0 else 0
        
        # 获取复杂度调�?        complexity_adj = traditional_scores[func_name]["breakdown"]["complexity_adjustment"]
        
        # 混合分数
        hybrid_score = (
            pagerank_weight * norm_pagerank * 100 +
            traditional_weight * norm_traditional * 100 +
            complexity_weight * (complexity_adj - 1.0) * 50
        )
        
        importance_level = _get_importance_level(hybrid_score)
        
        importance_scores[func_name] = {
            "total_score": round(hybrid_score, 2),
            "importance_level": importance_level,
            "breakdown": {
                "pagerank_score": round(norm_pagerank * 100, 2),
                "traditional_score": round(norm_traditional * 100, 2),
                "complexity_adjustment": round(complexity_adj, 2),
                "pagerank_weight": pagerank_weight,
                "traditional_weight": traditional_weight,
                "complexity_weight": complexity_weight
            },
            "metrics": {
                **traditional_scores[func_name]["metrics"],
                **pagerank_scores[func_name]["metrics"]
            }
        }
    
    # 按重要度排序
    sorted_importance = sorted(
        importance_scores.items(), 
        key=lambda x: x[1]["total_score"], 
        reverse=True
    )
    
    # 添加排名信息
    for rank, (func_name, score_info) in enumerate(sorted_importance, 1):
        score_info["rank"] = rank
    
    return {
        "algorithm": "hybrid",
        "scores": importance_scores,
        "top_functions": [
            {
                "function_name": func_name,
                "total_score": score_info["total_score"],
                "importance_level": score_info["importance_level"],
                "rank": score_info["rank"]
            }
            for func_name, score_info in sorted_importance[:20]
        ],
        "importance_distribution": _calculate_importance_distribution(importance_scores),
        "algorithm_weights": {
            "pagerank_weight": pagerank_weight,
            "traditional_weight": traditional_weight,
            "complexity_weight": complexity_weight
        }
    }


def _build_call_graph(function_map: Dict, direct_calls: Dict) -> 'nx.DiGraph':
    """构建调用�?""
    G = nx.DiGraph()
    
    # 添加所有函数作为节�?    for func_name in function_map.keys():
        G.add_node(func_name)
    
    # 添加调用关系作为�?    for caller, callees in direct_calls.items():
        for callee in callees:
            if callee in function_map:  # 只添加图中存在的函数
                G.add_edge(caller, callee, weight=1.0)
    
    return G


def _get_importance_level(score: float) -> str:
    """
    根据分数获取重要度等�?    
    Args:
        score: 重要度分�?        
    Returns:
        重要度等�?    """
    if score >= IMPORTANCE_THRESHOLDS["Critical"]:
        return "Critical"
    elif score >= IMPORTANCE_THRESHOLDS["High"]:
        return "High"
    elif score >= IMPORTANCE_THRESHOLDS["Medium"]:
        return "Medium"
    elif score >= IMPORTANCE_THRESHOLDS["Low"]:
        return "Low"
    else:
        return "Minimal"


def _calculate_importance_distribution(importance_scores: Dict) -> Dict:
    """
    计算重要度分�?    
    Args:
        importance_scores: 重要度分数字�?        
    Returns:
        重要度分布统�?    """
    distribution = {
        "Critical": 0,
        "High": 0,
        "Medium": 0,
        "Low": 0,
        "Minimal": 0
    }
    
    for score_info in importance_scores.values():
        level = score_info["importance_level"]
        if level in distribution:
            distribution[level] += 1
    
    return distribution


def update_function_importance(all_functions: List[Dict], importance_scores: Dict) -> None:
    """
    更新所有函数的重要度信�?    
    Args:
        all_functions: 所有函数列�?        importance_scores: 重要度分数字�?    """
    print("  更新函数重要度信�?..")
    
    scores_dict = importance_scores.get('scores', {})
    
    for func in all_functions:
        func_name = func["basic_info"]["function_name"]
        # 确保func_name是字符串类型
        if isinstance(func_name, bytes):
            func_name = func_name.decode('utf-8')
        if func_name in scores_dict:
            func["importance"] = scores_dict[func_name]
        else:
            func["importance"] = {
                "status": "calculated",
                "total_score": 0,
                "importance_level": "Minimal",
                "breakdown": {
                    "direct_call_weight": 0,
                    "transitive_call_weight": 0,
                    "depth_impact": 0,
                    "hub_effect": 0,
                    "incoming_call_frequency": 0,
                    "special_weight": 0,
                    "complexity_adjustment": 1.0
                },
                "metrics": {
                    "is_hub": False,
                    "is_leaf": False,
                    "is_coordinator": False,
                    "is_foundation": False
                }
            }


def analyze_importance_trends(importance_scores: Dict) -> Dict:
    """
    分析重要度趋�?    
    Args:
        importance_scores: 重要度分数字�?        
    Returns:
        重要度趋势分析结�?    """
    scores_dict = importance_scores.get('scores', {})
    
    if not scores_dict:
        return {}
    
    # 按分数分�?    score_groups = {
        "Critical": [],
        "High": [],
        "Medium": [],
        "Low": [],
        "Minimal": []
    }
    
    for func_name, score_info in scores_dict.items():
        level = score_info["importance_level"]
        score_groups[level].append({
            "function_name": func_name,
            "score": score_info["total_score"]
        })
    
    # 计算每个等级的统计信�?    trends = {}
    for level, functions in score_groups.items():
        if functions:
            scores = [f["score"] for f in functions]
            trends[level] = {
                "count": len(functions),
                "avg_score": sum(scores) / len(scores),
                "max_score": max(scores),
                "min_score": min(scores),
                "top_functions": sorted(functions, key=lambda x: x["score"], reverse=True)[:5]
            }
        else:
            trends[level] = {
                "count": 0,
                "avg_score": 0,
                "max_score": 0,
                "min_score": 0,
                "top_functions": []
            }
    
    return trends


def compare_algorithms(function_map: Dict, direct_calls: Dict, transitive_calls: Dict, transitive_paths: Dict) -> Dict:
    """
    比较不同算法的重要度计算结果
    
    Args:
        function_map: 函数名到函数信息的映�?        direct_calls: 直接调用关系
        transitive_calls: 传递调用关�?        transitive_paths: 传递调用路�?        
    Returns:
        算法比较结果
    """
    print("比较不同算法的重要度计算结果...")
    
    algorithms = ["traditional", "pagerank", "hybrid"]
    results = {}
    
    for algorithm in algorithms:
        print(f"计算 {algorithm} 算法...")
        if algorithm == "hybrid":
            result = calculate_function_importance(
                function_map, direct_calls, transitive_calls, transitive_paths,
                algorithm=algorithm, pagerank_weight=0.4, traditional_weight=0.4, complexity_weight=0.2
            )
        else:
            result = calculate_function_importance(
                function_map, direct_calls, transitive_calls, transitive_paths,
                algorithm=algorithm
            )
        results[algorithm] = result
    
    # 分析算法差异
    comparison = {
        "algorithms": algorithms,
        "results": results,
        "top_functions_comparison": {},
        "correlation_analysis": {},
        "algorithm_characteristics": {
            "traditional": {
                "description": "基于统计的多维度算法",
                "strengths": ["考虑复杂�?, "可解释性强", "计算简�?],
                "weaknesses": ["忽略全局影响", "权重主观", "缺乏传递性分�?]
            },
            "pagerank": {
                "description": "基于图结构的全局重要性算�?,
                "strengths": ["全局视角", "传递性分�?, "数学基础扎实"],
                "weaknesses": ["忽略代码复杂�?, "对孤立节点不友好", "需要完整图结构"]
            },
            "hybrid": {
                "description": "结合PageRank和传统算法的混合方法",
                "strengths": ["兼顾全局和局�?, "平衡多种因素", "可调节权�?],
                "weaknesses": ["计算复杂度高", "参数调优复杂", "结果解释性降�?]
            }
        }
    }
    
    # 比较�?0名函�?    for algorithm in algorithms:
        top_functions = results[algorithm]["top_functions"][:10]
        comparison["top_functions_comparison"][algorithm] = [
            {
                "function_name": func["function_name"],
                "score": func["total_score"],
                "rank": func["rank"]
            }
            for func in top_functions
        ]
    
    # 计算算法间的相关�?    if len(algorithms) >= 2:
        for i, alg1 in enumerate(algorithms):
            for alg2 in algorithms[i+1:]:
                scores1 = {func["function_name"] if isinstance(func["function_name"], str) else func["function_name"].decode('utf-8'): func["total_score"] 
                          for func in results[alg1]["top_functions"]}
                scores2 = {func["function_name"] if isinstance(func["function_name"], str) else func["function_name"].decode('utf-8'): func["total_score"] 
                          for func in results[alg2]["top_functions"]}
                
                # 计算共同函数的分数相关�?                common_functions = set(scores1.keys()) & set(scores2.keys())
                if len(common_functions) > 1:
                    scores1_list = [scores1[func] for func in common_functions]
                    scores2_list = [scores2[func] for func in common_functions]
                    
                    correlation = np.corrcoef(scores1_list, scores2_list)[0, 1] if len(scores1_list) > 1 else 0
                    comparison["correlation_analysis"][f"{alg1}_vs_{alg2}"] = {
                        "correlation": round(correlation, 3),
                        "common_functions": len(common_functions)
                    }
    
    return comparison


def get_importance_summary(importance_scores: Dict) -> Dict:
    """
    获取重要度分析摘�?    
    Args:
        importance_scores: 重要度分数字�?        
    Returns:
        重要度分析摘�?    """
    scores_dict = importance_scores.get('scores', {})
    distribution = importance_scores.get('importance_distribution', {})
    top_functions = importance_scores.get('top_functions', [])
    
    total_functions = len(scores_dict)
    critical_functions = distribution.get('Critical', 0)
    high_functions = distribution.get('High', 0)
    
    # 计算关键函数比例
    critical_ratio = critical_functions / total_functions if total_functions > 0 else 0
    high_importance_ratio = (critical_functions + high_functions) / total_functions if total_functions > 0 else 0
    
    return {
        "total_functions": total_functions,
        "critical_functions": critical_functions,
        "high_importance_functions": critical_functions + high_functions,
        "critical_ratio": round(critical_ratio * 100, 2),
        "high_importance_ratio": round(high_importance_ratio * 100, 2),
        "top_5_functions": top_functions[:5],
        "importance_distribution": distribution,
        "analysis_quality": "high" if total_functions > 10 else "low"
    }

