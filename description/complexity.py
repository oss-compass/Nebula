from typing import Any, Dict, List, Tuple
from .config import DEFAULT_WEIGHTS, COMPLEXITY_THRESHOLDS


def calculate_complexity_score(function_info: Dict, all_functions_data: List[Dict] = None, skip_calculation: bool = False) -> Dict[str, Any]:
    """计算函数的复杂度评分并分类
    
    Args:
        function_info: 单个函数信息
        all_functions_data: 所有函数数据，用于自适应评分
        skip_calculation: 是否跳过复杂度计算，直接返回moderate
        
    Returns:
        包含复杂度评分和分类的字典
    """
    try:
        # 如果跳过复杂度计算，直接返回moderate
        if skip_calculation:
            return {
                "complexity_level": "moderate",
                "complexity_score": 15.0,  # moderate的典型分数
                "cyclomatic_complexity": 0,
                "lines_of_code": 0,
                "branch_count": 0,
                "parameters": 0,
                "raw_metrics": {},
                "adaptive_info": {
                    "weights_used": None,
                    "population_size": 0,
                    "percentile_rank": None,
                    "calculation_skipped": True
                }
            }
        
        # 获取复杂度信息
        complexity = function_info.get('complexity', {})
        semantic = complexity.get('semantic_complexity', {})
        structural = complexity.get('structural_complexity', {})
        
        # 提取关键指标
        cyclomatic = semantic.get('cyclomatic_complexity', 0)
        lines = semantic.get('lines_of_code', 0)
        branches = structural.get('branch_count', 0)
        params = semantic.get('parameters', 0)
        
        # 如果有所有函数数据，计算自适应阈值
        if all_functions_data and len(all_functions_data) > 0:
            # 计算所有函数的统计信息
            all_metrics = _calculate_population_statistics(all_functions_data)
            
            # 使用自适应权重和阈值
            score, weights = _calculate_adaptive_score(
                cyclomatic, lines, branches, params, all_metrics
            )
            
            # 使用自适应分类标准
            level = _classify_by_percentile(score, all_metrics['score_percentiles'])
            
        else:
            # 使用默认权重和阈值
            weights = DEFAULT_WEIGHTS.copy()
            
            score = (
                cyclomatic * weights['cyclomatic'] +
                lines * weights['lines'] +
                branches * weights['branches'] +
                params * weights['params']
            )
            
            # 默认分类标准
            level = _classify_by_threshold(score)
        
        return {
            "complexity_level": level,
            "complexity_score": round(score, 2),
            "cyclomatic_complexity": cyclomatic,
            "lines_of_code": lines,
            "branch_count": branches,
            "parameters": params,
            "raw_metrics": {
                "cyclomatic": cyclomatic,
                "lines": lines,
                "branches": branches,
                "params": params
            },
            "adaptive_info": {
                "weights_used": weights if 'weights' in locals() else None,
                "population_size": len(all_functions_data) if all_functions_data else 0,
                "percentile_rank": _get_percentile_rank(score, all_functions_data) if all_functions_data else None
            }
        }
    except Exception as e:
        # 如果无法获取复杂度信息，返回默认值
        return {
            "complexity_level": "unknown",
            "complexity_score": 0,
            "cyclomatic_complexity": 0,
            "lines_of_code": 0,
            "branch_count": 0,
            "parameters": 0,
            "raw_metrics": {},
            "adaptive_info": {},
            "error": str(e)
        }


def _calculate_population_statistics(all_functions: List[Dict]) -> Dict[str, Any]:
    """计算所有函数的统计信息"""
    metrics = {
        'cyclomatic': [],
        'lines': [],
        'branches': [],
        'params': [],
        'scores': []
    }
    
    for func in all_functions:
        try:
            complexity = func.get('complexity', {})
            semantic = complexity.get('semantic_complexity', {})
            structural = complexity.get('structural_complexity', {})
            
            cyclomatic = semantic.get('cyclomatic_complexity', 0)
            lines = semantic.get('lines_of_code', 0)
            branches = structural.get('branch_count', 0)
            params = semantic.get('parameters', 0)
            
            # 过滤异常值（超过3个标准差的数据）
            if cyclomatic > 0:
                metrics['cyclomatic'].append(cyclomatic)
            if lines > 0:
                metrics['lines'].append(lines)
            if branches > 0:
                metrics['branches'].append(branches)
            if params > 0:
                metrics['params'].append(params)
                
        except Exception:
            continue
    
    # 计算统计信息
    stats = {}
    for metric_name, values in metrics.items():
        if values:
            values = sorted(values)
            stats[f'{metric_name}_mean'] = sum(values) / len(values)
            stats[f'{metric_name}_median'] = values[len(values) // 2]
            stats[f'{metric_name}_std'] = _calculate_std(values)
            stats[f'{metric_name}_min'] = values[0]
            stats[f'{metric_name}_max'] = values[-1]
            stats[f'{metric_name}_q25'] = values[len(values) // 4]
            stats[f'{metric_name}_q75'] = values[3 * len(values) // 4]
    
    # 计算所有函数的评分分布
    all_scores = []
    for func in all_functions:
        try:
            complexity = func.get('complexity', {})
            semantic = complexity.get('semantic_complexity', {})
            structural = complexity.get('structural_complexity', {})
            
            cyclomatic = semantic.get('cyclomatic_complexity', 0)
            lines = semantic.get('lines_of_code', 0)
            branches = structural.get('branch_count', 0)
            params = semantic.get('parameters', 0)
            
            # 使用默认权重计算基础评分
            score = (
                cyclomatic * DEFAULT_WEIGHTS['cyclomatic'] +
                lines * DEFAULT_WEIGHTS['lines'] +
                branches * DEFAULT_WEIGHTS['branches'] +
                params * DEFAULT_WEIGHTS['params']
            )
            all_scores.append(score)
        except Exception:
            continue
    
    if all_scores:
        all_scores.sort()
        stats['score_percentiles'] = {
            'p25': all_scores[len(all_scores) // 4],
            'p50': all_scores[len(all_scores) // 2],
            'p75': all_scores[3 * len(all_scores) // 4],
            'p90': all_scores[int(len(all_scores) * 0.9)],
            'p95': all_scores[int(len(all_scores) * 0.95)]
        }
    
    return stats


def _calculate_adaptive_score(cyclomatic: int, lines: int, branches: int, params: int, 
                            population_stats: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """计算自适应评分"""
    
    # 基于数据分布动态调整权重
    weights = {}
    
    # 圈复杂度权重：如果分布较广，权重增加
    cyclomatic_range = population_stats.get('cyclomatic_max', 0) - population_stats.get('cyclomatic_min', 0)
    if cyclomatic_range > 20:
        weights['cyclomatic'] = 0.5
    elif cyclomatic_range > 10:
        weights['cyclomatic'] = 0.4
    else:
        weights['cyclomatic'] = 0.3
    
    # 代码行数权重：根据分布调整
    lines_std = population_stats.get('lines_std', 0)
    if lines_std > 50:
        weights['lines'] = 0.02
    elif lines_std > 20:
        weights['lines'] = 0.015
    else:
        weights['lines'] = 0.01
    
    # 分支数权重：根据分布调整
    branches_std = population_stats.get('branches_std', 0)
    if branches_std > 10:
        weights['branches'] = 0.35
    elif branches_std > 5:
        weights['branches'] = 0.3
    else:
        weights['branches'] = 0.25
    
    # 参数数权重：根据分布调整
    params_std = population_stats.get('params_std', 0)
    if params_std > 5:
        weights['params'] = 0.15
    elif params_std > 2:
        weights['params'] = 0.1
    else:
        weights['params'] = 0.05
    
    # 归一化权重
    total_weight = sum(weights.values())
    for key in weights:
        weights[key] = weights[key] / total_weight
    
    # 计算评分
    score = (
        cyclomatic * weights['cyclomatic'] +
        lines * weights['lines'] +
        branches * weights['branches'] +
        params * weights['params']
    )
    
    return score, weights


def _classify_by_percentile(score: float, percentiles: Dict[str, float]) -> str:
    """基于百分位数分类"""
    if not percentiles:
        # 如果没有百分位数信息，使用默认分类
        return _classify_by_threshold(score)
    
    # 基于百分位数分类
    if score <= percentiles.get('p25', 10):
        return "simple"
    elif score <= percentiles.get('p75', 25):
        return "moderate"
    else:
        return "complex"


def _classify_by_threshold(score: float) -> str:
    """基于阈值分类"""
    if score < COMPLEXITY_THRESHOLDS["simple"]:
        return "simple"
    elif score < COMPLEXITY_THRESHOLDS["moderate"]:
        return "moderate"
    else:
        return "complex"


def _get_percentile_rank(score: float, all_functions: List[Dict]) -> float:
    """获取评分的百分位排名"""
    try:
        all_scores = []
        for func in all_functions:
            try:
                complexity = func.get('complexity', {})
                semantic = complexity.get('semantic_complexity', {})
                structural = complexity.get('structural_complexity', {})
                
                cyclomatic = semantic.get('cyclomatic_complexity', 0)
                lines = semantic.get('lines_of_code', 0)
                branches = structural.get('branch_count', 0)
                params = semantic.get('parameters', 0)
                
                func_score = (
                    cyclomatic * DEFAULT_WEIGHTS['cyclomatic'] +
                    lines * DEFAULT_WEIGHTS['lines'] +
                    branches * DEFAULT_WEIGHTS['branches'] +
                    params * DEFAULT_WEIGHTS['params']
                )
                all_scores.append(func_score)
            except Exception:
                continue
        
        if not all_scores:
            return 0.0
        
        all_scores.sort()
        rank = 0
        for i, s in enumerate(all_scores):
            if s <= score:
                rank = i + 1
        
        return (rank / len(all_scores)) * 100
        
    except Exception:
        return 0.0


def _calculate_std(values: List[float]) -> float:
    """计算标准差"""
    if len(values) < 2:
        return 0.0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5
