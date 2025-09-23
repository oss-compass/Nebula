import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass
import concurrent.futures
from threading import Lock
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available, semantic analysis will be disabled")

from .vector_embedding import CodeVectorizer, EmbeddingConfig, CodeEmbedding
from .multi_repo import MultiRepoSearch, RepoConfig, MultiRepoSearchConfig

logger = logging.getLogger(__name__)

@dataclass
class APISimilarityConfig:
    """API相似性分析配�?""
    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.8
    semantic_similarity_threshold: float = 0.7
    structural_similarity_threshold: float = 0.6
    max_apis_per_repo: int = 1000
    enable_parallel_analysis: bool = True
    max_workers: int = 4
    enable_cache: bool = True

@dataclass
class APISignature:
    """API签名"""
    name: str
    parameters: List[str]
    return_type: Optional[str]
    docstring: Optional[str]
    file_path: str
    line_number: int
    repo_name: str

@dataclass
class SimilarityResult:
    """相似性分析结�?""
    api1: APISignature
    api2: APISignature
    semantic_similarity: float
    structural_similarity: float
    overall_similarity: float
    similarity_type: str  # "exact", "semantic", "structural", "hybrid"
    explanation: str

@dataclass
class APIAnalysisResult:
    """API分析结果"""
    api_name: str
    total_occurrences: int
    repos_containing: List[str]
    similarity_groups: List[List[APISignature]]
    usage_patterns: Dict[str, Any]
    evolution_analysis: Optional[Dict[str, Any]] = None

class APISimilarityAnalyzer:
    """API相似性分析器"""
    
    def __init__(self, 
                 multi_repo_search: MultiRepoSearch,
                 config: Optional[APISimilarityConfig] = None):
        """
        初始化API相似性分析器
        
        Args:
            multi_repo_search: 多库搜索器实�?
            config: 分析配置
        """
        self.multi_repo_search = multi_repo_search
        self.config = config or APISimilarityConfig()
        self.analysis_cache = {}
        self.analysis_history = []
        self.cache_lock = Lock()
        
        # 初始化向量化�?
        self.vectorizer = None
        self._init_vectorizer()
        
        logger.info("API相似性分析器初始化完�?)
    
    def _init_vectorizer(self):
        """初始化向量化�?""
        try:
            embedding_config = EmbeddingConfig(
                model_name=self.config.embedding_model,
                device="cpu"
            )
            self.vectorizer = CodeVectorizer(embedding_config)
            logger.info(f"向量化器初始化成功，模型: {self.config.embedding_model}")
        except Exception as e:
            logger.error(f"向量化器初始化失�? {e}")
            self.vectorizer = None
    
    def analyze_api_similarity(self, 
                             api_name: str,
                             similarity_threshold: float = None,
                             analysis_type: str = "comprehensive") -> APIAnalysisResult:
        """
        分析API相似�?
        
        Args:
            api_name: API名称
            similarity_threshold: 相似度阈�?
            analysis_type: 分析类型 ("comprehensive", "semantic", "structural")
            
        Returns:
            API分析结果
        """
        logger.info(f"分析API相似�? {api_name}")
        
        # 检查缓�?
        cache_key = self._get_cache_key(api_name, similarity_threshold, analysis_type)
        if self.config.enable_cache:
            with self.cache_lock:
                if cache_key in self.analysis_cache:
                    logger.info("返回缓存结果")
                    return self.analysis_cache[cache_key]
        
        # 搜索所有相关API
        all_apis = self._find_all_related_apis(api_name)
        
        if not all_apis:
            logger.warning(f"未找到相关API: {api_name}")
            return APIAnalysisResult(
                api_name=api_name,
                total_occurrences=0,
                repos_containing=[],
                similarity_groups=[],
                usage_patterns={}
            )
        
        # 执行相似性分�?
        similarity_results = self._analyze_similarities(all_apis, similarity_threshold, analysis_type)
        
        # 构建相似性组
        similarity_groups = self._build_similarity_groups(similarity_results)
        
        # 分析使用模式
        usage_patterns = self._analyze_usage_patterns(all_apis)
        
        # 构建结果
        result = APIAnalysisResult(
            api_name=api_name,
            total_occurrences=len(all_apis),
            repos_containing=list(set(api.repo_name for api in all_apis)),
            similarity_groups=similarity_groups,
            usage_patterns=usage_patterns
        )
        
        # 缓存结果
        if self.config.enable_cache:
            with self.cache_lock:
                self.analysis_cache[cache_key] = result
        
        # 记录分析历史
        self._record_analysis(api_name, result)
        
        return result
    
    def find_duplicate_apis(self, 
                          similarity_threshold: float = None,
                          target_repos: Optional[List[str]] = None) -> Dict[str, List[SimilarityResult]]:
        """
        查找重复API
        
        Args:
            similarity_threshold: 相似度阈�?
            target_repos: 目标仓库列表
            
        Returns:
            重复API分析结果
        """
        logger.info("查找重复API")
        
        threshold = similarity_threshold or self.config.similarity_threshold
        
        # 获取所有API
        all_apis = self._get_all_apis(target_repos)
        
        if len(all_apis) < 2:
            logger.warning("API数量不足，无法进行重复检�?)
            return {}
        
        # 执行重复检�?
        duplicate_groups = defaultdict(list)
        
        if self.config.enable_parallel_analysis:
            duplicate_results = self._parallel_duplicate_detection(all_apis, threshold)
        else:
            duplicate_results = self._sequential_duplicate_detection(all_apis, threshold)
        
        # 组织结果
        for result in duplicate_results:
            if result.overall_similarity >= threshold:
                group_key = f"{result.api1.name}_{result.api2.name}"
                duplicate_groups[group_key].append(result)
        
        return dict(duplicate_groups)
    
    def recommend_similar_apis(self, 
                             api_name: str,
                             top_k: int = 10,
                             similarity_threshold: float = None) -> List[Dict[str, Any]]:
        """
        推荐相似API
        
        Args:
            api_name: API名称
            top_k: 返回的推荐数�?
            similarity_threshold: 相似度阈�?
            
        Returns:
            相似API推荐列表
        """
        logger.info(f"推荐相似API: {api_name}")
        
        # 分析API相似�?
        analysis_result = self.analyze_api_similarity(api_name, similarity_threshold)
        
        # 构建推荐列表
        recommendations = []
        threshold = similarity_threshold or self.config.similarity_threshold
        
        for group in analysis_result.similarity_groups:
            if len(group) > 1:  # 只考虑有相似API的组
                for api in group:
                    if api.name != api_name:  # 排除自身
                        # 计算与目标API的相似度
                        similarity_score = self._calculate_api_similarity(
                            self._find_api_by_name(api_name, analysis_result.similarity_groups),
                            api
                        )
                        
                        if similarity_score >= threshold:
                            recommendation = {
                                "api_name": api.name,
                                "similarity_score": similarity_score,
                                "repo_name": api.repo_name,
                                "file_path": api.file_path,
                                "parameters": api.parameters,
                                "return_type": api.return_type,
                                "docstring": api.docstring,
                                "explanation": f"�?{api_name} 的相似度�?{similarity_score:.3f}"
                            }
                            recommendations.append(recommendation)
        
        # 按相似度排序并限制数�?
        recommendations.sort(key=lambda x: x["similarity_score"], reverse=True)
        return recommendations[:top_k]
    
    def analyze_api_evolution(self, 
                            api_name: str,
                            target_repos: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        分析API演化
        
        Args:
            api_name: API名称
            target_repos: 目标仓库列表
            
        Returns:
            API演化分析结果
        """
        logger.info(f"分析API演化: {api_name}")
        
        # 获取所有相关API
        all_apis = self._find_all_related_apis(api_name, target_repos)
        
        if not all_apis:
            return {"error": f"未找到API: {api_name}"}
        
        # 分析演化模式
        evolution_analysis = {
            "api_name": api_name,
            "total_versions": len(all_apis),
            "repos_containing": list(set(api.repo_name for api in all_apis)),
            "parameter_evolution": self._analyze_parameter_evolution(all_apis),
            "return_type_evolution": self._analyze_return_type_evolution(all_apis),
            "docstring_evolution": self._analyze_docstring_evolution(all_apis),
            "usage_frequency": self._analyze_usage_frequency(all_apis)
        }
        
        return evolution_analysis
    
    def _find_all_related_apis(self, 
                              api_name: str,
                              target_repos: Optional[List[str]] = None) -> List[APISignature]:
        """查找所有相关API"""
        all_apis = []
        
        # 使用多库搜索查找相关API
        search_result = self.multi_repo_search.search_across_repos(
            query=f"API function {api_name}",
            search_type="hybrid",
            target_repos=target_repos
        )
        
        # 从搜索结果中提取API签名
        for cross_repo_result in search_result.cross_repo_results:
            for result in cross_repo_result.results:
                api_signature = self._extract_api_signature(result, cross_repo_result.repo_name)
                if api_signature:
                    all_apis.append(api_signature)
        
        return all_apis
    
    def _extract_api_signature(self, result, repo_name: str) -> Optional[APISignature]:
        """从搜索结果中提取API签名"""
        try:
            content = result.code_embedding.content
            
            # 提取函数�?
            name_match = re.search(r'def\s+(\w+)\s*\(', content)
            if not name_match:
                return None
            
            function_name = name_match.group(1)
            
            # 提取参数
            params_match = re.search(r'def\s+\w+\s*\((.*?)\)', content, re.DOTALL)
            parameters = []
            if params_match:
                params_str = params_match.group(1).strip()
                if params_str:
                    parameters = [param.strip().split(':')[0].strip() for param in params_str.split(',')]
            
            # 提取返回类型
            return_type = None
            return_match = re.search(r'->\s*(\w+)', content)
            if return_match:
                return_type = return_match.group(1)
            
            # 提取文档字符�?
            docstring = None
            docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
            if docstring_match:
                docstring = docstring_match.group(1).strip()
            
            return APISignature(
                name=function_name,
                parameters=parameters,
                return_type=return_type,
                docstring=docstring,
                file_path=getattr(result.code_embedding, 'file_path', 'Unknown'),
                line_number=getattr(result.code_embedding, 'line_number', 0),
                repo_name=repo_name
            )
            
        except Exception as e:
            logger.error(f"提取API签名失败: {e}")
            return None
    
    def _analyze_similarities(self, 
                            apis: List[APISignature],
                            similarity_threshold: float,
                            analysis_type: str) -> List[SimilarityResult]:
        """分析API相似�?""
        similarity_results = []
        threshold = similarity_threshold or self.config.similarity_threshold
        
        # 比较所有API�?
        for i in range(len(apis)):
            for j in range(i + 1, len(apis)):
                similarity_result = self._compare_apis(apis[i], apis[j], analysis_type)
                if similarity_result.overall_similarity >= threshold:
                    similarity_results.append(similarity_result)
        
        return similarity_results
    
    def _compare_apis(self, 
                     api1: APISignature, 
                     api2: APISignature,
                     analysis_type: str) -> SimilarityResult:
        """比较两个API"""
        # 计算语义相似�?
        semantic_similarity = self._calculate_semantic_similarity(api1, api2)
        
        # 计算结构相似�?
        structural_similarity = self._calculate_structural_similarity(api1, api2)
        
        # 计算整体相似�?
        overall_similarity = (semantic_similarity + structural_similarity) / 2
        
        # 确定相似性类�?
        similarity_type = self._determine_similarity_type(api1, api2, semantic_similarity, structural_similarity)
        
        # 生成解释
        explanation = self._generate_similarity_explanation(api1, api2, semantic_similarity, structural_similarity)
        
        return SimilarityResult(
            api1=api1,
            api2=api2,
            semantic_similarity=semantic_similarity,
            structural_similarity=structural_similarity,
            overall_similarity=overall_similarity,
            similarity_type=similarity_type,
            explanation=explanation
        )
    
    def _calculate_semantic_similarity(self, api1: APISignature, api2: APISignature) -> float:
        """计算语义相似�?""
        if not self.vectorizer:
            return 0.0
        
        try:
            # 构建API描述文本
            text1 = self._build_api_description(api1)
            text2 = self._build_api_description(api2)
            
            # 生成嵌入向量
            embedding1 = self.vectorizer.embed_text(text1)
            embedding2 = self.vectorizer.embed_text(text2)
            
            # 计算余弦相似�?
            return self._calculate_cosine_similarity(embedding1, embedding2)
            
        except Exception as e:
            logger.error(f"计算语义相似性失�? {e}")
            return 0.0
    
    def _calculate_structural_similarity(self, api1: APISignature, api2: APISignature) -> float:
        """计算结构相似�?""
        # 参数数量相似�?
        param_count_similarity = self._calculate_parameter_count_similarity(api1.parameters, api2.parameters)
        
        # 参数类型相似�?
        param_type_similarity = self._calculate_parameter_type_similarity(api1.parameters, api2.parameters)
        
        # 返回类型相似�?
        return_type_similarity = self._calculate_return_type_similarity(api1.return_type, api2.return_type)
        
        # 名称相似�?
        name_similarity = self._calculate_name_similarity(api1.name, api2.name)
        
        # 加权平均
        structural_similarity = (
            param_count_similarity * 0.3 +
            param_type_similarity * 0.3 +
            return_type_similarity * 0.2 +
            name_similarity * 0.2
        )
        
        return structural_similarity
    
    def _calculate_parameter_count_similarity(self, params1: List[str], params2: List[str]) -> float:
        """计算参数数量相似�?""
        count1, count2 = len(params1), len(params2)
        if count1 == 0 and count2 == 0:
            return 1.0
        if count1 == 0 or count2 == 0:
            return 0.0
        
        return 1.0 - abs(count1 - count2) / max(count1, count2)
    
    def _calculate_parameter_type_similarity(self, params1: List[str], params2: List[str]) -> float:
        """计算参数类型相似�?""
        if not params1 and not params2:
            return 1.0
        if not params1 or not params2:
            return 0.0
        
        # 简单的类型匹配
        type1 = set(re.findall(r':\s*(\w+)', ' '.join(params1)))
        type2 = set(re.findall(r':\s*(\w+)', ' '.join(params2)))
        
        if not type1 and not type2:
            return 1.0
        if not type1 or not type2:
            return 0.0
        
        intersection = len(type1 & type2)
        union = len(type1 | type2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_return_type_similarity(self, return_type1: Optional[str], return_type2: Optional[str]) -> float:
        """计算返回类型相似�?""
        if return_type1 is None and return_type2 is None:
            return 1.0
        if return_type1 is None or return_type2 is None:
            return 0.0
        
        return 1.0 if return_type1 == return_type2 else 0.0
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """计算名称相似�?""
        # 简单的编辑距离相似�?
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        distance = levenshtein_distance(name1.lower(), name2.lower())
        max_length = max(len(name1), len(name2))
        
        return 1.0 - (distance / max_length) if max_length > 0 else 0.0
    
    def _calculate_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """计算余弦相似�?""
        try:
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        except Exception as e:
            logger.error(f"计算余弦相似度失�? {e}")
            return 0.0
    
    def _build_api_description(self, api: APISignature) -> str:
        """构建API描述文本"""
        description_parts = [api.name]
        
        if api.parameters:
            description_parts.append(f"parameters: {', '.join(api.parameters)}")
        
        if api.return_type:
            description_parts.append(f"returns: {api.return_type}")
        
        if api.docstring:
            description_parts.append(api.docstring)
        
        return " ".join(description_parts)
    
    def _determine_similarity_type(self, 
                                 api1: APISignature, 
                                 api2: APISignature,
                                 semantic_similarity: float,
                                 structural_similarity: float) -> str:
        """确定相似性类�?""
        if api1.name == api2.name:
            return "exact"
        elif semantic_similarity >= self.config.semantic_similarity_threshold:
            return "semantic"
        elif structural_similarity >= self.config.structural_similarity_threshold:
            return "structural"
        else:
            return "hybrid"
    
    def _generate_similarity_explanation(self, 
                                       api1: APISignature, 
                                       api2: APISignature,
                                       semantic_similarity: float,
                                       structural_similarity: float) -> str:
        """生成相似性解�?""
        explanations = []
        
        if semantic_similarity >= self.config.semantic_similarity_threshold:
            explanations.append(f"语义相似度高 ({semantic_similarity:.3f})")
        
        if structural_similarity >= self.config.structural_similarity_threshold:
            explanations.append(f"结构相似度高 ({structural_similarity:.3f})")
        
        if api1.parameters == api2.parameters:
            explanations.append("参数完全相同")
        
        if api1.return_type == api2.return_type:
            explanations.append("返回类型相同")
        
        return "; ".join(explanations) if explanations else "相似性较�?
    
    def _build_similarity_groups(self, similarity_results: List[SimilarityResult]) -> List[List[APISignature]]:
        """构建相似性组"""
        # 使用并查集算法构建相似性组
        api_to_group = {}
        groups = []
        
        for result in similarity_results:
            api1, api2 = result.api1, result.api2
            
            if api1 not in api_to_group and api2 not in api_to_group:
                # 创建新组
                new_group = [api1, api2]
                groups.append(new_group)
                api_to_group[api1] = len(groups) - 1
                api_to_group[api2] = len(groups) - 1
            elif api1 in api_to_group and api2 not in api_to_group:
                # 将api2添加到api1的组
                group_idx = api_to_group[api1]
                groups[group_idx].append(api2)
                api_to_group[api2] = group_idx
            elif api2 in api_to_group and api1 not in api_to_group:
                # 将api1添加到api2的组
                group_idx = api_to_group[api2]
                groups[group_idx].append(api1)
                api_to_group[api1] = group_idx
            elif api1 in api_to_group and api2 in api_to_group and api_to_group[api1] != api_to_group[api2]:
                # 合并两个�?
                group1_idx = api_to_group[api1]
                group2_idx = api_to_group[api2]
                
                # 将group2的所有API添加到group1
                groups[group1_idx].extend(groups[group2_idx])
                
                # 更新所有API的组索引
                for api in groups[group2_idx]:
                    api_to_group[api] = group1_idx
                
                # 删除group2
                groups.pop(group2_idx)
                
                # 更新所有组索引
                for i, group in enumerate(groups):
                    for api in group:
                        api_to_group[api] = i
        
        return groups
    
    def _analyze_usage_patterns(self, apis: List[APISignature]) -> Dict[str, Any]:
        """分析使用模式"""
        patterns = {
            "total_apis": len(apis),
            "repo_distribution": Counter(api.repo_name for api in apis),
            "parameter_patterns": self._analyze_parameter_patterns(apis),
            "return_type_patterns": self._analyze_return_type_patterns(apis),
            "naming_patterns": self._analyze_naming_patterns(apis)
        }
        
        return patterns
    
    def _analyze_parameter_patterns(self, apis: List[APISignature]) -> Dict[str, Any]:
        """分析参数模式"""
        param_counts = [len(api.parameters) for api in apis]
        
        return {
            "average_parameter_count": np.mean(param_counts) if param_counts else 0,
            "max_parameter_count": max(param_counts) if param_counts else 0,
            "min_parameter_count": min(param_counts) if param_counts else 0,
            "parameter_count_distribution": Counter(param_counts)
        }
    
    def _analyze_return_type_patterns(self, apis: List[APISignature]) -> Dict[str, Any]:
        """分析返回类型模式"""
        return_types = [api.return_type for api in apis if api.return_type]
        
        return {
            "total_with_return_type": len(return_types),
            "return_type_distribution": Counter(return_types),
            "most_common_return_type": Counter(return_types).most_common(1)[0] if return_types else None
        }
    
    def _analyze_naming_patterns(self, apis: List[APISignature]) -> Dict[str, Any]:
        """分析命名模式"""
        names = [api.name for api in apis]
        
        # 分析命名前缀
        prefixes = [name.split('_')[0] for name in names if '_' in name]
        
        return {
            "total_apis": len(names),
            "unique_names": len(set(names)),
            "naming_prefixes": Counter(prefixes),
            "most_common_prefix": Counter(prefixes).most_common(1)[0] if prefixes else None
        }
    
    def _get_all_apis(self, target_repos: Optional[List[str]] = None) -> List[APISignature]:
        """获取所有API"""
        all_apis = []
        
        # 这里需要根据实际的多库搜索器接口调�?
        # 假设有一个方法可以获取所有API
        repos_to_search = target_repos or list(self.multi_repo_search.repo_searchers.keys())
        
        for repo_name in repos_to_search:
            if repo_name in self.multi_repo_search.repo_searchers:
                # 这里需要根据实际的搜索器接口调�?
                # 假设可以搜索所有函�?
                try:
                    searcher = self.multi_repo_search.repo_searchers[repo_name]
                    # 搜索所有函�?
                    results = searcher.search_by_natural_language(
                        query="function definition",
                        limit=self.config.max_apis_per_repo,
                        search_type="keyword"
                    )
                    
                    for result in results:
                        api_signature = self._extract_api_signature(result, repo_name)
                        if api_signature:
                            all_apis.append(api_signature)
                            
                except Exception as e:
                    logger.error(f"获取仓库 {repo_name} 的API失败: {e}")
        
        return all_apis
    
    def _parallel_duplicate_detection(self, apis: List[APISignature], threshold: float) -> List[SimilarityResult]:
        """并行重复检�?""
        duplicate_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # 提交比较任务
            futures = []
            for i in range(len(apis)):
                for j in range(i + 1, len(apis)):
                    future = executor.submit(self._compare_apis, apis[i], apis[j], "comprehensive")
                    futures.append(future)
            
            # 收集结果
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result.overall_similarity >= threshold:
                        duplicate_results.append(result)
                except Exception as e:
                    logger.error(f"并行重复检测失�? {e}")
        
        return duplicate_results
    
    def _sequential_duplicate_detection(self, apis: List[APISignature], threshold: float) -> List[SimilarityResult]:
        """顺序重复检�?""
        duplicate_results = []
        
        for i in range(len(apis)):
            for j in range(i + 1, len(apis)):
                try:
                    result = self._compare_apis(apis[i], apis[j], "comprehensive")
                    if result.overall_similarity >= threshold:
                        duplicate_results.append(result)
                except Exception as e:
                    logger.error(f"比较API失败: {e}")
        
        return duplicate_results
    
    def _analyze_parameter_evolution(self, apis: List[APISignature]) -> Dict[str, Any]:
        """分析参数演化"""
        param_evolution = {
            "parameter_count_evolution": [len(api.parameters) for api in apis],
            "parameter_type_evolution": [api.parameters for api in apis],
            "evolution_patterns": []
        }
        
        # 分析演化模式
        if len(apis) > 1:
            # 按仓库分组分�?
            repo_groups = defaultdict(list)
            for api in apis:
                repo_groups[api.repo_name].append(api)
            
            for repo_name, repo_apis in repo_groups.items():
                if len(repo_apis) > 1:
                    # 分析同一仓库内的演化
                    param_counts = [len(api.parameters) for api in repo_apis]
                    if len(set(param_counts)) > 1:
                        param_evolution["evolution_patterns"].append({
                            "repo": repo_name,
                            "parameter_count_changes": param_counts,
                            "has_evolution": True
                        })
        
        return param_evolution
    
    def _analyze_return_type_evolution(self, apis: List[APISignature]) -> Dict[str, Any]:
        """分析返回类型演化"""
        return_evolution = {
            "return_type_evolution": [api.return_type for api in apis],
            "evolution_patterns": []
        }
        
        # 分析演化模式
        if len(apis) > 1:
            return_types = [api.return_type for api in apis if api.return_type]
            if len(set(return_types)) > 1:
                return_evolution["evolution_patterns"].append({
                    "has_return_type_evolution": True,
                    "return_types": return_types
                })
        
        return return_evolution
    
    def _analyze_docstring_evolution(self, apis: List[APISignature]) -> Dict[str, Any]:
        """分析文档字符串演�?""
        docstring_evolution = {
            "docstring_evolution": [api.docstring for api in apis if api.docstring],
            "evolution_patterns": []
        }
        
        # 分析演化模式
        if len(apis) > 1:
            docstrings = [api.docstring for api in apis if api.docstring]
            if len(docstrings) > 1:
                docstring_evolution["evolution_patterns"].append({
                    "has_docstring_evolution": True,
                    "docstring_count": len(docstrings)
                })
        
        return docstring_evolution
    
    def _analyze_usage_frequency(self, apis: List[APISignature]) -> Dict[str, Any]:
        """分析使用频率"""
        usage_frequency = {
            "repo_usage_frequency": Counter(api.repo_name for api in apis),
            "total_usage_count": len(apis),
            "most_used_repo": Counter(api.repo_name for api in apis).most_common(1)[0] if apis else None
        }
        
        return usage_frequency
    
    def _find_api_by_name(self, api_name: str, similarity_groups: List[List[APISignature]]) -> Optional[APISignature]:
        """根据名称查找API"""
        for group in similarity_groups:
            for api in group:
                if api.name == api_name:
                    return api
        return None
    
    def _calculate_api_similarity(self, api1: Optional[APISignature], api2: APISignature) -> float:
        """计算API相似�?""
        if not api1:
            return 0.0
        
        result = self._compare_apis(api1, api2, "comprehensive")
        return result.overall_similarity
    
    def _get_cache_key(self, api_name: str, similarity_threshold: float, analysis_type: str) -> str:
        """生成缓存�?""
        import hashlib
        key_data = f"{api_name}_{similarity_threshold}_{analysis_type}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _record_analysis(self, api_name: str, result: APIAnalysisResult):
        """记录分析历史"""
        analysis_record = {
            "timestamp": json.dumps({"timestamp": "now"}),
            "api_name": api_name,
            "total_occurrences": result.total_occurrences,
            "repos_containing": len(result.repos_containing),
            "similarity_groups": len(result.similarity_groups)
        }
        self.analysis_history.append(analysis_record)
        
        # 限制历史记录数量
        if len(self.analysis_history) > 100:
            self.analysis_history = self.analysis_history[-100:]
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """获取分析历史"""
        return self.analysis_history.copy()
    
    def clear_cache(self):
        """清空缓存"""
        with self.cache_lock:
            self.analysis_cache.clear()
        logger.info("API相似性分析缓存已清空")

def create_api_similarity_analyzer(multi_repo_search: MultiRepoSearch,
                                 config: APISimilarityConfig = None) -> APISimilarityAnalyzer:
    """创建API相似性分析器实例"""
    return APISimilarityAnalyzer(multi_repo_search, config)
