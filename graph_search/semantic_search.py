import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available, semantic search will be disabled")

try:
    import jieba
    import jieba.analyse
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    logging.warning("jieba not available, keyword extraction will be limited")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("openai not available, AI-enhanced search will be disabled")

from .base import BaseSearch
from .config import config

logger = logging.getLogger(__name__)

class SemanticSearch(BaseSearch):
    def __init__(self, 
                 neo4j_uri: Optional[str] = None,
                 neo4j_user: Optional[str] = None,
                 neo4j_password: Optional[str] = None,
                 neo4j_database: Optional[str] = None,
                 embedding_model: Optional[str] = None,
                 openai_api_key: Optional[str] = None):
        """
        初始化语义搜索类
        
        Args:
            neo4j_uri: Neo4j数据库URI
            neo4j_user: Neo4j用户名
            neo4j_password: Neo4j密码
            neo4j_database: 数据库名称
            embedding_model: 嵌入模型名称
            openai_api_key: OpenAI API密钥
        """
        super().__init__(neo4j_uri, neo4j_user, neo4j_password, neo4j_database)
        
        self.embedding_model_name = embedding_model or config.embedding_model
        self._embedding_model = None
        self._embedding_cache = {}
        
        # OpenAI配置
        self.openai_api_key = openai_api_key or config.openai_api_key
        self.use_openai = bool(self.openai_api_key and OPENAI_AVAILABLE)
        if self.use_openai:
            openai.api_key = self.openai_api_key
        
        logger.info(f"语义搜索初始化完成，嵌入模型: {self.embedding_model_name}")
    
    @property
    def embedding_model(self):
        if self._embedding_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"正在加载嵌入模型: {self.embedding_model_name}")
                self._embedding_model = SentenceTransformer(
                    self.embedding_model_name,
                    device=config.embedding_device
                )
                logger.info("嵌入模型加载成功")
            except Exception as e:
                logger.error(f"无法加载嵌入模型: {e}")
                self._embedding_model = None
        return self._embedding_model
    
    def search_by_natural_language(self, 
                                  query: str, 
                                  limit: int = 10,
                                  search_type: str = "hybrid",
                                  similarity_threshold: float = None) -> List[Dict[str, Any]]:
        """
        使用自然语言搜索代码
        
        Args:
            query: 自然语言查询
            limit: 返回结果数量限制
            search_type: 搜索类型 ("semantic", "keyword", "hybrid", "ai")
            similarity_threshold: 相似度阈值
            
        Returns:
            搜索结果列表
        """
        logger.info(f"执行自然语言搜索: {query}, 类型: {search_type}")
        
        threshold = similarity_threshold or config.similarity_threshold
        
        if search_type == "semantic":
            return self._semantic_search(query, limit, threshold)
        elif search_type == "keyword":
            return self._keyword_search(query, limit)
        elif search_type == "hybrid":
            return self._hybrid_search(query, limit, threshold)
        elif search_type == "ai":
            return self._ai_enhanced_search(query, limit, threshold)
        else:
            raise ValueError(f"不支持的搜索类型: {search_type}")
    
    def _semantic_search(self, query: str, limit: int, threshold: float) -> List[Dict[str, Any]]:
        if self.embedding_model is None:
            logger.warning("嵌入模型不可用，回退到关键词搜索")
            return self._keyword_search(query, limit)
        
        try:
            # 生成查询嵌入
            query_embedding = self.embedding_model.encode(query)
            
            # 从Neo4j获取所有函数的文本描述
            functions = self._get_all_functions_with_descriptions()
            
            # 计算相似度
            results = []
            for func in functions:
                # 构建函数描述文本
                description_text = self._build_function_description(func)
                
                # 生成函数描述的嵌入
                func_embedding = self.embedding_model.encode(description_text)
                
                # 计算余弦相似度
                similarity = np.dot(query_embedding, func_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(func_embedding)
                )
                
                if similarity >= threshold:
                    results.append({
                        **func,
                        "similarity_score": float(similarity),
                        "description_text": description_text
                    })
            
            # 按相似度排序
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"语义搜索失败: {e}")
            logger.info("回退到关键词搜索")
            return self._keyword_search(query, limit)
    
    def _keyword_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        keywords = self._extract_keywords(query)
        
        if not keywords:
            return []
        
        keyword_conditions = []
        for keyword in keywords:
            keyword_conditions.append(f"""
                (fn.name CONTAINS '{keyword}' OR 
                 fn.docstring_description CONTAINS '{keyword}' OR
                 fn.context_summary CONTAINS '{keyword}' OR
                 fn.source_code CONTAINS '{keyword}')
            """)
        
        cypher_query = f"""
        MATCH (fn:Function)
        WHERE {' OR '.join(keyword_conditions)}
        RETURN fn
        ORDER BY fn.lines_of_code DESC
        LIMIT {limit}
        """
        
        result = self._run_query(cypher_query)
        return [dict(record["fn"]) for record in result]
    
    def _hybrid_search(self, query: str, limit: int, threshold: float) -> List[Dict[str, Any]]:
        if self.embedding_model is None:
            logger.warning("嵌入模型不可用，混合搜索回退到关键词搜索")
            return self._keyword_search(query, limit)
        
        semantic_results = self._semantic_search(query, limit * 2, threshold)
        keyword_results = self._keyword_search(query, limit * 2)

        all_results = {}

        for result in semantic_results:
            func_id = result.get("id")
            if func_id:
                all_results[func_id] = {
                    **result,
                    "semantic_score": result.get("similarity_score", 0),
                    "keyword_score": 0
                }

        for result in keyword_results:
            func_id = result.get("id")
            if func_id:
                if func_id in all_results:
                    all_results[func_id]["keyword_score"] = 1.0
                else:
                    all_results[func_id] = {
                        **result,
                        "semantic_score": 0,
                        "keyword_score": 1.0
                    }
        
        for result in all_results.values():
            result["combined_score"] = (
                result["semantic_score"] * 0.7 + 
                result["keyword_score"] * 0.3
            )

        sorted_results = sorted(
            all_results.values(), 
            key=lambda x: x["combined_score"], 
            reverse=True
        )
        
        return sorted_results[:limit]
    
    def _ai_enhanced_search(self, query: str, limit: int, threshold: float) -> List[Dict[str, Any]]:

        if not self.use_openai:
            logger.warning("OpenAI API未配置，回退到混合搜索")
            return self._hybrid_search(query, limit, threshold)
        
        try:
            hybrid_results = self._hybrid_search(query, limit * 2, threshold)
            
            enhanced_results = []
            
            for result in hybrid_results[:limit]:
                prompt = f"""
                用户查询: {query}
                
                函数信息:
                - 名称: {result.get('name', 'N/A')}
                - 描述: {result.get('docstring_description', 'N/A')}
                - 上下文: {result.get('context_summary', 'N/A')}
                - 复杂度: {result.get('complexity_level', 'N/A')}
                - 相似度分数: {result.get('similarity_score', result.get('combined_score', 0)):.3f}
                
                请分析这个函数与用户查询的相关性，并给出:
                1. 相关性评分 (0-10)
                2. 相关性解释 (50字以内)
                3. 推荐理由 (30字以内)
                
                请以JSON格式返回:
                {{
                    "relevance_score": 分数,
                    "relevance_explanation": "解释",
                    "recommendation_reason": "推荐理由"
                }}
                """
                
                response = openai.ChatCompletion.create(
                    model=config.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.3
                )
                
                try:
                    ai_analysis = json.loads(response.choices[0].message.content)
                    enhanced_result = {
                        **result,
                        "ai_relevance_score": ai_analysis.get("relevance_score", 0),
                        "ai_relevance_explanation": ai_analysis.get("relevance_explanation", ""),
                        "ai_recommendation_reason": ai_analysis.get("recommendation_reason", "")
                    }
                    enhanced_results.append(enhanced_result)
                except json.JSONDecodeError:
                    # 如果AI返回的不是有效JSON，使用原始结果
                    enhanced_results.append(result)
            
            # 按AI相关性评分重新排序
            enhanced_results.sort(key=lambda x: x.get("ai_relevance_score", 0), reverse=True)
            
            return enhanced_results[:limit]
            
        except Exception as e:
            logger.error(f"AI增强搜索失败: {e}")
            return self._hybrid_search(query, limit, threshold)
    
    def _get_all_functions_with_descriptions(self) -> List[Dict[str, Any]]:
        """获取所有函数及其描述信息"""
        query = """
        MATCH (fn:Function)
        RETURN fn
        """
        
        result = self._run_query(query)
        return [dict(record["fn"]) for record in result]
    
    def _build_function_description(self, func: Dict[str, Any]) -> str:
        """构建函数的描述文本"""
        parts = []
        
        # 函数名
        if func.get("name"):
            parts.append(f"函数名: {func['name']}")
        
        # 文档字符串
        if func.get("docstring_description"):
            parts.append(f"描述: {func['docstring_description']}")
        
        # 参数信息
        if func.get("docstring_args"):
            parts.append(f"参数: {func['docstring_args']}")
        
        # 返回值信息
        if func.get("docstring_returns"):
            parts.append(f"返回值: {func['docstring_returns']}")
        
        # 上下文摘要
        if func.get("context_summary"):
            parts.append(f"上下文: {func['context_summary']}")
        
        # 复杂度信息
        if func.get("complexity_level"):
            parts.append(f"复杂度: {func['complexity_level']}")
        
        # 源代码（前100个字符）
        if func.get("source_code"):
            source_preview = func['source_code'][:100] + "..." if len(func['source_code']) > 100 else func['source_code']
            parts.append(f"代码: {source_preview}")
        
        return " | ".join(parts)
    
    def _extract_keywords(self, query: str) -> List[str]:
        """提取查询中的关键词"""
        keywords = set()
        
        # 使用jieba提取关键词（如果可用）
        if JIEBA_AVAILABLE:
            try:
                jieba_keywords = jieba.analyse.extract_tags(query, topK=10)
                keywords.update(jieba_keywords)
            except Exception as e:
                logger.warning(f"jieba关键词提取失败: {e}")
        
        # 添加一些编程相关的关键词
        programming_keywords = [
            "function", "method", "class", "api", "call", "return", "parameter",
            "error", "exception", "file", "data", "process", "validate",
            "calculate", "generate", "create", "update", "delete", "list",
            "get", "set", "add", "remove", "find", "search", "sort", "filter",
            "函数", "方法", "类", "调用", "返回", "参数", "错误", "异常",
            "文件", "数据", "处理", "验证", "计算", "生成", "创建", "更新",
            "删除", "列表", "查找", "搜索", "排序", "过滤"
        ]
        
        # 检查是否包含编程关键词
        query_lower = query.lower()
        for keyword in programming_keywords:
            if keyword.lower() in query_lower:
                keywords.add(keyword)
        
        # 简单的英文单词提取
        english_words = re.findall(r'\b[a-zA-Z]+\b', query)
        keywords.update([word.lower() for word in english_words if len(word) > 2])
        
        return list(keywords)
    
    def search_by_complexity(self, 
                           complexity_level: Optional[str] = None,
                           min_lines: Optional[int] = None,
                           max_lines: Optional[int] = None,
                           min_complexity: Optional[float] = None,
                           max_complexity: Optional[float] = None,
                           limit: int = 100) -> List[Dict[str, Any]]:
        """
        按复杂度搜索函数
        
        Args:
            complexity_level: 复杂度级别 ("simple", "moderate", "complex")
            min_lines: 最小代码行数
            max_lines: 最大代码行数
            min_complexity: 最小圈复杂度
            max_complexity: 最大圈复杂度
            limit: 结果数量限制
            
        Returns:
            搜索结果列表
        """
        conditions = []
        
        if complexity_level:
            conditions.append(f"fn.complexity_level = '{complexity_level}'")
        
        if min_lines is not None:
            conditions.append(f"fn.lines_of_code >= {min_lines}")
        
        if max_lines is not None:
            conditions.append(f"fn.lines_of_code <= {max_lines}")
        
        if min_complexity is not None:
            conditions.append(f"fn.cyclomatic_complexity >= {min_complexity}")
        
        if max_complexity is not None:
            conditions.append(f"fn.cyclomatic_complexity <= {max_complexity}")
        
        if not conditions:
            conditions.append("1=1")  # 默认条件
        
        query = f"""
        MATCH (fn:Function)
        WHERE {' AND '.join(conditions)}
        RETURN fn
        ORDER BY fn.cyclomatic_complexity DESC
        LIMIT {limit}
        """
        
        result = self._run_query(query)
        return [dict(record["fn"]) for record in result]
    
    def search_similar_functions(self, 
                               function_name: str,
                               limit: int = 10,
                               similarity_threshold: float = None) -> List[Dict[str, Any]]:
        """
        查找相似函数
        
        Args:
            function_name: 参考函数名
            limit: 结果数量限制
            similarity_threshold: 相似度阈值
            
        Returns:
            相似函数列表
        """
        if self.embedding_model is None:
            logger.warning("嵌入模型不可用，无法进行相似函数搜索")
            return []
        
        # 获取参考函数
        ref_func_query = """
        MATCH (fn:Function {name: $function_name})
        RETURN fn
        """
        
        ref_result = self._run_query_single(ref_func_query, {"function_name": function_name})
        if not ref_result:
            logger.warning(f"未找到函数: {function_name}")
            return []
        
        ref_func = dict(ref_result["fn"])
        ref_description = self._build_function_description(ref_func)
        ref_embedding = self.embedding_model.encode(ref_description)
        
        # 获取所有其他函数
        all_funcs_query = """
        MATCH (fn:Function)
        WHERE fn.name <> $function_name
        RETURN fn
        """
        
        all_funcs = self._run_query(all_funcs_query, {"function_name": function_name})
        
        # 计算相似度
        threshold = similarity_threshold or config.similarity_threshold
        similar_funcs = []
        
        for record in all_funcs:
            func = dict(record["fn"])
            func_description = self._build_function_description(func)
            func_embedding = self.embedding_model.encode(func_description)
            
            similarity = np.dot(ref_embedding, func_embedding) / (
                np.linalg.norm(ref_embedding) * np.linalg.norm(func_embedding)
            )
            
            if similarity >= threshold:
                similar_funcs.append({
                    **func,
                    "similarity_score": float(similarity),
                    "reference_function": function_name
                })
        
        # 按相似度排序
        similar_funcs.sort(key=lambda x: x["similarity_score"], reverse=True)
        return similar_funcs[:limit]
