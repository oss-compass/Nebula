#!/usr/bin/env python3
"""
图数据库索引器
将向量索引的数据同步到 Neo4j 图数据库，建立代码结构的关系图谱
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
import re

# 尝试导入Neo4j
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

from .vector_embedding import CodeEmbedding

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GraphIndexerConfig:
    """图数据库索引器配置"""
    # Neo4j 配置
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "90879449Drq"
    neo4j_database: Optional[str] = None
    
    # 索引配置
    batch_size: int = 100
    enable_relationships: bool = True
    enable_complexity_analysis: bool = True
    
    # 关系类型配置
    relationship_types: List[str] = None
    
    def __post_init__(self):
        if self.relationship_types is None:
            self.relationship_types = [
                "CALLS",           # 函数调用关系
                "HAS_METHOD",      # 类包含方法
                "DECLARES",        # 文件声明函数/类
                "IMPORTS",         # 导入关系
                "INHERITS",        # 继承关系
                "IMPLEMENTS"       # 实现关系
            ]


class GraphIndexer:
    """图数据库索引器"""
    
    def __init__(self, config: GraphIndexerConfig):
        self.config = config
        self.driver = None
        self.indexed_functions = set()  # 已索引的函数集合
        self.indexed_classes = set()    # 已索引的类集合
        self.indexed_files = set()      # 已索引的文件集合
        
        if NEO4J_AVAILABLE:
            self._init_neo4j()
        else:
            logger.warning("Neo4j not available, graph indexing will be disabled")
    
    def _init_neo4j(self):
        """初始化Neo4j连接"""
        try:
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
            # 测试连接
            with self.driver.session(database=self.config.neo4j_database) as session:
                session.run("RETURN 1")
            logger.info("Neo4j connection established for graph indexing")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None
    
    def index_embeddings(self, embeddings: List[CodeEmbedding], indexed_files: Dict[str, Any]):
        """将向量嵌入索引到图数据库"""
        if not self.driver:
            logger.warning("Neo4j not available, skipping graph indexing")
            return
        
        logger.info(f"Starting graph indexing for {len(embeddings)} embeddings")
        
        with self.driver.session(database=self.config.neo4j_database) as session:
            # 创建索引和约束
            self._create_indexes_and_constraints(session)
            
            # 批量处理嵌入
            batch_count = 0
            for i in range(0, len(embeddings), self.config.batch_size):
                batch = embeddings[i:i + self.config.batch_size]
                self._index_embedding_batch(session, batch, indexed_files)
                batch_count += 1
                
                if batch_count % 10 == 0:
                    logger.info(f"Indexed {i + len(batch)} embeddings...")
            
            # 建立关系
            if self.config.enable_relationships:
                self._build_relationships(session, embeddings, indexed_files)
            
            logger.info(f"Graph indexing completed for {len(embeddings)} embeddings")
    
    def _create_indexes_and_constraints(self, session):
        """创建索引和约束"""
        try:
            # 创建函数节点索引
            session.run("CREATE INDEX function_name_index IF NOT EXISTS FOR (f:Function) ON (f.name)")
            session.run("CREATE INDEX function_filepath_index IF NOT EXISTS FOR (f:Function) ON (f.filepath)")
            
            # 创建类节点索引
            session.run("CREATE INDEX class_name_index IF NOT EXISTS FOR (c:Class) ON (c.name)")
            session.run("CREATE INDEX class_filepath_index IF NOT EXISTS FOR (c:Class) ON (c.filepath)")
            
            # 创建文件节点索引
            session.run("CREATE INDEX file_path_index IF NOT EXISTS FOR (f:File) ON (f.path)")
            
            # 创建唯一约束
            session.run("CREATE CONSTRAINT function_unique IF NOT EXISTS FOR (f:Function) REQUIRE (f.name, f.filepath) IS UNIQUE")
            session.run("CREATE CONSTRAINT class_unique IF NOT EXISTS FOR (c:Class) REQUIRE (c.name, c.filepath) IS UNIQUE")
            session.run("CREATE CONSTRAINT file_unique IF NOT EXISTS FOR (f:File) REQUIRE f.path IS UNIQUE")
            
            logger.info("Created indexes and constraints")
        except Exception as e:
            logger.warning(f"Failed to create some indexes/constraints: {e}")
    
    def _index_embedding_batch(self, session, embeddings: List[CodeEmbedding], indexed_files: Dict[str, Any]):
        """批量索引嵌入数据"""
        for embedding in embeddings:
            try:
                metadata = embedding.metadata
                function_name = metadata.get('name', 'unknown')
                file_path = metadata.get('filepath', 'unknown')
                
                # 跳过已索引的函数
                func_key = (function_name, file_path)
                if func_key in self.indexed_functions:
                    continue
                
                # 获取文件信息
                indexed_file = indexed_files.get(file_path)
                
                # 创建函数节点
                self._create_function_node(session, embedding, indexed_file)
                
                # 创建文件节点
                if file_path not in self.indexed_files:
                    self._create_file_node(session, file_path, indexed_file)
                    self.indexed_files.add(file_path)
                
                # 创建类节点（如果有父类）
                parent_class = metadata.get('parent_class_name')
                if parent_class and (parent_class, file_path) not in self.indexed_classes:
                    self._create_class_node(session, parent_class, file_path, indexed_file)
                    self.indexed_classes.add((parent_class, file_path))
                
                self.indexed_functions.add(func_key)
                
            except Exception as e:
                logger.error(f"Failed to index embedding {embedding.id}: {e}")
    
    def _create_function_node(self, session, embedding: CodeEmbedding, indexed_file: Any = None):
        """创建函数节点"""
        metadata = embedding.metadata
        
        # 计算复杂度分数
        complexity_score = self._calculate_complexity_score(embedding.content)
        
        # 计算重要性分数
        importance_score = self._calculate_importance_score(metadata, embedding.content)
        
        # 确定函数类型
        function_type = self._determine_function_type(metadata, embedding.content)
        
        query = """
        MERGE (f:Function {name: $name, filepath: $filepath})
        SET f.content = $content,
            f.start_line = $start_line,
            f.end_line = $end_line,
            f.language = $language,
            f.function_type = $function_type,
            f.complexity_score = $complexity_score,
            f.importance_score = $importance_score,
            f.is_async = $is_async,
            f.is_test = $is_test,
            f.is_decorator = $is_decorator,
            f.parameters_count = $parameters_count,
            f.return_type = $return_type,
            f.docstring = $docstring,
            f.parent_class_name = $parent_class_name,
            f.indexed_at = $indexed_at
        """
        
        params = {
            'name': metadata.get('name', 'unknown'),
            'filepath': metadata.get('filepath', 'unknown'),
            'content': embedding.content,
            'start_line': metadata.get('start_line', 0),
            'end_line': metadata.get('end_line', 0),
            'language': metadata.get('language', 'unknown'),
            'function_type': function_type,
            'complexity_score': complexity_score,
            'importance_score': importance_score,
            'is_async': self._is_async_function(embedding.content),
            'is_test': self._is_test_function(metadata.get('name', ''), embedding.content),
            'is_decorator': self._is_decorator_function(embedding.content),
            'parameters_count': self._count_parameters(embedding.content),
            'return_type': self._extract_return_type(embedding.content),
            'docstring': self._extract_docstring(embedding.content),
            'parent_class_name': metadata.get('parent_class_name'),
            'indexed_at': datetime.now().isoformat()
        }
        
        session.run(query, params)
    
    def _create_file_node(self, session, file_path: str, indexed_file: Any = None):
        """创建文件节点"""
        if not indexed_file:
            return
        
        query = """
        MERGE (f:File {path: $path})
        SET f.language = $language,
            f.file_size = $file_size,
            f.line_count = $line_count,
            f.functions_count = $functions_count,
            f.classes_count = $classes_count,
            f.imports = $imports,
            f.indexed_at = $indexed_at
        """
        
        params = {
            'path': file_path,
            'language': indexed_file.language,
            'file_size': indexed_file.file_size,
            'line_count': indexed_file.metadata.get('line_count', 0),
            'functions_count': len(indexed_file.functions),
            'classes_count': len(indexed_file.classes),
            'imports': json.dumps(indexed_file.imports),
            'indexed_at': datetime.now().isoformat()
        }
        
        session.run(query, params)
    
    def _create_class_node(self, session, class_name: str, file_path: str, indexed_file: Any = None):
        """创建类节点"""
        query = """
        MERGE (c:Class {name: $name, filepath: $filepath})
        SET c.file_path = $file_path,
            c.language = $language,
            c.indexed_at = $indexed_at
        """
        
        params = {
            'name': class_name,
            'filepath': file_path,
            'file_path': file_path,
            'language': indexed_file.language if indexed_file else 'unknown',
            'indexed_at': datetime.now().isoformat()
        }
        
        session.run(query, params)
    
    def _build_relationships(self, session, embeddings: List[CodeEmbedding], indexed_files: Dict[str, Any]):
        """建立节点之间的关系"""
        logger.info("Building relationships between nodes...")
        
        # 建立文件-函数关系
        self._build_file_function_relationships(session)
        
        # 建立类-方法关系
        self._build_class_method_relationships(session)
        
        # 建立函数调用关系
        self._build_function_call_relationships(session, embeddings)
        
        # 建立导入关系
        self._build_import_relationships(session, indexed_files)
        
        logger.info("Relationship building completed")
    
    def _build_file_function_relationships(self, session):
        """建立文件-函数关系"""
        query = """
        MATCH (file:File), (func:Function)
        WHERE file.path = func.filepath
        MERGE (file)-[:DECLARES]->(func)
        """
        session.run(query)
    
    def _build_class_method_relationships(self, session):
        """建立类-方法关系"""
        query = """
        MATCH (cls:Class), (func:Function)
        WHERE cls.name = func.parent_class_name AND cls.filepath = func.filepath
        MERGE (cls)-[:HAS_METHOD]->(func)
        """
        session.run(query)
    
    def _build_function_call_relationships(self, session, embeddings: List[CodeEmbedding]):
        """建立函数调用关系"""
        for embedding in embeddings:
            try:
                function_name = embedding.metadata.get('name', '')
                file_path = embedding.metadata.get('filepath', '')
                content = embedding.content
                
                # 提取函数调用
                called_functions = self._extract_function_calls(content, embeddings)
                
                for called_func in called_functions:
                    query = """
                    MATCH (caller:Function {name: $caller_name, filepath: $caller_filepath})
                    MATCH (callee:Function {name: $callee_name})
                    WHERE callee.filepath = $caller_filepath OR callee.filepath CONTAINS $callee_name
                    MERGE (caller)-[:CALLS]->(callee)
                    """
                    
                    session.run(query, {
                        'caller_name': function_name,
                        'caller_filepath': file_path,
                        'callee_name': called_func
                    })
                    
            except Exception as e:
                logger.error(f"Failed to build call relationships for {embedding.id}: {e}")
    
    def _build_import_relationships(self, session, indexed_files: Dict[str, Any]):
        """建立导入关系"""
        for file_path, indexed_file in indexed_files.items():
            try:
                for import_stmt in indexed_file.imports:
                    # 简化的导入关系建立
                    query = """
                    MATCH (file:File {path: $file_path})
                    MERGE (file)-[:IMPORTS {module: $import_stmt}]->(imported:Module {name: $import_stmt})
                    """
                    session.run(query, {
                        'file_path': file_path,
                        'import_stmt': import_stmt
                    })
            except Exception as e:
                logger.error(f"Failed to build import relationships for {file_path}: {e}")
    
    def _calculate_complexity_score(self, content: str) -> float:
        """计算复杂度分数"""
        score = 0.0
        
        # 基于代码行数
        lines = content.split('\n')
        score += min(len(lines) / 10.0, 2.0)
        
        # 基于控制流结构
        score += content.count('if ') * 0.5
        score += content.count('for ') * 0.5
        score += content.count('while ') * 0.5
        score += content.count('try:') * 0.3
        score += content.count('except') * 0.3
        
        # 基于嵌套层级
        max_indent = 0
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent)
        score += min(max_indent / 4.0, 2.0)
        
        return round(score, 2)
    
    def _calculate_importance_score(self, metadata: Dict[str, Any], content: str) -> float:
        """计算重要性分数"""
        score = 0.0
        
        # 基于函数名
        function_name = metadata.get('name', '').lower()
        if any(keyword in function_name for keyword in ['main', 'init', 'setup', 'config']):
            score += 2.0
        elif any(keyword in function_name for keyword in ['get', 'set', 'create', 'build']):
            score += 1.5
        elif any(keyword in function_name for keyword in ['test', 'check', 'validate']):
            score += 1.0
        
        # 基于文档字符串
        if '"""' in content or "'''" in content:
            score += 1.0
        
        # 基于参数数量
        param_count = self._count_parameters(content)
        if param_count > 5:
            score += 1.0
        elif param_count > 2:
            score += 0.5
        
        # 基于是否为主函数
        if 'if __name__ == "__main__"' in content:
            score += 2.0
        
        return round(score, 2)
    
    def _determine_function_type(self, metadata: Dict[str, Any], content: str) -> str:
        """确定函数类型"""
        function_name = metadata.get('name', '').lower()
        
        if 'test' in function_name or 'test_' in function_name:
            return 'test'
        elif 'async' in content or 'await' in content:
            return 'async'
        elif '@' in content and 'def' in content:
            return 'decorator'
        elif 'class' in content and 'def' in content:
            return 'method'
        else:
            return 'regular'
    
    def _is_async_function(self, content: str) -> bool:
        """判断是否为异步函数"""
        return 'async def' in content or 'await' in content
    
    def _is_test_function(self, function_name: str, content: str) -> bool:
        """判断是否为测试函数"""
        return (function_name.startswith('test_') or 
                function_name.endswith('_test') or 
                'assert' in content or 
                'unittest' in content)
    
    def _is_decorator_function(self, content: str) -> bool:
        """判断是否为装饰器函数"""
        return '@' in content and 'def' in content
    
    def _count_parameters(self, content: str) -> int:
        """计算参数数量"""
        # 简化的参数计数
        def_match = re.search(r'def\s+\w+\s*\(([^)]*)\)', content)
        if def_match:
            params_str = def_match.group(1)
            if params_str.strip():
                return len([p.strip() for p in params_str.split(',') if p.strip()])
        return 0
    
    def _extract_return_type(self, content: str) -> str:
        """提取返回类型"""
        # 简化的返回类型提取
        if '->' in content:
            match = re.search(r'->\s*(\w+)', content)
            if match:
                return match.group(1)
        return ''
    
    def _extract_docstring(self, content: str) -> str:
        """提取文档字符串"""
        # 简化的文档字符串提取
        if '"""' in content:
            match = re.search(r'"""([^"]*)"""', content, re.DOTALL)
            if match:
                return match.group(1).strip()
        elif "'''" in content:
            match = re.search(r"'''([^']*)'''", content, re.DOTALL)
            if match:
                return match.group(1).strip()
        return ''
    
    def _extract_function_calls(self, content: str, all_embeddings: List[CodeEmbedding]) -> List[str]:
        """提取函数调用"""
        called_functions = []
        
        # 获取所有函数名
        all_function_names = {emb.metadata.get('name', '') for emb in all_embeddings}
        
        # 简化的函数调用提取
        for func_name in all_function_names:
            if func_name and func_name in content and f'def {func_name}' not in content:
                called_functions.append(func_name)
        
        return called_functions
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """获取图数据库统计信息"""
        if not self.driver:
            return {"error": "Graph database not available"}
        
        try:
            with self.driver.session(database=self.config.neo4j_database) as session:
                # 基本统计
                result = session.run("""
                    MATCH (f:Function)
                    RETURN count(f) as total_functions,
                           avg(f.complexity_score) as avg_complexity,
                           max(f.complexity_score) as max_complexity,
                           min(f.complexity_score) as min_complexity
                """)
                
                stats = dict(result.single())
                
                # 关系统计
                result = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) as rel_type, count(r) as count
                    ORDER BY count DESC
                """)
                
                relationships = {record["rel_type"]: record["count"] for record in result}
                stats["relationships"] = relationships
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting graph stats: {e}")
            return {"error": str(e)}
    
    def clear_graph(self):
        """清空图数据库"""
        if not self.driver:
            return
        
        try:
            with self.driver.session(database=self.config.neo4j_database) as session:
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("Graph database cleared")
        except Exception as e:
            logger.error(f"Failed to clear graph database: {e}")
    
    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()


def create_default_graph_config() -> GraphIndexerConfig:
    """创建默认图索引器配置"""
    return GraphIndexerConfig(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="90879449Drq",
        neo4j_database=None,
        batch_size=100,
        enable_relationships=True,
        enable_complexity_analysis=True
    )


if __name__ == "__main__":
    # 测试代码
    from .vector_embedding import CodeEmbedding
    import numpy as np
    
    # 创建测试数据
    test_embedding = CodeEmbedding(
        id="test_func_1",
        content="def test_function(x):\n    return x + 1",
        embedding=np.random.rand(384),
        metadata={
            'name': 'test_function',
            'filepath': 'test.py',
            'start_line': 1,
            'end_line': 2,
            'language': 'python',
            'function_type': 'regular'
        },
        model_name="test_model",
        timestamp=datetime.now().isoformat()
    )
    
    test_indexed_file = {
        'file_path': "test.py",
        'content_hash': "test_hash",
        'last_modified': datetime.now().isoformat(),
        'file_size': 100,
        'language': "python",
        'functions': [{'name': 'test_function', 'line': 1}],
        'classes': [],
        'imports': []
    }
    
    # 测试图索引器
    config = create_default_graph_config()
    indexer = GraphIndexer(config)
    
    try:
        # 清空现有数据
        indexer.clear_graph()
        
        # 索引测试数据
        indexer.index_embeddings([test_embedding], {"test.py": test_indexed_file})
        
        # 获取统计信息
        stats = indexer.get_graph_stats()
        print(f"Graph stats: {stats}")
        
    finally:
        indexer.close()
