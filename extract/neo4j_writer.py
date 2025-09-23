from typing import Dict, List, Any, Optional
from neo4j import GraphDatabase
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jWriter:
    """Neo4j图数据库写入�?""
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """
        初始化Neo4j连接
        
        Args:
            uri: Neo4j数据库URI (�? bolt://localhost:7687)
            username: 用户�?
            password: 密码
            database: 数据库名�?
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        self.session = None
        
    def __enter__(self):
        """上下文管理器入口"""
        self.session = self.driver.session(database=self.database)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        if self.session:
            self.session.close()
        self.driver.close()
    
    def clear_database(self):
        """清空数据库（可选，用于重新开始）"""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("数据库已清空")
    
    def write_analysis_result(self, result: Dict[str, Any]) -> None:
        """
        将分析结果写入Neo4j
        
        Args:
            result: 分析结果字典
        """
        logger.info(f"开始写入分析结果到Neo4j数据�? {self.database}")
        
        # 验证数据结构
        self._validate_data_structure(result)
        
        # 1. 创建仓库节点
        self._create_repository_node(result)
        
        # 2. 创建文件节点
        self._create_file_nodes(result)
        
        # 3. 创建类节�?
        self._create_class_nodes(result)
        
        # 4. 创建函数节点
        self._create_function_nodes(result)
        
        # 5. 创建函数调用关系
        self._create_call_relationships(result)
        
        # 6. 创建传递调用关�?
        self._create_transitive_relationships(result)
        
        # 7. 创建依赖关系
        self._create_dependency_relationships(result)
        
        logger.info("分析结果写入完成")
    
    def _validate_data_structure(self, result: Dict[str, Any]) -> None:
        """验证数据结构"""
        logger.info("验证数据结构...")
        
        # 检查基本字�?
        required_fields = ["repository", "functions"]
        for field in required_fields:
            if field not in result:
                logger.error(f"缺少必需字段: {field}")
                raise ValueError(f"缺少必需字段: {field}")
        
        # 检查函数数�?
        functions = result.get("functions", [])
        logger.info(f"数据验证: 找到 {len(functions)} 个函�?)
        
        if functions:
            # 检查第一个函数的数据结构
            first_func = functions[0]
            logger.info(f"第一个函数的数据结构: {list(first_func.keys())}")
            
            # 检查基本字�?
            basic_info = first_func.get("basic_info", {})
            logger.info(f"basic_info字段: {list(basic_info.keys())}")
            
            context = first_func.get("context", {})
            logger.info(f"context字段: {list(context.keys())}")
            
            complexity = first_func.get("complexity", {})
            logger.info(f"complexity字段: {list(complexity.keys())}")
            
            importance = first_func.get("importance", {})
            logger.info(f"importance字段: {list(importance.keys())}")
        
        logger.info("数据结构验证完成")
    
    def _convert_to_neo4j_compatible(self, data: Any) -> Any:
        """
        将数据转换为Neo4j兼容的格�?
        
        Args:
            data: 要转换的数据
            
        Returns:
            Neo4j兼容的数据格�?
        """
        if data is None:
            return None
        elif isinstance(data, (str, int, float, bool)):
            return data
        elif isinstance(data, list):
            # 处理列表
            converted_list = []
            for item in data:
                if isinstance(item, dict):
                    # 将字典转换为字符�?
                    converted_list.append(str(item))
                elif isinstance(item, (str, int, float, bool)):
                    converted_list.append(item)
                else:
                    converted_list.append(str(item))
            return converted_list
        elif isinstance(data, dict):
            # 将字典转换为字符�?
            return str(data)
        else:
            # 其他类型转换为字符串
            return str(data)
    
    def _create_repository_node(self, result: Dict[str, Any]) -> None:
        """创建仓库节点"""
        repo_name = result.get("repository", "unknown")
        analysis_timestamp = result.get("analysis_timestamp", datetime.now().isoformat())
        total_functions = result.get("total_functions", 0)
        
        query = """
        CREATE (r:Repository {
            name: $name,
            analysis_timestamp: $timestamp,
            total_functions: $total_functions,
            created_at: datetime()
        })
        """
        
        self.session.run(query, {
            "name": repo_name,
            "timestamp": analysis_timestamp,
            "total_functions": total_functions
        })
        
        logger.info(f"创建仓库节点: {repo_name}")
    
    def _create_file_nodes(self, result: Dict[str, Any]) -> None:
        """创建文件节点"""
        functions = result.get("functions", [])
        file_paths = set()
        
        logger.info(f"开始创建文件节点，分析 {len(functions)} 个函�?)
        
        # 收集所有文件路�?
        for func in functions:
            file_path = func.get("context", {}).get("file_path")
            if file_path:
                file_paths.add(file_path)
        
        logger.info(f"找到 {len(file_paths)} 个文�?)
        
        if not file_paths:
            logger.warning("没有找到文件路径数据，跳过文件节点创�?)
            return
        
        # 创建文件节点
        for file_path in file_paths:
            try:
                query = """
                MATCH (r:Repository {name: $repo_name})
                CREATE (f:File {
                    path: $path,
                    created_at: datetime()
                })
                CREATE (f)-[:BELONGS_TO]->(r)
                """
                
                self.session.run(query, {
                    "repo_name": result.get("repository", "unknown"),
                    "path": file_path
                })
                
            except Exception as e:
                logger.error(f"创建文件节点失败 (文件路径: {file_path}): {e}")
                continue
        
        logger.info(f"创建�?{len(file_paths)} 个文件节�?)
    
    def _create_class_nodes(self, result: Dict[str, Any]) -> None:
        """创建类节�?""
        functions = result.get("functions", [])
        classes = set()
        
        logger.info(f"开始创建类节点，分�?{len(functions)} 个函�?)
        
        # 收集所有类�?
        for func in functions:
            parent_class = func.get("context", {}).get("parent_class")
            if parent_class:
                classes.add(parent_class)
        
        logger.info(f"找到 {len(classes)} 个类: {list(classes)}")
        
        if not classes:
            logger.warning("没有找到类数据，跳过类节点创�?)
            return
        
        # 创建类节�?
        for class_name in classes:
            try:
                query = """
                MATCH (r:Repository {name: $repo_name})
                CREATE (c:Class {
                    name: $name,
                    created_at: datetime()
                })
                CREATE (c)-[:BELONGS_TO]->(r)
                """
                
                self.session.run(query, {
                    "repo_name": result.get("repository", "unknown"),
                    "name": class_name
                })
                
            except Exception as e:
                logger.error(f"创建类节点失�?(类名: {class_name}): {e}")
                continue
        
        logger.info(f"创建�?{len(classes)} 个类节点")
    
    def _create_function_nodes(self, result: Dict[str, Any]) -> None:
        """创建函数节点"""
        functions = result.get("functions", [])
        repo_name = result.get("repository", "unknown")
        
        logger.info(f"开始创建函数节点，�?{len(functions)} 个函�?)
        
        if not functions:
            logger.warning("没有找到函数数据，跳过函数节点创�?)
            return
        
        for i, func in enumerate(functions):
            try:
                basic_info = func.get("basic_info", {})
                complexity = func.get("complexity", {})
                context = func.get("context", {})
                importance = func.get("importance", {})
                
                # 提取复杂度信�?
                semantic_complexity = complexity.get("semantic_complexity", {})
                syntactic_complexity = complexity.get("syntactic_complexity", {})
                structural_complexity = complexity.get("structural_complexity", {})
                
                # 提取重要度信�?
                total_score = importance.get("total_score", 0)
                importance_level = importance.get("importance_level", "Unknown")
                
                query = """
                MATCH (r:Repository {name: $repo_name})
                CREATE (func:Function {
                    name: $name,
                    source_code: $source_code,
                    start_line: $start_line,
                    end_line: $end_line,
                    start_column: $start_column,
                    end_column: $end_column,
                    comments: $comments,
                    parameters: $parameters,
                    return_type: $return_type,
                    parent_class: $parent_class,
                    file_path: $file_path,
                    imports: $imports,
                    internal_calls: $internal_calls,
                    external_calls: $external_calls,
                    cyclomatic_complexity: $cyclomatic_complexity,
                    lines_of_code: $lines_of_code,
                    parameters_count: $parameters_count,
                    token_count: $token_count,
                    complexity_score: $complexity_score,
                    return_types: $return_types,
                    external_dependencies: $external_dependencies,
                    syntax_depth: $syntax_depth,
                    branch_count: $branch_count,
                    total_score: $total_score,
                    importance_level: $importance_level,
                    is_hub: $is_hub,
                    is_leaf: $is_leaf,
                    is_coordinator: $is_coordinator,
                    is_foundation: $is_foundation,
                    created_at: datetime()
                })
                CREATE (func)-[:BELONGS_TO]->(r)
                """
                
                # 准备参数
                code_location = basic_info.get("code_location", {})
                metrics = importance.get("metrics", {})
                
                # 处理可能包含字典的列表字段，转换为Neo4j兼容的格�?
                comments = basic_info.get("comments", [])
                parameters = basic_info.get("parameters", [])
                imports = context.get("imports", [])
                internal_calls = context.get("function_calls", {}).get("internal_calls", [])
                external_calls = context.get("function_calls", {}).get("external_calls", [])
                
                # 将复杂数据结构转换为字符�?
                comments_str = self._convert_to_neo4j_compatible(comments)
                parameters_str = self._convert_to_neo4j_compatible(parameters)
                imports_str = self._convert_to_neo4j_compatible(imports)
                internal_calls_str = self._convert_to_neo4j_compatible(internal_calls)
                external_calls_str = self._convert_to_neo4j_compatible(external_calls)
                
                params = {
                    "repo_name": repo_name,
                    "name": basic_info.get("function_name", ""),
                    "source_code": basic_info.get("source_code", ""),
                    "start_line": code_location.get("start_line", 0),
                    "end_line": code_location.get("end_line", 0),
                    "start_column": code_location.get("start_column", 0),
                    "end_column": code_location.get("end_column", 0),
                    "comments": comments_str,
                    "parameters": parameters_str,
                    "return_type": basic_info.get("return_type", ""),
                    "parent_class": context.get("parent_class", ""),
                    "file_path": context.get("file_path", ""),
                    "imports": imports_str,
                    "internal_calls": internal_calls_str,
                    "external_calls": external_calls_str,
                    "cyclomatic_complexity": semantic_complexity.get("cyclomatic_complexity", 0),
                    "lines_of_code": semantic_complexity.get("lines_of_code", 0),
                    "parameters_count": semantic_complexity.get("parameters", 0),
                    "token_count": semantic_complexity.get("token_count", 0),
                    "complexity_score": semantic_complexity.get("complexity_score", 0),
                    "return_types": syntactic_complexity.get("return_types", 0),
                    "external_dependencies": syntactic_complexity.get("external_dependencies", 0),
                    "syntax_depth": syntactic_complexity.get("syntax_depth", 0),
                    "branch_count": structural_complexity.get("branch_count", 0),
                    "total_score": total_score,
                    "importance_level": importance_level,
                    "is_hub": metrics.get("is_hub", False),
                    "is_leaf": metrics.get("is_leaf", False),
                    "is_coordinator": metrics.get("is_coordinator", False),
                    "is_foundation": metrics.get("is_foundation", False)
                }
                
                self.session.run(query, params)
                
                if i % 100 == 0:  # �?00个函数打印一次进�?
                    logger.info(f"已处�?{i+1}/{len(functions)} 个函�?)
                    
            except Exception as e:
                logger.error(f"创建函数节点失败 (函数 {i+1}): {e}")
                logger.error(f"函数数据: {func}")
                continue
        
        logger.info(f"创建�?{len(functions)} 个函数节�?)
    
    def _create_call_relationships(self, result: Dict[str, Any]) -> None:
        """创建函数调用关系"""
        functions = result.get("functions", [])
        repo_name = result.get("repository", "unknown")
        
        for func in functions:
            func_name = func.get("basic_info", {}).get("function_name", "")
            internal_calls = func.get("context", {}).get("function_calls", {}).get("internal_calls", [])
            
            # 创建内部调用关系
            for called_func in internal_calls:
                query = """
                MATCH (caller:Function {name: $caller_name})-[:BELONGS_TO]->(r:Repository {name: $repo_name})
                MATCH (callee:Function {name: $callee_name})-[:BELONGS_TO]->(r)
                CREATE (caller)-[:CALLS {type: 'internal', created_at: datetime()}]->(callee)
                """
                
                self.session.run(query, {
                    "repo_name": repo_name,
                    "caller_name": func_name,
                    "callee_name": called_func
                })
        
        logger.info("创建函数调用关系完成")
    
    def _create_transitive_relationships(self, result: Dict[str, Any]) -> None:
        """创建传递调用关�?""
        transitive_calls = result.get("transitive_calls", {})
        transitive_graph = transitive_calls.get("transitive_graph", {})
        edges = transitive_graph.get("edges", [])
        repo_name = result.get("repository", "unknown")
        
        for edge in edges:
            source = edge.get("source", "")
            target = edge.get("target", "")
            depth = edge.get("depth", 1)
            
            query = """
            MATCH (source:Function {name: $source_name})-[:BELONGS_TO]->(r:Repository {name: $repo_name})
            MATCH (target:Function {name: $target_name})-[:BELONGS_TO]->(r)
            CREATE (source)-[:TRANSITIVE_CALLS {
                depth: $depth,
                created_at: datetime()
            }]->(target)
            """
            
            self.session.run(query, {
                "repo_name": repo_name,
                "source_name": source,
                "target_name": target,
                "depth": depth
            })
        
        logger.info(f"创建�?{len(edges)} 个传递调用关�?)
    
    def _create_dependency_relationships(self, result: Dict[str, Any]) -> None:
        """创建依赖关系"""
        api_relationships = result.get("api_call_relationships", {})
        dependencies = api_relationships.get("dependencies", {})
        repo_name = result.get("repository", "unknown")
        
        # 文件级依�?
        file_deps = dependencies.get("file_level", {}).get("dependencies", {})
        for source_file, target_files in file_deps.items():
            for target_file in target_files:
                query = """
                MATCH (r:Repository {name: $repo_name})
                MATCH (source:File {path: $source_path})-[:BELONGS_TO]->(r)
                MATCH (target:File {path: $target_path})-[:BELONGS_TO]->(r)
                CREATE (source)-[:DEPENDS_ON {level: 'file', created_at: datetime()}]->(target)
                """
                
                self.session.run(query, {
                    "repo_name": repo_name,
                    "source_path": source_file,
                    "target_path": target_file
                })
        
        # 类级依赖
        class_deps = dependencies.get("class_level", {}).get("dependencies", {})
        for source_class, target_classes in class_deps.items():
            for target_class in target_classes:
                query = """
                MATCH (r:Repository {name: $repo_name})
                MATCH (source:Class {name: $source_name})-[:BELONGS_TO]->(r)
                MATCH (target:Class {name: $target_name})-[:BELONGS_TO]->(r)
                CREATE (source)-[:DEPENDS_ON {level: 'class', created_at: datetime()}]->(target)
                """
                
                self.session.run(query, {
                    "repo_name": repo_name,
                    "source_name": source_class,
                    "target_name": target_class
                })
        
        logger.info("创建依赖关系完成")
    
    def create_indexes(self) -> None:
        """创建索引以提高查询性能"""
        indexes = [
            "CREATE INDEX function_name_index IF NOT EXISTS FOR (f:Function) ON (f.name)",
            "CREATE INDEX function_importance_index IF NOT EXISTS FOR (f:Function) ON (f.importance_level)",
            "CREATE INDEX function_complexity_index IF NOT EXISTS FOR (f:Function) ON (f.complexity_score)",
            "CREATE INDEX file_path_index IF NOT EXISTS FOR (f:File) ON (f.path)",
            "CREATE INDEX class_name_index IF NOT EXISTS FOR (c:Class) ON (c.name)",
            "CREATE INDEX repository_name_index IF NOT EXISTS FOR (r:Repository) ON (r.name)"
        ]
        
        for index_query in indexes:
            try:
                self.session.run(index_query)
            except Exception as e:
                logger.warning(f"创建索引失败: {e}")
        
        logger.info("索引创建完成")


def write_to_neo4j(result: Dict[str, Any], 
                   uri: str = "bolt://localhost:7687",
                   username: str = "neo4j", 
                   password: str = "90879449Drq",
                   database: str = "neo4j",
                   clear_db: bool = False) -> None:
    """
    将分析结果写入Neo4j数据库的便捷函数
    
    Args:
        result: 分析结果字典
        uri: Neo4j数据库URI
        username: 用户�?
        password: 密码
        database: 数据库名�?
        clear_db: 是否清空数据�?
    """
    try:
        with Neo4jWriter(uri, username, password, database) as writer:
            if clear_db:
                writer.clear_database()
            
            # 创建索引
            writer.create_indexes()
            
            # 写入数据
            writer.write_analysis_result(result)
            
        logger.info("数据成功写入Neo4j数据�?)
        
    except Exception as e:
        logger.error(f"写入Neo4j数据库失�? {e}")
        raise
