import os
import json
import sqlite3
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import defaultdict

from neo4j import GraphDatabase


def _get_sqlite_path() -> str:
    home = os.path.expanduser("~")
    return os.path.join(home, ".continue", "index", "sync.db")


def _sanitize_value_for_neo4j(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        # 仅当所有元素都是原始类型或 None 时保留为数组；否则序列化为 JSON 字符串
        if all(isinstance(x, (str, int, float, bool)) or x is None for x in value):
            return list(value)
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    # 其他类型（如 set、自定义对象）转换为字符串
    return str(value)


def _sanitize_props_for_neo4j(props: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _sanitize_value_for_neo4j(v) for k, v in props.items()}


class SimplifiedNeo4jIngestor:
    
    def __init__(self, uri: str, user: str, password: str, database: Optional[str] = None):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database
        self._function_cache = {}  # 缓存函数信息用于关系创建
        self._class_cache = {}     # 缓存类信息

    def close(self):
        self._driver.close()

    def _run(self, query: str, parameters: Optional[Dict[str, Any]] = None):
        with self._driver.session(database=self._database) as session:
            return session.run(query, parameters or {})

    def clean_database(self):
        """清除数据库中的所有节点和关系"""
        print("正在清除数据库...")
        # 删除所有关系
        self._run("MATCH ()-[r]->() DELETE r")
        # 删除所有节点
        self._run("MATCH (n) DELETE n")
        print("数据库清除完成")

    def ensure_constraints(self):
        """创建所有必要的约束和索引"""
        queries = [
            # 基础约束
            "CREATE CONSTRAINT repo_name IF NOT EXISTS FOR (r:Repository) REQUIRE r.name IS UNIQUE",
            "CREATE CONSTRAINT file_id IF NOT EXISTS FOR (f:File) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT function_id IF NOT EXISTS FOR (fn:Function) REQUIRE fn.id IS UNIQUE",
            "CREATE CONSTRAINT class_id IF NOT EXISTS FOR (c:Class) REQUIRE c.id IS UNIQUE",
            
            # 索引
            "CREATE INDEX function_name_idx IF NOT EXISTS FOR (fn:Function) ON (fn.name)",
            "CREATE INDEX class_name_idx IF NOT EXISTS FOR (c:Class) ON (c.name)",
            "CREATE INDEX file_path_idx IF NOT EXISTS FOR (f:File) ON (f.path)",
            "CREATE INDEX complexity_score_idx IF NOT EXISTS FOR (fn:Function) ON (fn.complexity_score)",
        ]
        for q in queries:
            try:
                self._run(q)
            except Exception as e:
                print(f"Warning: Could not create constraint/index: {e}")

    def upsert_repository(self, name: str, extra: Dict[str, Any]):
        """创建或更新仓库节点"""
        sanitized = _sanitize_props_for_neo4j(extra or {})
        self._run(
            """
            MERGE (r:Repository {name: $name})
            SET r += $extra,
                r.analysis_timestamp = datetime(),
                r.last_updated = datetime()
            """,
            {"name": name, "extra": sanitized},
        )

    def upsert_file(self, repo: str, path: str, file_metadata: Dict[str, Any] = None):
        """创建或更新文件节点"""
        file_id = f"{repo}:{path}"
        
        # 分析文件类型和特征
        file_type = self._analyze_file_type(path)
        file_stats = self._calculate_file_stats(file_metadata or {})
        
        props = {
            "id": file_id,
            "path": path,
            "filename": os.path.basename(path),
            "directory": os.path.dirname(path),
            "file_type": file_type,
            "file_extension": os.path.splitext(path)[1],
            **file_stats
        }
        
        props = _sanitize_props_for_neo4j(props)
        
        self._run(
            """
            MATCH (r:Repository {name: $repo})
            MERGE (f:File {id: $id})
            SET f += $props
            MERGE (r)-[:CONTAINS]->(f)
            """,
            {"repo": repo, "id": file_id, "props": props},
        )
        return file_id

    def _analyze_file_type(self, path: str) -> str:
        """分析文件类型"""
        if path.endswith('_test.py') or 'test_' in path:
            return 'test'
        elif path.endswith('.py'):
            if '__init__.py' in path:
                return 'package_init'
            elif 'setup.py' in path:
                return 'setup'
            elif 'requirements' in path:
                return 'requirements'
            else:
                return 'module'
        elif path.endswith('.md'):
            return 'documentation'
        elif path.endswith('.json'):
            return 'configuration'
        else:
            return 'other'

    def _calculate_file_stats(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """计算文件统计信息"""
        return {
            "total_functions": metadata.get("total_functions", 0),
            "functions_with_docstring": metadata.get("functions_with_docstring", 0),
            "complexity_distribution": metadata.get("complexity_distribution", {}),
        }

    def upsert_function(self, repo: str, file_id: str, func: Dict[str, Any]):
        """创建或更新函数节点，包含更丰富的属性"""
        name = func.get("basic_info", {}).get("function_name") or ""
        start_line = func.get("basic_info", {}).get("code_location", {}).get("start_line")
        end_line = func.get("basic_info", {}).get("code_location", {}).get("end_line")
        func_id = f"{repo}:{file_id}:{name}:{start_line}-{end_line}"

        # 基础属性
        props = {
            "id": func_id,
            "name": name,
            "filepath": func.get("context", {}).get("file_path"),
            "start_line": start_line,
            "end_line": end_line,
            "source_code": func.get("basic_info", {}).get("source_code"),
            "comments": func.get("basic_info", {}).get("comments", []),
            "return_type": func.get("basic_info", {}).get("return_type", ""),
        }

        # 复杂度信息
        complexity = func.get("complexity", {})
        if complexity:
            semantic_complexity = complexity.get("semantic_complexity", {})
            syntax_complexity = complexity.get("syntax_complexity", {})
            structural_complexity = complexity.get("structural_complexity", {})
            
            props.update({
                "cyclomatic_complexity": semantic_complexity.get("cyclomatic_complexity"),
                "lines_of_code": semantic_complexity.get("lines_of_code"),
                "parameters_count": semantic_complexity.get("parameters"),
                "token_count": semantic_complexity.get("token_count"),
                "complexity_score": semantic_complexity.get("complexity_score"),
                "return_types_count": syntax_complexity.get("return_types_count"),
                "external_dependencies_count": syntax_complexity.get("external_dependencies_count"),
                "syntax_depth": syntax_complexity.get("syntax_depth"),
                "branch_count": structural_complexity.get("branch_count"),
            })

        # 重要性信息
        importance = func.get("importance", {})
        if importance:
            props.update({
                "importance_score": importance.get("total_score"),
                "importance_level": importance.get("importance_level"),
                "importance_rank": importance.get("rank"),
                "is_hub": importance.get("metrics", {}).get("is_hub", False),
                "is_leaf": importance.get("metrics", {}).get("is_leaf", False),
                "is_coordinator": importance.get("metrics", {}).get("is_coordinator", False),
                "is_foundation": importance.get("metrics", {}).get("is_foundation", False),
                "max_call_depth": importance.get("metrics", {}).get("max_call_depth"),
                "total_influence_scope": importance.get("metrics", {}).get("total_influence_scope"),
            })

        # 上下文信息
        context = func.get("context", {})
        if context:
            parent_class = context.get("parent_class")
            if parent_class:
                # parent_class 可能是字符串或对象
                if isinstance(parent_class, str):
                    props["parent_class_name"] = parent_class
                    props["parent_class_type"] = "class"
                else:
                    props["parent_class_name"] = parent_class.get("name")
                    props["parent_class_type"] = parent_class.get("type")
            
            props.update({
                "imports": context.get("imports", []),
                "function_calls_count": len(context.get("function_calls", {}).get("internal_calls", [])) + 
                                     len(context.get("function_calls", {}).get("external_calls", [])),
            })

        # 描述信息
        description_info = func.get("description_info", {})
        if description_info:
            docstring = description_info.get("docstring", {})
            props.update({
                "docstring_description": docstring.get("description"),
                "docstring_args": docstring.get("args"),
                "docstring_returns": docstring.get("returns"),
                "duration_seconds": description_info.get("duration"),
                "complexity_level": description_info.get("complexity_level"),
                "context_summary": description_info.get("context_summary"),
                "has_error": description_info.get("has_error", False),
                "error_message": description_info.get("error"),
            })

        # 函数类型分析
        props.update(self._analyze_function_type(func))

        props = _sanitize_props_for_neo4j(props)

        # 缓存函数信息用于后续关系创建
        self._function_cache[func_id] = {
            "name": name,
            "file_id": file_id,
            "function_data": func
        }

        self._run(
            """
            MERGE (fn:Function {id: $id})
            SET fn += $props
            WITH fn
            MATCH (f:File {id: $file_id})
            MERGE (f)-[:DECLARES]->(fn)
            """,
            {"id": func_id, "props": props, "file_id": file_id},
        )
        return func_id

    def _analyze_function_type(self, func: Dict[str, Any]) -> Dict[str, Any]:
        """分析函数类型和特征"""
        name = func.get("basic_info", {}).get("function_name", "")
        source_code = func.get("basic_info", {}).get("source_code", "")
        
        function_type = "regular"
        is_async = False
        is_generator = False
        is_decorator = False
        is_test = False
        
        if name.startswith("test_") or name.startswith("test"):
            function_type = "test"
            is_test = True
        elif name.startswith("__") and name.endswith("__"):
            function_type = "magic_method"
        elif name.startswith("_"):
            function_type = "private"
        elif "async def" in source_code:
            function_type = "async"
            is_async = True
        elif "yield" in source_code:
            function_type = "generator"
            is_generator = True
        elif "@" in source_code and "def" in source_code:
            function_type = "decorated"
        elif "def " in source_code and "(" in source_code and ")" in source_code:
            # 检查是否是装饰器函数
            lines = source_code.split('\n')
            for line in lines:
                if "def " in line and "return" in source_code:
                    is_decorator = True
                    function_type = "decorator"
                    break
        
        return {
            "function_type": function_type,
            "is_async": is_async,
            "is_generator": is_generator,
            "is_decorator": is_decorator,
            "is_test": is_test,
        }

    def create_class_relationships(self, repo: str, functions: List[Dict[str, Any]]):
        """创建类关系"""
        class_functions = defaultdict(list)
        
        # 按类分组函数
        for func in functions:
            parent_class = func.get("context", {}).get("parent_class")
            if parent_class:
                # parent_class 可能是字符串或对象
                if isinstance(parent_class, str):
                    class_name = parent_class
                else:
                    class_name = parent_class.get("name")
                if class_name:
                    class_functions[class_name].append(func)
        
        # 为每个类创建类节点
        for class_name, class_funcs in class_functions.items():
            class_id = f"{repo}:class:{class_name}"
            
            # 计算类的统计信息
            total_methods = len(class_funcs)
            public_methods = len([f for f in class_funcs if not f.get("basic_info", {}).get("function_name", "").startswith("_")])
            private_methods = total_methods - public_methods
            magic_methods = len([f for f in class_funcs if f.get("basic_info", {}).get("function_name", "").startswith("__")])
            
            class_props = {
                "id": class_id,
                "name": class_name,
                "total_methods": total_methods,
                "public_methods": public_methods,
                "private_methods": private_methods,
                "magic_methods": magic_methods,
                "is_abstract": any("abstract" in f.get("basic_info", {}).get("source_code", "") for f in class_funcs),
            }
            
            class_props = _sanitize_props_for_neo4j(class_props)
            
            self._run(
                """
                MERGE (c:Class {id: $id})
                SET c += $props
                """,
                {"id": class_id, "props": class_props},
            )
            
            # 创建类与函数的关系
            for func in class_funcs:
                func_name = func.get("basic_info", {}).get("function_name")
                file_path = func.get("context", {}).get("file_path", "")
                start_line = func.get("basic_info", {}).get("code_location", {}).get("start_line")
                end_line = func.get("basic_info", {}).get("code_location", {}).get("end_line")
                file_id = f"{repo}:{file_path}"
                func_id = f"{repo}:{file_id}:{func_name}:{start_line}-{end_line}"
                
                # 确定方法类型
                method_type = "instance_method"
                if func_name.startswith("__") and func_name.endswith("__"):
                    method_type = "magic_method"
                elif func_name.startswith("_"):
                    method_type = "private_method"
                elif "staticmethod" in func.get("basic_info", {}).get("source_code", ""):
                    method_type = "static_method"
                elif "classmethod" in func.get("basic_info", {}).get("source_code", ""):
                    method_type = "class_method"
                
                self._run(
                    """
                    MATCH (c:Class {id: $class_id})
                    MATCH (fn:Function {id: $func_id})
                    MERGE (c)-[:HAS_METHOD {method_type: $method_type}]->(fn)
                    """,
                    {"class_id": class_id, "func_id": func_id, "method_type": method_type},
                )

    def create_function_call_relationships(self, repo: str, functions: List[Dict[str, Any]]):
        """创建函数调用关系"""
        # 创建函数名到函数ID的映射
        func_name_to_id = {}
        for func in functions:
            name = func.get("basic_info", {}).get("function_name")
            if name:
                file_path = func.get("context", {}).get("file_path", "")
                start_line = func.get("basic_info", {}).get("code_location", {}).get("start_line")
                end_line = func.get("basic_info", {}).get("code_location", {}).get("end_line")
                file_id = f"{repo}:{file_path}"
                func_id = f"{repo}:{file_id}:{name}:{start_line}-{end_line}"
                func_name_to_id[name] = func_id

        # 创建内部调用关系
        for func in functions:
            caller_name = func.get("basic_info", {}).get("function_name")
            if not caller_name:
                continue
                
            file_path = func.get("context", {}).get("file_path", "")
            start_line = func.get("basic_info", {}).get("code_location", {}).get("start_line")
            end_line = func.get("basic_info", {}).get("code_location", {}).get("end_line")
            file_id = f"{repo}:{file_path}"
            caller_id = f"{repo}:{file_id}:{caller_name}:{start_line}-{end_line}"

            # 处理内部调用
            internal_calls = func.get("context", {}).get("function_calls", {}).get("internal_calls", [])
            for call in internal_calls:
                callee_name = call.get("function_name")
                if callee_name and callee_name in func_name_to_id:
                    callee_id = func_name_to_id[callee_name]
                    
                    self._run(
                        """
                        MATCH (caller:Function {id: $caller_id})
                        MATCH (callee:Function {id: $callee_id})
                        MERGE (caller)-[:CALLS]->(callee)
                        """,
                        {
                            "caller_id": caller_id,
                            "callee_id": callee_id,
                        },
                    )

    def ingest_complete_json_to_neo4j(self, complete_json_path: str, neo4j_uri: str, neo4j_user: str, 
                                    neo4j_password: str, database: Optional[str] = None, clean_db: bool = False):
        """主函数：将完整的JSON数据导入到Neo4j"""
        print(f"开始导入数据: {complete_json_path}")
        
        with open(complete_json_path, 'r', encoding='utf-8') as f:
            complete_data = json.load(f)
        
        # 从文件名推断仓库名
        repo_name = os.path.basename(complete_json_path).replace("_complete_for_neo4j.json", "")
        functions = complete_data.get("functions", [])
        # 元数据直接在根级别
        metadata = {
            "total_functions": complete_data.get("total_functions", 0),
            "functions_with_docstring": complete_data.get("functions_with_docstring", 0),
            "complexity_distribution": complete_data.get("complexity_distribution", {}),
            "generation_timestamp": complete_data.get("analysis_timestamp", ""),
        }
        
        ingestor = SimplifiedNeo4jIngestor(neo4j_uri, neo4j_user, neo4j_password, database)
        try:
            # 如果需要清除数据库，先清除所有数据
            if clean_db:
                ingestor.clean_database()
            
            ingestor.ensure_constraints()
            
            # 创建仓库节点，包含元数据信息
            repo_extra = {
                "total_functions": metadata.get("total_functions", 0),
                "functions_with_docstring": metadata.get("functions_with_docstring", 0),
                "complexity_distribution": metadata.get("complexity_distribution", {}),
                "generation_timestamp": metadata.get("generation_timestamp", ""),
            }
            ingestor.upsert_repository(repo_name, repo_extra)
            
            # 按文件分组函数
            file_functions = defaultdict(list)
            for func in functions:
                file_path = func.get("context", {}).get("file_path", "")
                if file_path:
                    file_functions[file_path].append(func)
            
            # 为每个文件创建文件节点
            for file_path, file_funcs in file_functions.items():
                file_metadata = {
                    "total_functions": len(file_funcs),
                    "functions_with_docstring": len([f for f in file_funcs if f.get("description_info", {}).get("docstring", {}).get("description")]),
                }
                file_id = ingestor.upsert_file(repo_name, file_path, file_metadata)
                
                # 为每个函数创建函数节点
                for func in file_funcs:
                    func_id = ingestor.upsert_function(repo_name, file_id, func)
            
            # 创建各种关系
            print("创建函数调用关系...")
            ingestor.create_function_call_relationships(repo_name, functions)
            
            print("创建类关系...")
            ingestor.create_class_relationships(repo_name, functions)
            
            print(f"数据导入完成！共处理 {len(functions)} 个函数")
            
        finally:
            ingestor.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Simplified Neo4j ingestion for complete JSON data")
    parser.add_argument("complete_json", help="Path to complete JSON file (e.g., attrs_complete_for_neo4j.json)")
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", "90879449Drq"))
    parser.add_argument("--database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    parser.add_argument("--clean-db", action="store_true", help="Clear the database before ingesting data")

    args = parser.parse_args()
    
    ingestor = SimplifiedNeo4jIngestor(args.uri, args.user, args.password, args.database)
    ingestor.ingest_complete_json_to_neo4j(args.complete_json, args.uri, args.user, args.password, args.database, args.clean_db)


if __name__ == "__main__":
    main()
