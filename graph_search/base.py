import logging
from typing import Dict, List, Any, Optional
from neo4j import GraphDatabase
from .config import config

logger = logging.getLogger(__name__)

class BaseSearch:

    def __init__(self, 
                 neo4j_uri: Optional[str] = None,
                 neo4j_user: Optional[str] = None,
                 neo4j_password: Optional[str] = None,
                 neo4j_database: Optional[str] = None):
        """
        初始化基础搜索类
        
        Args:
            neo4j_uri: Neo4j数据库URI
            neo4j_user: Neo4j用户名
            neo4j_password: Neo4j密码
            neo4j_database: 数据库名称
        """
        self.neo4j_uri = neo4j_uri or config.neo4j_uri
        self.neo4j_user = neo4j_user or config.neo4j_user
        self.neo4j_password = neo4j_password or config.neo4j_password
        self.neo4j_database = neo4j_database or config.neo4j_database
        
        self.driver = GraphDatabase.driver(
            self.neo4j_uri, 
            auth=(self.neo4j_user, self.neo4j_password)
        )
        
        logger.info(f"Neo4j连接已建立: {self.neo4j_uri}")
    
    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭")
    
    def _run_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        try:
            with self.driver.session(database=self.neo4j_database) as session:
                result = session.run(query, parameters or {})
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"查询执行失败: {e}")
            logger.error(f"查询语句: {query}")
            logger.error(f"查询参数: {parameters}")
            raise
    
    def _run_query_single(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        try:
            with self.driver.session(database=self.neo4j_database) as session:
                result = session.run(query, parameters or {})
                record = result.single()
                return dict(record) if record else None
        except Exception as e:
            logger.error(f"查询执行失败: {e}")
            logger.error(f"查询语句: {query}")
            logger.error(f"查询参数: {parameters}")
            raise
    
    def get_database_info(self) -> Dict[str, Any]:
        """获取数据库基本信息"""
        queries = {
            "total_functions": "MATCH (fn:Function) RETURN count(fn) as count",
            "total_files": "MATCH (f:File) RETURN count(f) as count",
            "total_repositories": "MATCH (r:Repository) RETURN count(r) as count",
            "total_relationships": "MATCH ()-[r]->() RETURN count(r) as count"
        }
        
        info = {}
        for key, query in queries.items():
            try:
                result = self._run_query_single(query)
                info[key] = result["count"] if result else 0
            except Exception as e:
                logger.warning(f"获取{key}信息失败: {e}")
                info[key] = 0
        
        return info
    
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
