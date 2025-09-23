#!/usr/bin/env python3

import os
import json
import sqlite3
import hashlib
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# 尝试导入 sync �?try:
    from sync import SyncIndex, SyncIndexConfig
    SYNC_AVAILABLE = True
except ImportError:
    SYNC_AVAILABLE = False
    # 创建一个模拟的 sync 类用于开�?    class SyncIndex:
        def __init__(self, config):
            self.config = config
            self.index = {}
            self.repo_path = getattr(config, 'repo_path', '')
        
        def index_file(self, file_path: str, content: str):
            self.index[file_path] = content
        
        def get_file_content(self, file_path: str) -> Optional[str]:
            # 如果内存中没有，尝试从文件系统读�?            if file_path in self.index:
                return self.index[file_path]
            
            # 尝试从文件系统读�?            try:
                full_path = os.path.join(self.repo_path, file_path)
                if os.path.exists(full_path):
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        self.index[file_path] = content
                        return content
            except Exception as e:
                logger.warning(f"Failed to read file {file_path}: {e}")
            
            return None
        
        def get_changed_files(self, since: str = None) -> List[str]:
            # 扫描文件系统获取所有文�?            if not self.repo_path or not os.path.exists(self.repo_path):
                return []
            
            files = []
            for root, dirs, filenames in os.walk(self.repo_path):
                # 排除不需要的目录
                dirs[:] = [d for d in dirs if not any(pattern in d for pattern in self.config.exclude_patterns)]
                
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, self.repo_path)
                    
                    # 检查文件扩展名
                    if any(rel_path.endswith(ext) for ext in self.config.include_extensions):
                        # 检查排除模�?                        if not any(pattern in rel_path for pattern in self.config.exclude_patterns):
                            files.append(rel_path.replace('\\', '/'))  # 统一使用正斜�?            
            return files
    
    class SyncIndexConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            # 确保有默认的配置
            if not hasattr(self, 'include_extensions'):
                self.include_extensions = [".py", ".js", ".ts", ".java", ".cpp", ".c", ".h"]
            if not hasattr(self, 'exclude_patterns'):
                self.exclude_patterns = ["__pycache__", ".git", "node_modules", "*.pyc"]

from .vector_embedding import CodeEmbeddingManager, EmbeddingConfig, CodeEmbedding
from .graph_indexer import GraphIndexerConfig, create_default_graph_config

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SyncIndexerConfig:
    """Sync 索引器配�?""
    # Sync 配置
    sync_config: Dict[str, Any] = None
    repo_path: str = ""
    index_path: str = "./sync_index"
    
    # 嵌入配置
    embedding_config: EmbeddingConfig = None
    
    # 图数据库配置
    graph_config: GraphIndexerConfig = None
    enable_graph_indexing: bool = True
    
    # 过滤配置
    include_extensions: List[str] = None
    exclude_patterns: List[str] = None
    max_file_size: int = 1024 * 1024  # 1MB
    
    # 索引配置
    batch_size: int = 100
    update_interval: int = 3600  # 1小时
    enable_incremental: bool = True
    
    def __post_init__(self):
        if self.sync_config is None:
            self.sync_config = {
                "max_file_size": self.max_file_size,
                "include_extensions": self.include_extensions or [".py", ".js", ".ts", ".java", ".cpp", ".c", ".h"],
                "exclude_patterns": self.exclude_patterns or ["__pycache__", ".git", "node_modules", "*.pyc"]
            }
        
        if self.embedding_config is None:
            from .vector_embedding import create_default_config
            self.embedding_config = create_default_config()
        
        if self.graph_config is None:
            self.graph_config = create_default_graph_config()


@dataclass
class IndexedFile:
    """索引文件信息"""
    file_path: str
    content_hash: str
    last_modified: str
    file_size: int
    language: str
    functions: List[Dict[str, Any]]
    classes: List[Dict[str, Any]]
    imports: List[str]
    metadata: Dict[str, Any]


class SyncSemanticIndexer:
    """基于 Sync 的语义索引器"""
    
    def __init__(self, config: SyncIndexerConfig):
        self.config = config
        self.sync_index = None
        self.embedding_manager = None
        self.graph_indexer = None
        self.index_db = None
        self.indexed_files = {}  # file_path -> IndexedFile
        
        # 初始化组�?        self._init_sync_index()
        self._init_embedding_manager()
        self._init_graph_indexer()
        self._init_index_db()
    
    def _init_sync_index(self):
        """初始�?Sync 索引"""
        if not SYNC_AVAILABLE:
            logger.warning("Sync library not available, using mock implementation")
        
        try:
            sync_config = SyncIndexConfig(**self.config.sync_config)
            # 添加repo_path到sync配置�?            sync_config.repo_path = self.config.repo_path
            self.sync_index = SyncIndex(sync_config)
            logger.info("Sync index initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize sync index: {e}")
            raise
    
    def _init_embedding_manager(self):
        """初始化嵌入管理器"""
        self.embedding_manager = CodeEmbeddingManager(self.config.embedding_config)
        logger.info("Embedding manager initialized")
    
    def _init_graph_indexer(self):
        """初始化图数据库索引器"""
        if self.config.enable_graph_indexing:
            # 图索引器功能暂时禁用
            self.graph_indexer = None
            logger.info("Graph indexing disabled (not implemented)")
        else:
            logger.info("Graph indexing disabled")
    
    def _init_index_db(self):
        """初始化索引数据库"""
        os.makedirs(self.config.index_path, exist_ok=True)
        db_path = os.path.join(self.config.index_path, "semantic_index.db")
        
        self.index_db = sqlite3.connect(db_path)
        
        # 创建�?        self.index_db.execute("""
            CREATE TABLE IF NOT EXISTS indexed_files (
                file_path TEXT PRIMARY KEY,
                content_hash TEXT,
                last_modified TEXT,
                file_size INTEGER,
                language TEXT,
                functions TEXT,
                classes TEXT,
                imports TEXT,
                metadata TEXT,
                indexed_at TEXT
            )
        """)
        
        self.index_db.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                file_path TEXT,
                function_id TEXT,
                embedding_id TEXT,
                content TEXT,
                metadata TEXT,
                PRIMARY KEY (file_path, function_id)
            )
        """)
        
        self.index_db.execute("""
            CREATE TABLE IF NOT EXISTS index_stats (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        """)
        
        self.index_db.commit()
        logger.info("Index database initialized")
    
    def _get_file_hash(self, file_path: str, content: str) -> str:
        """计算文件哈希"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _detect_language(self, file_path: str) -> str:
        """检测文件语言"""
        ext = Path(file_path).suffix.lower()
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala'
        }
        return language_map.get(ext, 'unknown')
    
    def _extract_code_structure(self, content: str, language: str) -> Dict[str, Any]:
        """提取代码结构"""
        # 这里应该使用 AST 解析器，但为了简化，使用正则表达�?        import re
        
        functions = []
        classes = []
        imports = []
        
        if language == 'python':
            # 提取函数
            func_pattern = r'def\s+(\w+)\s*\([^)]*\):'
            for match in re.finditer(func_pattern, content):
                functions.append({
                    'name': match.group(1),
                    'line': content[:match.start()].count('\n') + 1
                })
            
            # 提取�?            class_pattern = r'class\s+(\w+)(?:\([^)]*\))?:'
            for match in re.finditer(class_pattern, content):
                classes.append({
                    'name': match.group(1),
                    'line': content[:match.start()].count('\n') + 1
                })
            
            # 提取导入
            import_pattern = r'(?:from\s+\S+\s+)?import\s+([^\n]+)'
            for match in re.finditer(import_pattern, content):
                imports.append(match.group(1).strip())
        
        elif language in ['javascript', 'typescript']:
            # 提取函数
            func_pattern = r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>)'
            for match in re.finditer(func_pattern, content):
                func_name = match.group(1) or match.group(2)
                if func_name:
                    functions.append({
                        'name': func_name,
                        'line': content[:match.start()].count('\n') + 1
                    })
            
            # 提取�?            class_pattern = r'class\s+(\w+)'
            for match in re.finditer(class_pattern, content):
                classes.append({
                    'name': match.group(1),
                    'line': content[:match.start()].count('\n') + 1
                })
        
        return {
            'functions': functions,
            'classes': classes,
            'imports': imports
        }
    
    def _should_index_file(self, file_path: str, file_size: int) -> bool:
        """判断是否应该索引文件"""
        # 检查文件大�?        if file_size > self.config.max_file_size:
            return False
        
        # 检查扩展名
        ext = Path(file_path).suffix.lower()
        if self.config.include_extensions and ext not in self.config.include_extensions:
            return False
        
        # 检查排除模�?        if self.config.exclude_patterns:
            for pattern in self.config.exclude_patterns:
                if pattern in file_path or Path(file_path).match(pattern):
                    return False
        
        return True
    
    def _index_file_content(self, file_path: str, content: str) -> IndexedFile:
        """索引单个文件内容"""
        content_hash = self._get_file_hash(file_path, content)
        language = self._detect_language(file_path)
        structure = self._extract_code_structure(content, language)
        
        indexed_file = IndexedFile(
            file_path=file_path,
            content_hash=content_hash,
            last_modified=datetime.now().isoformat(),
            file_size=len(content),
            language=language,
            functions=structure['functions'],
            classes=structure['classes'],
            imports=structure['imports'],
            metadata={
                'line_count': content.count('\n') + 1,
                'character_count': len(content),
                'has_docstring': '"""' in content or "'''" in content
            }
        )
        
        return indexed_file
    
    def _generate_function_embeddings(self, file_path: str, content: str, 
                                    indexed_file: IndexedFile) -> List[CodeEmbedding]:
        """为文件中的函数生成嵌�?""
        if not indexed_file.functions:
            return []
        
        # 准备所有函数的代码和元数据
        func_codes = []
        func_metadata_list = []
        
        for func in indexed_file.functions:
            func_name = func['name']
            func_line = func['line']
            
            # 提取函数代码（简化版本）
            lines = content.split('\n')
            func_start = func_line - 1
            
            # 找到函数结束位置（简化）
            func_end = func_start
            indent_level = None
            
            for i in range(func_start, len(lines)):
                line = lines[i]
                if i == func_start:
                    # 第一行，获取缩进级别
                    indent_level = len(line) - len(line.lstrip())
                elif line.strip() and len(line) - len(line.lstrip()) <= indent_level:
                    # 找到同级别或更少缩进的行，函数结�?                    func_end = i
                    break
                else:
                    func_end = i + 1
            
            func_code = '\n'.join(lines[func_start:func_end])
            func_codes.append(func_code)
            
            # 创建函数元数�?            func_metadata = {
                'name': func_name,
                'filepath': file_path,
                'start_line': func_line,
                'end_line': func_end,
                'language': indexed_file.language,
                'function_type': 'regular',
                'file_type': 'module'
            }
            func_metadata_list.append(func_metadata)
        
        # 批量生成嵌入
        embeddings = self.embedding_manager.vectorizer.embed_texts(func_codes, func_metadata_list)
        
        return embeddings
    
    def index_repository(self, repo_path: str = None) -> Dict[str, Any]:
        """索引整个代码�?""
        if repo_path:
            self.config.repo_path = repo_path
        
        if not self.config.repo_path:
            raise ValueError("Repository path not specified")
        
        logger.info(f"Starting repository indexing: {self.config.repo_path}")
        
        # 使用 sync 索引文件
        changed_files = self.sync_index.get_changed_files()
        
        indexed_count = 0
        skipped_count = 0
        error_count = 0
        
        for file_path in changed_files:
            try:
                # 获取文件内容
                content = self.sync_index.get_file_content(file_path)
                if not content:
                    continue
                
                # 检查是否应该索�?                if not self._should_index_file(file_path, len(content)):
                    skipped_count += 1
                    continue
                
                # 索引文件
                indexed_file = self._index_file_content(file_path, content)
                
                # 生成函数嵌入
                function_embeddings = self._generate_function_embeddings(
                    file_path, content, indexed_file
                )
                
                # 保存到数据库
                self._save_indexed_file(indexed_file)
                self._save_function_embeddings(file_path, function_embeddings)
                
                # 添加到嵌入管理器
                self.embedding_manager.add_embeddings(function_embeddings)
                
                self.indexed_files[file_path] = indexed_file
                indexed_count += 1
                
                if indexed_count % self.config.batch_size == 0:
                    logger.info(f"Indexed {indexed_count} files...")
                
            except Exception as e:
                logger.error(f"Error indexing file {file_path}: {e}")
                error_count += 1
        
        # 同步到图数据�?        if self.graph_indexer and self.embedding_manager.embeddings_index:
            logger.info("Syncing embeddings to graph database...")
            try:
                all_embeddings = list(self.embedding_manager.embeddings_index.values())
                self.graph_indexer.index_embeddings(all_embeddings, self.indexed_files)
                logger.info("Graph database sync completed")
            except Exception as e:
                logger.error(f"Failed to sync to graph database: {e}")
        
        # 更新统计信息
        stats = {
            'indexed_files': indexed_count,
            'skipped_files': skipped_count,
            'error_files': error_count,
            'total_functions': len(self.embedding_manager.embeddings_index),
            'indexed_at': datetime.now().isoformat()
        }
        
        self._update_stats(stats)
        
        logger.info(f"Repository indexing completed: {stats}")
        return stats
    
    def _save_indexed_file(self, indexed_file: IndexedFile):
        """保存索引文件到数据库"""
        self.index_db.execute("""
            INSERT OR REPLACE INTO indexed_files 
            (file_path, content_hash, last_modified, file_size, language, 
             functions, classes, imports, metadata, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            indexed_file.file_path,
            indexed_file.content_hash,
            indexed_file.last_modified,
            indexed_file.file_size,
            indexed_file.language,
            json.dumps(indexed_file.functions),
            json.dumps(indexed_file.classes),
            json.dumps(indexed_file.imports),
            json.dumps(indexed_file.metadata),
            datetime.now().isoformat()
        ))
        self.index_db.commit()
    
    def _save_function_embeddings(self, file_path: str, embeddings: List[CodeEmbedding]):
        """保存函数嵌入到数据库"""
        for embedding in embeddings:
            func_id = f"{embedding.metadata.get('name', 'unknown')}_{embedding.metadata.get('start_line', 0)}"
            
            self.index_db.execute("""
                INSERT OR REPLACE INTO embeddings 
                (file_path, function_id, embedding_id, content, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                file_path,
                func_id,
                embedding.id,
                embedding.content,
                json.dumps(embedding.metadata)
            ))
        
        self.index_db.commit()
    
    def _update_stats(self, stats: Dict[str, Any]):
        """更新统计信息"""
        for key, value in stats.items():
            self.index_db.execute("""
                INSERT OR REPLACE INTO index_stats (key, value, updated_at)
                VALUES (?, ?, ?)
            """, (key, str(value), datetime.now().isoformat()))
        
        self.index_db.commit()
    
    def get_indexed_files(self) -> List[IndexedFile]:
        """获取所有已索引的文�?""
        cursor = self.index_db.execute("""
            SELECT file_path, content_hash, last_modified, file_size, language,
                   functions, classes, imports, metadata, indexed_at
            FROM indexed_files
            ORDER BY indexed_at DESC
        """)
        
        files = []
        for row in cursor.fetchall():
            file_path, content_hash, last_modified, file_size, language, \
            functions, classes, imports, metadata, indexed_at = row
            
            indexed_file = IndexedFile(
                file_path=file_path,
                content_hash=content_hash,
                last_modified=last_modified,
                file_size=file_size,
                language=language,
                functions=json.loads(functions),
                classes=json.loads(classes),
                imports=json.loads(imports),
                metadata=json.loads(metadata)
            )
            files.append(indexed_file)
        
        return files
    
    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        cursor = self.index_db.execute("SELECT key, value FROM index_stats")
        stats = dict(cursor.fetchall())
        
        # 添加嵌入管理器统�?        embedding_stats = self.embedding_manager.get_stats()
        stats.update(embedding_stats)
        
        return stats
    
    def search_semantic(self, query: str, top_k: int = 10, 
                       threshold: float = 0.0, 
                       filters: Dict[str, Any] = None) -> List[Tuple[CodeEmbedding, float]]:
        """语义搜索"""
        if not self.embedding_manager.embeddings_index:
            logger.warning("No embeddings available for search")
            return []
        
        # 执行搜索
        results = self.embedding_manager.search(query, top_k=top_k, threshold=threshold)
        
        # 应用过滤�?        if filters:
            filtered_results = []
            for embedding, similarity in results:
                if self._matches_filters(embedding, filters):
                    filtered_results.append((embedding, similarity))
            results = filtered_results
        
        return results
    
    def _matches_filters(self, embedding: CodeEmbedding, filters: Dict[str, Any]) -> bool:
        """检查嵌入是否匹配过滤器"""
        metadata = embedding.metadata
        
        for key, value in filters.items():
            if key == 'language' and metadata.get('language') != value:
                return False
            elif key == 'file_type' and metadata.get('file_type') != value:
                return False
            elif key == 'function_type' and metadata.get('function_type') != value:
                return False
            elif key == 'min_complexity' and metadata.get('complexity_score', 0) < value:
                return False
            elif key == 'max_complexity' and metadata.get('complexity_score', 0) > value:
                return False
            elif key == 'file_pattern' and value not in metadata.get('filepath', ''):
                return False
        
        return True
    
    def incremental_update(self) -> Dict[str, Any]:
        """增量更新索引"""
        if not self.config.enable_incremental:
            return {'message': 'Incremental update disabled'}
        
        logger.info("Starting incremental update...")
        
        # 获取变更的文�?        changed_files = self.sync_index.get_changed_files()
        
        updated_count = 0
        for file_path in changed_files:
            # 检查文件是否已索引
            cursor = self.index_db.execute(
                "SELECT content_hash FROM indexed_files WHERE file_path = ?",
                (file_path,)
            )
            existing_hash = cursor.fetchone()
            
            # 获取文件内容
            content = self.sync_index.get_file_content(file_path)
            if not content:
                continue
            
            current_hash = self._get_file_hash(file_path, content)
            
            # 如果文件已更改，重新索引
            if not existing_hash or existing_hash[0] != current_hash:
                try:
                    indexed_file = self._index_file_content(file_path, content)
                    function_embeddings = self._generate_function_embeddings(
                        file_path, content, indexed_file
                    )
                    
                    self._save_indexed_file(indexed_file)
                    self._save_function_embeddings(file_path, function_embeddings)
                    self.embedding_manager.add_embeddings(function_embeddings)
                    
                    updated_count += 1
                except Exception as e:
                    logger.error(f"Error updating file {file_path}: {e}")
        
        stats = {
            'updated_files': updated_count,
            'updated_at': datetime.now().isoformat()
        }
        
        self._update_stats(stats)
        logger.info(f"Incremental update completed: {stats}")
        return stats
    
    def save_index(self, filepath: str):
        """保存索引到文�?""
        # 保存嵌入索引
        embedding_filepath = filepath.replace('.json', '_embeddings.json')
        self.embedding_manager.save_index(embedding_filepath)
        
        # 保存向量化器状态（如果是TF-IDF�?        if self.config.embedding_config.model_type == "tfidf":
            vectorizer_filepath = filepath.replace('.json', '_vectorizer.pkl')
            self.embedding_manager.vectorizer.save_vectorizer(vectorizer_filepath)
        
        # 保存文件索引
        indexed_files = self.get_indexed_files()
        data = {
            'config': asdict(self.config),
            'indexed_files': [asdict(f) for f in indexed_files],
            'stats': self.get_stats(),
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str):
        """从文件加载索�?""
        # 加载嵌入索引
        embedding_filepath = filepath.replace('.json', '_embeddings.json')
        if os.path.exists(embedding_filepath):
            self.embedding_manager.load_index(embedding_filepath)
        
        # 加载向量化器状态（如果是TF-IDF�?        if self.config.embedding_config.model_type == "tfidf":
            vectorizer_filepath = filepath.replace('.json', '_vectorizer.pkl')
            self.embedding_manager.vectorizer.load_vectorizer(vectorizer_filepath)
        
        # 加载文件索引
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 重建索引文件
        for file_data in data.get('indexed_files', []):
            indexed_file = IndexedFile(**file_data)
            self.indexed_files[indexed_file.file_path] = indexed_file
        
        logger.info(f"Index loaded from {filepath}")
    
    def close(self):
        """关闭资源"""
        if self.embedding_manager:
            self.embedding_manager.close()
        
        if self.graph_indexer:
            self.graph_indexer.close()
        
        if self.index_db:
            self.index_db.close()
        
        logger.info("Sync semantic indexer closed")


def create_default_sync_config(repo_path: str) -> SyncIndexerConfig:
    """创建默认�?Sync 索引器配�?""
    from .vector_embedding import create_default_config
    
    return SyncIndexerConfig(
        repo_path=repo_path,
        index_path="./sync_index",
        embedding_config=create_default_config(),
        enable_graph_indexing=True,
        include_extensions=[".py", ".js", ".ts", ".java", ".cpp", ".c", ".h"],
        exclude_patterns=["__pycache__", ".git", "node_modules", "*.pyc", "*.class"],
        max_file_size=1024 * 1024,  # 1MB
        batch_size=100,
        update_interval=3600,
        enable_incremental=True
    )


if __name__ == "__main__":
    # 测试代码
    import tempfile
    import shutil
    
    # 创建测试目录
    test_dir = tempfile.mkdtemp()
    test_file = os.path.join(test_dir, "test.py")
    
    with open(test_file, 'w') as f:
        f.write("""
def calculate_fibonacci(n):
    '''Calculate the nth Fibonacci number'''
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def binary_search(arr, target):
    '''Binary search in sorted array'''
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
""")
    
    try:
        # 创建索引�?        config = create_default_sync_config(test_dir)
        indexer = SyncSemanticIndexer(config)
        
        # 索引代码�?        stats = indexer.index_repository()
        print("Indexing stats:", stats)
        
        # 搜索测试
        results = indexer.search_semantic("find element in array", top_k=2)
        print("\nSearch results:")
        for embedding, similarity in results:
            print(f"  Similarity: {similarity:.4f}")
            print(f"  Function: {embedding.metadata.get('name', 'Unknown')}")
            print(f"  File: {embedding.metadata.get('filepath', 'Unknown')}")
            print()
        
        # 显示统计信息
        stats = indexer.get_stats()
        print("Index statistics:")
        print(json.dumps(stats, indent=2))
        
        indexer.close()
        
    finally:
        # 清理测试目录
        shutil.rmtree(test_dir)
