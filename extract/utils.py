"""
工具函数模块
包含仓库克隆、语言检测、节点处理等基础工具函数
"""

import os
import subprocess
import json
import time
import logging
from pathlib import Path
from urllib.parse import urlparse
from tree_sitter import Parser, Node
# from tree_sitter_languages import get_language  # 不再使用
from typing import Dict, List, Optional, Tuple

from .config import EXT_TO_LANG, DEFAULT_TARGET_DIR

# 配置日志
logger = logging.getLogger(__name__)


class SetEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理set对象和bytes对象"""
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, bytes):
            return obj.decode('utf-8')
        return super().default(obj)


def clone_repo(repo_url: str, sparse_checkout_paths: Optional[List[str]] = None, 
               max_retries: int = 3, use_ssh: bool = False) -> Tuple[Path, str]:
    """
    克隆仓库到本地，支持稀疏检出和网络重试机制
    
    Args:
        repo_url: 仓库URL
        sparse_checkout_paths: 稀疏检出路径列表
        max_retries: 最大重试次数
        use_ssh: 是否使用SSH协议（需要配置SSH密钥）
        
    Returns:
        tuple: (本地路径, 仓库名称)
    """
    parsed = urlparse(repo_url)
    repo_name = Path(parsed.path).stem
    local_path = Path(DEFAULT_TARGET_DIR) / repo_name

    if not local_path.exists():
        logger.info(f"正在克隆仓库: {repo_url}")
        
        # 如果使用SSH，转换URL
        if use_ssh and "github.com" in repo_url:
            ssh_url = repo_url.replace("https://github.com/", "git@github.com:")
            repo_url = ssh_url
            logger.info(f"使用SSH协议: {repo_url}")
        
        # 尝试多种Git配置来解决SSL问题
        git_configs = [
            # 标准配置
            [],
            # 禁用SSL验证（不推荐，但可以解决连接问题）
            ["-c", "http.sslVerify=false"],
            # 使用较旧的SSL协议
            ["-c", "http.sslVersion=TLSv1.2"],
            # 增加超时时间
            ["-c", "http.lowSpeedLimit=0", "-c", "http.lowSpeedTime=999999"]
        ]
        
        success = False
        last_error = None
        
        for attempt in range(max_retries):
            for config_idx, git_config in enumerate(git_configs):
                try:
                    logger.info(f"尝试克隆 (第{attempt + 1}次，配置{config_idx + 1}): {repo_url}")
                    
                    if sparse_checkout_paths:
                        # 稀疏检出模式
                        logger.info(f"启用稀疏检出，只检出路径: {', '.join(sparse_checkout_paths)}")
                        
                        # 初始化仓库
                        cmd = ["git"] + git_config + ["clone", "--no-checkout", repo_url, str(local_path)]
                        subprocess.run(cmd, check=True, timeout=300)
                        
                        # 配置稀疏检出
                        subprocess.run(["git", "sparse-checkout", "init", "--cone"], 
                                     cwd=local_path, check=True, timeout=60)
                        for path in sparse_checkout_paths:
                            subprocess.run(["git", "sparse-checkout", "set", path], 
                                         cwd=local_path, check=True, timeout=60)
                        subprocess.run(["git", "checkout"], cwd=local_path, check=True, timeout=60)
                    else:
                        # 标准克隆
                        cmd = ["git"] + git_config + ["clone", repo_url, str(local_path)]
                        subprocess.run(cmd, check=True, timeout=300)
                    
                    success = True
                    logger.info(f"✅ 成功克隆仓库: {repo_name}")
                    break
                    
                except subprocess.CalledProcessError as e:
                    last_error = e
                    logger.warning(f"❌ 克隆失败 (配置{config_idx + 1}): {e}")
                    
                    # 清理失败的克隆目录
                    if local_path.exists():
                        import shutil
                        shutil.rmtree(local_path, ignore_errors=True)
                    
                    # 如果不是最后一次尝试，等待一段时间
                    if attempt < max_retries - 1 or config_idx < len(git_configs) - 1:
                        wait_time = 2 ** attempt  # 指数退避
                        logger.info(f"等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                
                except subprocess.TimeoutExpired:
                    last_error = Exception("Git操作超时")
                    logger.warning(f"⏰ Git操作超时 (配置{config_idx + 1})")
                    
                    # 清理超时的克隆目录
                    if local_path.exists():
                        import shutil
                        shutil.rmtree(local_path, ignore_errors=True)
            
            if success:
                break
        
        if not success:
            error_msg = f"❌ 克隆仓库失败，已尝试 {max_retries} 次: {repo_url}"
            if last_error:
                error_msg += f"\n最后错误: {last_error}"
            logger.error(error_msg)
            raise Exception(error_msg)
    else:
        logger.info(f"仓库已存在: {local_path}")
        
        if sparse_checkout_paths:
            logger.info("稀疏检出路径已设置")

    return local_path, repo_name


def detect_language(file_path: Path) -> Optional[str]:
    """
    根据文件扩展名检测语言
    
    Args:
        file_path: 文件路径
        
    Returns:
        语言名称或None
    """
    return EXT_TO_LANG.get(file_path.suffix)


def get_parser(lang: str) -> Optional[Parser]:
    """
    获取tree-sitter解析器
    
    Args:
        lang: 语言名称
        
    Returns:
        解析器实例或None
    """
    try:
        # 直接使用各个语言的tree-sitter包
        if lang == "java":
            import tree_sitter_java
            from tree_sitter import Language
            language_capsule = tree_sitter_java.language()
            language = Language(language_capsule)
            parser = Parser()
            parser.language = language
            return parser
        elif lang == "python":
            import tree_sitter_python
            from tree_sitter import Language
            language_capsule = tree_sitter_python.language()
            language = Language(language_capsule)
            parser = Parser()
            parser.language = language
            return parser
        elif lang == "javascript":
            import tree_sitter_javascript
            from tree_sitter import Language
            language_capsule = tree_sitter_javascript.language()
            language = Language(language_capsule)
            parser = Parser()
            parser.language = language
            return parser
        elif lang == "typescript":
            import tree_sitter_typescript
            from tree_sitter import Language
            language_capsule = tree_sitter_typescript.language_typescript()
            language = Language(language_capsule)
            parser = Parser()
            parser.language = language
            return parser
        elif lang == "c":
            import tree_sitter_c
            from tree_sitter import Language
            language_capsule = tree_sitter_c.language()
            language = Language(language_capsule)
            parser = Parser()
            parser.language = language
            return parser
        elif lang == "cpp":
            import tree_sitter_cpp
            from tree_sitter import Language
            language_capsule = tree_sitter_cpp.language()
            language = Language(language_capsule)
            parser = Parser()
            parser.language = language
            return parser
        elif lang == "bash":
            import tree_sitter_bash
            from tree_sitter import Language
            language_capsule = tree_sitter_bash.language()
            language = Language(language_capsule)
            parser = Parser()
            parser.language = language
            return parser
        else:
            print(f"不支持的语言: {lang}")
            return None
        
    except Exception as e:
        print(f"无法获取{lang}语言的解析器: {e}")
        return None


def get_node_text(node: Node, code_bytes: bytes) -> str:
    """
    获取节点的源代码文本
    
    Args:
        node: Tree-sitter节点
        code_bytes: 源代码字节
        
    Returns:
        节点文本
    """
    return code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def get_node_location(node: Node) -> Dict:
    """
    获取节点的位置信息
    
    Args:
        node: Tree-sitter节点
        
    Returns:
        位置信息字典
    """
    return {
        "start_line": node.start_point[0] + 1,
        "end_line": node.end_point[0] + 1,
        "start_column": node.start_point[1],
        "end_column": node.end_point[1]
    }


def find_parent_class(node: Node, code_bytes: bytes) -> Optional[Dict]:
    """
    查找父类定义
    
    Args:
        node: Tree-sitter节点
        code_bytes: 源代码字节
        
    Returns:
        父类信息或None
    """
    current = node.parent
    while current:
        if current.type in ["class_definition", "class_declaration", "struct_declaration", 
                           "type_declaration", "struct_item", "class", "defmodule", 
                           "message", "class"]:
            # 获取类名
            name_node = None
            if current.type == "class_definition":  # Python
                name_node = current.child_by_field_name("name")
            elif current.type == "class_declaration":  # Java/JS/TS/C#/Swift/Scala
                name_node = current.child_by_field_name("name")
            elif current.type == "struct_declaration":  # C
                name_node = current.child_by_field_name("name")
            elif current.type == "type_declaration":  # Go
                name_node = current.child_by_field_name("name")
            elif current.type == "struct_item":  # Rust
                name_node = current.child_by_field_name("name")
            elif current.type == "class":  # Ruby/CodeQL
                name_node = current.child_by_field_name("name")
            elif current.type == "defmodule":  # Elixir
                name_node = current.child_by_field_name("name")
            elif current.type == "message":  # Protocol Buffers
                name_node = current.child_by_field_name("name")
            
            if name_node:
                return {
                    "name": get_node_text(name_node, code_bytes),
                    "type": current.type,
                    "location": get_node_location(current),
                }
        current = current.parent
    return None


def extract_comments(node: Node, code_bytes: bytes, lang: str) -> List[Dict]:
    """
    提取节点相关的注释
    
    Args:
        node: Tree-sitter节点
        code_bytes: 源代码字节
        lang: 语言名称
        
    Returns:
        注释列表
    """
    from .config import NODE_TYPES
    
    comments = []
    comment_types = NODE_TYPES["comments"].get(lang, [])
    
    def collect_comments(n: Node):
        if n.type in comment_types:
            comment_text = get_node_text(n, code_bytes).strip()
            if comment_text:
                comments.append({
                    "text": comment_text,
                })
        for child in n.children:
            collect_comments(child)
    
    # 收集函数内部的注释
    collect_comments(node)
    
    # 收集函数前的注释（查找前一个兄弟节点）
    if node.parent:
        siblings = list(node.parent.children)
        try:
            node_index = siblings.index(node)
            if node_index > 0:
                prev_sibling = siblings[node_index - 1]
                if prev_sibling.type in comment_types:
                    comment_text = get_node_text(prev_sibling, code_bytes).strip()
                    if comment_text:
                        comments.append({
                            "text": comment_text,
                        })
        except ValueError:
            pass
    
    return comments

