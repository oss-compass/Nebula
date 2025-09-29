#!/usr/bin/env python3

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import threading
from collections import defaultdict

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_processor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProgressTracker:
    
    def __init__(self, total_repos: int):
        self.total_repos = total_repos
        self.current_repo = 0
        self.start_time = time.time()
        self.repo_times = []
        self.stage_times = defaultdict(list)
        self.lock = threading.Lock()
    
    def start_repo(self, repo_name: str):
        with self.lock:
            self.current_repo += 1
            self.repo_start_time = time.time()
            print(f"\n{'='*80}")
            print(f"[{self.current_repo}/{self.total_repos}] 开始处理仓库: {repo_name}")
            print(f"{'='*80}")
    
    def end_repo(self, repo_name: str, success: bool = True):
        with self.lock:
            repo_time = time.time() - self.repo_start_time
            self.repo_times.append(repo_time)
            
            status = "成功" if success else "失败"
            print(f"\n{status} 完成仓库 {repo_name} (耗时: {self._format_time(repo_time)})")
            
            # 显示总体进度
            avg_time = sum(self.repo_times) / len(self.repo_times)
            remaining_repos = self.total_repos - self.current_repo
            estimated_remaining = avg_time * remaining_repos
            
            print(f"总体进度: {self.current_repo}/{self.total_repos} ({self.current_repo/self.total_repos*100:.1f}%)")
            print(f"平均耗时: {self._format_time(avg_time)}")
            if remaining_repos > 0:
                print(f"预计剩余时间: {self._format_time(estimated_remaining)}")
    
    def start_stage(self, stage_name: str):
        with self.lock:
            self.stage_start_time = time.time()
            print(f"\n开始阶段: {stage_name}")
    
    def end_stage(self, stage_name: str, details: str = ""):
        with self.lock:
            stage_time = time.time() - self.stage_start_time
            self.stage_times[stage_name].append(stage_time)
            print(f"完成阶段: {stage_name} (耗时: {self._format_time(stage_time)})")
            if details:
                print(f"   {details}")
    
    def _format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            return f"{seconds/60:.1f}分钟"
        else:
            return f"{seconds/3600:.1f}小时"
    
    def get_final_stats(self) -> Dict[str, Any]:
        total_time = time.time() - self.start_time
        return {
            "total_repos": self.total_repos,
            "processed_repos": len(self.repo_times),
            "total_time": total_time,
            "average_repo_time": sum(self.repo_times) / len(self.repo_times) if self.repo_times else 0,
            "stage_times": dict(self.stage_times)
        }


class BatchProcessor:
    
    def __init__(self, 
                 txt_file: str,
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j", 
                 neo4j_password: str = "90879449Drq",
                 neo4j_database: Optional[str] = None,
                 clean_db: bool = False,
                 filter_large_functions: bool = False,
                 concurrent: int = 4,
                 batch_size: int = 32,
                 use_cache: bool = True,
                 model_name: str = "all-MiniLM-L6-v2",
                 model_type: str = "sentence_transformers"):
        self.txt_file = txt_file
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.neo4j_database = neo4j_database
        self.clean_db = clean_db
        self.filter_large_functions = filter_large_functions
        self.concurrent = concurrent
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.model_name = model_name
        self.model_type = model_type
        
        # 统计信息
        self.stats = {
            'total_repos': 0,
            'successful_repos': 0,
            'failed_repos': 0,
            'start_time': None,
            'end_time': None,
            'repo_results': []
        }
    
    def read_repo_urls(self) -> List[str]:
        try:
            with open(self.txt_file, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            # 为每个链接添加.git后缀（如果还没有的话）
            processed_urls = []
            for url in urls:
                if url.endswith('.git'):
                    processed_urls.append(url)
                else:
                    processed_urls.append(url + '.git')
            
            logger.info(f"从 {self.txt_file} 读取到 {len(processed_urls)} 个GitHub库链接")
            logger.debug(f"处理后的链接示例: {processed_urls[:3] if processed_urls else []}")
            return processed_urls
        except FileNotFoundError:
            logger.error(f"文件不存在: {self.txt_file}")
            raise
        except Exception as e:
            logger.error(f"读取文件失败: {e}")
            raise
    
    def extract_repo_name(self, url: str) -> str:
        from urllib.parse import urlparse
        from pathlib import Path
        
        # 使用与extract.main相同的逻辑
        parsed = urlparse(url)
        repo_name = Path(parsed.path).stem  # 只取路径的最后一部分，去掉.git
        
        return repo_name
    
    def run_extract(self, repo_url: str) -> Optional[str]:
        repo_name = self.extract_repo_name(repo_url)
        
        # 检查是否已经完成extract步骤
        output_file = os.path.join("output", "extract_output", f"{repo_name}_api_extraction.json")
        if os.path.exists(output_file):
            logger.info(f"extract步骤已完成，跳过: {output_file}")
            return output_file
        
        logger.info(f"开始extract流程: {repo_name}")
        
        try:
            # 构建命令
            cmd = [
                sys.executable, "-m", "extract.main", 
                repo_url
            ]
            
            if self.filter_large_functions:
                cmd.append("--filter-large-functions")
            
            # 执行命令并显示进度
            logger.info(f"执行命令: {' '.join(cmd)}")
            print(f"\n正在提取API信息: {repo_name}")
            
            # 使用实时输出显示进度
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=os.getcwd(),
                env=os.environ.copy(),
                bufsize=1,
                universal_newlines=True
            )
            
            # 实时显示输出
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # 清理输出并显示进度信息
                    line = output.strip()
                    if line and not line.startswith('DEBUG'):
                        print(f"  [文件] {line}")
                        # 强制刷新输出缓冲区
                        sys.stdout.flush()
            
            result = subprocess.CompletedProcess(
                args=cmd,
                returncode=process.returncode,
                stdout="",  # 已经实时显示了
                stderr=""
            )
            
            # 记录详细的执行结果
            logger.info(f"命令执行完成，返回码: {result.returncode}")
            if result.stdout:
                logger.debug(f"标准输出: {result.stdout}")
            if result.stderr:
                logger.debug(f"标准错误: {result.stderr}")
            
            if result.returncode == 0:
                output_file = os.path.join("output", "extract_output", f"{repo_name}_api_extraction.json")
                if os.path.exists(output_file):
                    # 检查是否找到了函数
                    try:
                        with open(output_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            total_functions = data.get('total_functions', 0)
                            if total_functions == 0:
                                logger.warning(f"extract完成但未找到任何函数: {repo_name} (可能是非支持语言的库或空库)")
                                return "NO_FUNCTIONS_FOUND"  # 特殊标记表示没有找到函数
                            else:
                                logger.info(f"extract完成: {output_file} (找到 {total_functions} 个函数)")
                                return output_file
                    except Exception as e:
                        logger.error(f"无法解析extract输出文件: {output_file}, 错误: {str(e)}")
                        return None
                else:
                    logger.error(f"extract输出文件不存在: {output_file}")
                    return None
            else:
                # 检查是否是KeyError相关的错误
                if "KeyError" in result.stderr and "direct_call_weight" in result.stderr:
                    logger.warning(f"extract遇到KeyError，但可能已生成部分结果: {repo_name}")
                    # 检查是否仍然生成了输出文件
                    output_file = os.path.join("output", "extract_output", f"{repo_name}_api_extraction.json")
                    if os.path.exists(output_file):
                        logger.info(f"尽管有KeyError，但找到了输出文件: {output_file}")
                        return output_file
                    else:
                        logger.error(f"KeyError导致extract完全失败: {repo_name}")
                        return None
                else:
                    logger.error(f"extract失败，返回码: {result.returncode}")
                    logger.error(f"错误输出: {result.stderr}")
                    if result.stdout:
                        logger.error(f"标准输出: {result.stdout}")
                    return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"extract超时: {repo_name}")
            return None
        except Exception as e:
            logger.error(f"extract异常: {repo_name}, 错误: {str(e)}")
            return None
    
    def run_description(self, extraction_file: str) -> Optional[str]:
        # 从完整路径中提取repo_name
        filename = os.path.basename(extraction_file)
        repo_name = filename.replace("_api_extraction.json", "")
        
        # 检查是否已经完成description步骤
        complete_file = os.path.join("output", "description_output", "complete", f"{repo_name}_complete_for_neo4j.json")
        if os.path.exists(complete_file):
            logger.info(f"description步骤已完成，跳过: {complete_file}")
            return complete_file
        
        logger.info(f"开始description流程: {repo_name}")
        
        try:
            # 构建命令，添加--skip-complexity参数以使用默认moderate复杂度/重要度
            cmd = [
                sys.executable, "-m", "description.main",
                extraction_file,
                "--skip-complexity"
            ]
            
            # 执行命令并显示进度
            logger.info(f"执行命令: {' '.join(cmd)}")
            print(f"\n正在生成函数描述: {repo_name}")
            
            # 使用实时输出显示进度
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=os.getcwd(),
                env=os.environ.copy(),
                bufsize=1,
                universal_newlines=True
            )
            
            # 实时显示输出
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # 清理输出并显示进度信息
                    line = output.strip()
                    if line and not line.startswith('DEBUG'):
                        print(f"  [AI] {line}")
                        # 强制刷新输出缓冲区
                        sys.stdout.flush()
            
            result = subprocess.CompletedProcess(
                args=cmd,
                returncode=process.returncode,
                stdout="",  # 已经实时显示了
                stderr=""
            )
            
            # 记录详细的执行结果
            logger.info(f"命令执行完成，返回码: {result.returncode}")
            if result.stdout:
                logger.debug(f"标准输出: {result.stdout}")
            if result.stderr:
                logger.debug(f"标准错误: {result.stderr}")
            
            if result.returncode == 0:
                complete_file = os.path.join("output", "description_output", "complete", f"{repo_name}_complete_for_neo4j.json")
                if os.path.exists(complete_file):
                    logger.info(f"description完成: {complete_file}")
                    return complete_file
                else:
                    logger.error(f"description输出文件不存在: {complete_file}")
                    return None
            else:
                logger.error(f"description失败，返回码: {result.returncode}")
                logger.error(f"错误输出: {result.stderr}")
                if result.stdout:
                    logger.error(f"标准输出: {result.stdout}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"description超时: {repo_name}")
            return None
        except Exception as e:
            logger.error(f"description异常: {e}")
            return None
    
    def run_vector_embedding(self, complete_file: str) -> bool:
        # 从完整路径中提取repo_name
        filename = os.path.basename(complete_file)
        repo_name = filename.replace("_complete_for_neo4j.json", "")
        
        # 检查是否已经完成vector_embedding步骤
        index_file = os.path.join("vector_embedding_output", "embeddings", f"{repo_name}_index_embeddings.json")
        if os.path.exists(index_file):
            logger.info(f"vector_embedding步骤已完成，跳过: {index_file}")
            return True
        
        logger.info(f"开始vector_embedding流程: {repo_name}")
        print(f"\n正在生成向量嵌入: {repo_name}")
        
        try:
            # 构建命令
            # 正确处理文件路径和变量，避免转义问题
            import json as json_module
            complete_file_escaped = json_module.dumps(complete_file)
            model_name_escaped = json_module.dumps(self.model_name)
            model_type_escaped = json_module.dumps(self.model_type)
            # 布尔值需要特殊处理，因为json.dumps会输出小写的true/false
            use_cache_escaped = str(self.use_cache)
            repo_name_escaped = json_module.dumps(repo_name)
            
            cmd = [
                sys.executable, "-c",
                f"""
import sys
sys.path.append('.')
from semantic_search.vector_embedding import CodeEmbeddingManager, EmbeddingConfig
import json

print("  加载数据...")
# 加载数据
complete_file = {complete_file_escaped}
with open(complete_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("  创建配置...")
# 创建配置
config = EmbeddingConfig(
    model_name={model_name_escaped},
    model_type={model_type_escaped},
    cache_dir='./embedding_cache',
    use_cache={use_cache_escaped}
)

# 创建管理器
manager = CodeEmbeddingManager(config)

# 提取函数数据
functions = data.get('functions', [])
if not functions:
    print("没有找到函数数据")
    sys.exit(1)

# 准备嵌入数据
texts = []
metadata_list = []

for func in functions:
    # 提取函数代码
    source_code = func.get('basic_info', {{}}).get('source_code', '')
    if not source_code:
        continue
    
    # 提取元数据
    metadata = {{
        'name': func.get('basic_info', {{}}).get('function_name', ''),
        'filepath': func.get('context', {{}}).get('file_path', ''),
        'parent_class_name': func.get('context', {{}}).get('parent_class', ''),
        'parameters_count': func.get('complexity', {{}}).get('semantic_complexity', {{}}).get('parameters', 0),
        'return_type': func.get('basic_info', {{}}).get('return_type', ''),
        'docstring_description': func.get('description_info', {{}}).get('docstring', {{}}).get('description', ''),
        'complexity_score': func.get('complexity', {{}}).get('semantic_complexity', {{}}).get('complexity_score', 0),
        'function_type': 'test' if func.get('basic_info', {{}}).get('function_name', '').startswith('test_') else 'regular',
        'repo_name': {repo_name_escaped}
    }}
    
    texts.append(source_code)
    metadata_list.append(metadata)

if not texts:
    print("没有有效的函数代码")
    sys.exit(1)

print(f"开始生成嵌入，共 {{len(texts)}} 个函数")

# 生成嵌入
embeddings = manager.vectorizer.embed_texts(texts, metadata_list)
manager.add_embeddings(embeddings)

# 创建输出目录和子目录
import os
output_dir = os.path.join('output', 'vector_embedding_output')
embeddings_dir = os.path.join(output_dir, 'embeddings')
vectorizers_dir = os.path.join(output_dir, 'vectorizers')

os.makedirs(embeddings_dir, exist_ok=True)
os.makedirs(vectorizers_dir, exist_ok=True)

# 保存索引到embeddings子文件夹
repo_name = {repo_name_escaped}
index_file = os.path.join(embeddings_dir, f'{{repo_name}}_index_embeddings.json')
manager.save_index(index_file)

# 保存向量化器到vectorizers子文件夹（如果是TF-IDF）
if config.model_type == 'tfidf':
    vectorizer_file = os.path.join(vectorizers_dir, f'{{repo_name}}_index_vectorizer.pkl')
    manager.vectorizer.save_vectorizer(vectorizer_file)

# 显示统计信息
stats = manager.get_stats()
print(f"嵌入完成: {{stats}}")

manager.close()
print("vector_embedding流程完成")
"""
            ]
            
            # 执行命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30分钟超时
            )
            
            if result.returncode == 0:
                index_file = os.path.join("vector_embedding_output", "embeddings", f"{repo_name}_index_embeddings.json")
                if os.path.exists(index_file):
                    logger.info(f"vector_embedding完成: {index_file}")
                    return True
                else:
                    logger.error(f"vector_embedding输出文件不存在: {index_file}")
                    return False
            else:
                logger.error(f"vector_embedding失败: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"vector_embedding超时: {repo_name}")
            return False
        except Exception as e:
            logger.error(f"vector_embedding异常: {e}")
            return False
    
    def run_ingest(self, complete_file: str) -> bool:
        # 从完整路径中提取repo_name
        filename = os.path.basename(complete_file)
        repo_name = filename.replace("_complete_for_neo4j.json", "")
        
        # 检查是否已经完成ingest步骤（通过检查标记文件）
        ingest_marker_file = os.path.join("output", "ingest_output", f"{repo_name}_ingest_complete.txt")
        if os.path.exists(ingest_marker_file):
            logger.info(f"ingest步骤已完成，跳过: {repo_name}")
            return True
        
        logger.info(f"开始ingest流程: {repo_name}")
        print(f"\n正在导入Neo4j数据库: {repo_name}")
        
        try:
            # 构建命令
            cmd = [
                sys.executable, "ingest.py",
                complete_file,
                "--uri", self.neo4j_uri,
                "--user", self.neo4j_user,
                "--password", self.neo4j_password
            ]
            
            if self.neo4j_database:
                cmd.extend(["--database", self.neo4j_database])
            
            if self.clean_db:
                cmd.append("--clean-db")
            
            # 执行命令并显示进度
            logger.info(f"执行命令: {' '.join(cmd)}")
            
            # 使用实时输出显示进度
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=os.getcwd(),
                env=os.environ.copy(),
                bufsize=1,
                universal_newlines=True
            )
            
            # 实时显示输出
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # 清理输出并显示进度信息
                    line = output.strip()
                    if line and not line.startswith('DEBUG'):
                        print(f"  [DB] {line}")
                        # 强制刷新输出缓冲区
                        sys.stdout.flush()
            
            result = subprocess.CompletedProcess(
                args=cmd,
                returncode=process.returncode,
                stdout="",  # 已经实时显示了
                stderr=""
            )
            
            if result.returncode == 0:
                logger.info(f"ingest完成: {repo_name}")
                # 创建标记文件表示ingest已完成
                os.makedirs("output/ingest_output", exist_ok=True)
                with open(ingest_marker_file, 'w', encoding='utf-8') as f:
                    f.write(f"ingest completed at {datetime.now().isoformat()}\n")
                return True
            else:
                logger.error(f"ingest失败: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"ingest超时: {repo_name}")
            return False
        except Exception as e:
            logger.error(f"ingest异常: {e}")
            return False
    
    def process_single_repo(self, repo_url: str, progress_tracker: ProgressTracker = None) -> Dict[str, Any]:
        repo_name = self.extract_repo_name(repo_url)
        result = {
            'repo_url': repo_url,
            'repo_name': repo_name,
            'success': False,
            'steps': {
                'extract': False,
                'description': False,
                'vector_embedding': False,
                'ingest': False
            },
            'error': None,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'step_details': {}
        }
        
        try:
            if progress_tracker:
                progress_tracker.start_repo(repo_name)
            
            logger.info(f"开始处理仓库: {repo_name} ({repo_url})")
            
            # Step 1: Extract
            if progress_tracker:
                progress_tracker.start_stage("API提取 (Extract)")
            
            extraction_file = self.run_extract(repo_url)
            if extraction_file:
                result['steps']['extract'] = True
                
                # 检查是否是没有找到函数的情况
                if extraction_file == "NO_FUNCTIONS_FOUND":
                    logger.info(f"跳过后续步骤: {repo_name} (未找到任何函数)")
                    result['step_details']['extract'] = {
                        'function_count': 0,
                        'file_count': 0,
                        'skip_reason': 'No functions found (possibly non-supported language repository or empty repository)'
                    }
                    result['steps']['description'] = True  # 标记为跳过
                    result['steps']['vector_embedding'] = True  # 标记为跳过
                    result['steps']['ingest'] = True  # 标记为跳过
                    result['step_details']['description'] = {'skip_reason': 'No functions to process'}
                    result['step_details']['vector_embedding'] = {'skip_reason': 'No functions to process'}
                    result['step_details']['ingest'] = {'skip_reason': 'No functions to process'}
                    
                    if progress_tracker:
                        progress_tracker.end_stage("API提取 (Extract)", "未找到任何函数，跳过后续步骤")
                        progress_tracker.end_stage("函数描述生成 (Description)", "跳过 (无函数)")
                        progress_tracker.end_stage("向量嵌入 (Vector Embedding)", "跳过 (无函数)")
                        progress_tracker.end_stage("Neo4j导入 (Ingest)", "跳过 (无函数)")
                        progress_tracker.end_repo(repo_name, True)
                    
                    result['end_time'] = datetime.now().isoformat()
                    return result
                
                # 获取extract结果的统计信息
                try:
                    with open(extraction_file, 'r', encoding='utf-8') as f:
                        extract_data = json.load(f)
                        function_count = len(extract_data.get('functions', []))
                        result['step_details']['extract'] = {
                            'function_count': function_count,
                            'file_count': extract_data.get('total_files', 0)
                        }
                        extract_details = f"提取了 {function_count} 个函数，{extract_data.get('total_files', 0)} 个文件"
                except:
                    extract_details = "提取完成"
                
                if progress_tracker:
                    progress_tracker.end_stage("API提取 (Extract)", extract_details)
            else:
                result['error'] = "extract步骤失败"
                if progress_tracker:
                    progress_tracker.end_repo(repo_name, False)
                return result
            
            # Step 2: Description
            if progress_tracker:
                progress_tracker.start_stage("函数描述生成 (Description)")
            
            complete_file = self.run_description(extraction_file)
            if complete_file:
                result['steps']['description'] = True
                # 获取description结果的统计信息
                try:
                    with open(complete_file, 'r', encoding='utf-8') as f:
                        desc_data = json.load(f)
                        desc_count = len(desc_data.get('functions', []))
                        result['step_details']['description'] = {
                            'function_count': desc_count
                        }
                        desc_details = f"生成了 {desc_count} 个函数的描述"
                except:
                    desc_details = "描述生成完成"
                
                if progress_tracker:
                    progress_tracker.end_stage("函数描述生成 (Description)", desc_details)
            else:
                result['error'] = "description步骤失败"
                if progress_tracker:
                    progress_tracker.end_repo(repo_name, False)
                return result
            
            # Step 3: Vector Embedding
            if progress_tracker:
                progress_tracker.start_stage("向量嵌入 (Vector Embedding)")
            
            if self.run_vector_embedding(complete_file):
                result['steps']['vector_embedding'] = True
                if progress_tracker:
                    progress_tracker.end_stage("向量嵌入 (Vector Embedding)", "向量嵌入完成")
            else:
                result['error'] = "vector_embedding步骤失败"
                if progress_tracker:
                    progress_tracker.end_repo(repo_name, False)
                return result
            
            # Step 4: Ingest
            if progress_tracker:
                progress_tracker.start_stage("数据导入 (Neo4j Ingest)")
            
            if self.run_ingest(complete_file):
                result['steps']['ingest'] = True
                result['success'] = True
                if progress_tracker:
                    progress_tracker.end_stage("数据导入 (Neo4j Ingest)", "数据导入Neo4j完成")
            else:
                result['error'] = "ingest步骤失败"
                if progress_tracker:
                    progress_tracker.end_repo(repo_name, False)
                return result
            
        except Exception as e:
            result['error'] = f"处理异常: {str(e)}"
            logger.error(f"处理仓库异常 {repo_name}: {e}")
            if progress_tracker:
                progress_tracker.end_repo(repo_name, False)
        
        finally:
            result['end_time'] = datetime.now().isoformat()
            if progress_tracker and result['success']:
                progress_tracker.end_repo(repo_name, True)
        
        return result
    
    def process_all_repos(self):
        # 读取仓库链接
        repo_urls = self.read_repo_urls()
        self.stats['total_repos'] = len(repo_urls)
        self.stats['start_time'] = datetime.now().isoformat()
        
        logger.info(f"开始批量处理 {len(repo_urls)} 个仓库")
        
        # 处理每个仓库
        for i, repo_url in enumerate(repo_urls, 1):
            logger.info(f"处理进度: {i}/{len(repo_urls)}")
            
            result = self.process_single_repo(repo_url)
            self.stats['repo_results'].append(result)
            
            if result['success']:
                self.stats['successful_repos'] += 1
                logger.info(f"仓库处理成功: {result['repo_name']}")
            else:
                self.stats['failed_repos'] += 1
                logger.error(f"仓库处理失败: {result['repo_name']} - {result['error']}")
            
            # 显示进度
            progress = (i / len(repo_urls)) * 100
            logger.info(f"总体进度: {progress:.1f}% ({i}/{len(repo_urls)})")
        
        self.stats['end_time'] = datetime.now().isoformat()
        
        # 保存统计信息
        self.save_statistics()
        
        # 显示最终结果
        self.print_final_summary()
    
    def save_statistics(self):
        stats_file = f"batch_processing_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        logger.info(f"统计信息已保存到: {stats_file}")
    
    def print_final_summary(self):
        logger.info("=" * 60)
        logger.info("批量处理完成！")
        logger.info("=" * 60)
        logger.info(f"总仓库数: {self.stats['total_repos']}")
        logger.info(f"成功处理: {self.stats['successful_repos']}")
        logger.info(f"处理失败: {self.stats['failed_repos']}")
        logger.info(f"成功率: {(self.stats['successful_repos'] / self.stats['total_repos'] * 100):.1f}%")
        
        if self.stats['start_time'] and self.stats['end_time']:
            start = datetime.fromisoformat(self.stats['start_time'])
            end = datetime.fromisoformat(self.stats['end_time'])
            duration = end - start
            logger.info(f"总耗时: {duration}")
        
        # 显示失败的仓库
        failed_repos = [r for r in self.stats['repo_results'] if not r['success']]
        if failed_repos:
            logger.info("\n失败的仓库:")
            for repo in failed_repos:
                logger.info(f"  - {repo['repo_name']}: {repo['error']}")
        
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='批量处理多个GitHub库')
    parser.add_argument('txt_file', help='包含GitHub库链接的txt文件路径')
    parser.add_argument('--neo4j-uri', default='bolt://localhost:7687', help='Neo4j数据库URI')
    parser.add_argument('--neo4j-user', default='neo4j', help='Neo4j用户名')
    parser.add_argument('--neo4j-password', default='90879449Drq', help='Neo4j密码')
    parser.add_argument('--neo4j-database', help='Neo4j数据库名')
    parser.add_argument('--clean-db', action='store_true', help='清空数据库')
    parser.add_argument('--filter-large-functions', action='store_true', help='过滤大函数')
    parser.add_argument('--concurrent', type=int, default=4, help='并发数')
    parser.add_argument('--batch-size', type=int, default=32, help='批处理大小')
    parser.add_argument('--use-cache', action='store_true', default=True, help='使用缓存')
    parser.add_argument('--model-name', default='all-MiniLM-L6-v2', help='嵌入模型名称')
    parser.add_argument('--model-type', default='sentence_transformers', help='嵌入模型类型')
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = BatchProcessor(
        txt_file=args.txt_file,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        neo4j_database=args.neo4j_database,
        clean_db=args.clean_db,
        filter_large_functions=args.filter_large_functions,
        concurrent=args.concurrent,
        batch_size=args.batch_size,
        use_cache=args.use_cache,
        model_name=args.model_name,
        model_type=args.model_type
    )
    
    # 开始处理
    try:
        processor.process_all_repos()
    except KeyboardInterrupt:
        logger.info("用户中断处理")
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()
