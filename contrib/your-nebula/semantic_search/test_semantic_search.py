#!/usr/bin/env python3
"""
语义搜索功能测试脚本
"""

import os
import sys
import tempfile
import shutil
import json
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """测试基本功能"""
    print("🧪 Testing Semantic Search Basic Functionality")
    print("=" * 60)
    
    # 创建测试目录和文件
    test_dir = tempfile.mkdtemp()
    test_file = os.path.join(test_dir, "test.py")
    
    test_code = '''
def calculate_fibonacci(n):
    """Calculate the nth Fibonacci number"""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def binary_search(arr, target):
    """Binary search in sorted array"""
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

def linear_search(arr, target):
    """Linear search in array"""
    for i, item in enumerate(arr):
        if item == target:
            return i
    return -1

class SearchAlgorithms:
    """Collection of search algorithms"""
    
    def __init__(self):
        self.algorithms = {
            'binary': binary_search,
            'linear': linear_search
        }
    
    def search(self, arr, target, algorithm='binary'):
        """Search using specified algorithm"""
        if algorithm in self.algorithms:
            return self.algorithms[algorithm](arr, target)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
'''
    
    with open(test_file, 'w') as f:
        f.write(test_code)
    
    try:
        # 测试导入
        print("1. Testing imports...")
        try:
            from .vector_embedding import create_default_config, CodeEmbeddingManager
            from sync_indexer import create_default_sync_config, SyncSemanticIndexer
            from semantic_search import create_advanced_searcher, SearchQuery
            print("   ✅ All imports successful")
        except ImportError as e:
            print(f"   ❌ Import error: {e}")
            return False
        
        # 测试配置创建
        print("2. Testing configuration creation...")
        try:
            embedding_config = create_default_config()
            sync_config = create_default_sync_config(test_dir)
            print("   ✅ Configuration creation successful")
        except Exception as e:
            print(f"   ❌ Configuration error: {e}")
            return False
        
        # 测试索引器创建
        print("3. Testing indexer creation...")
        try:
            indexer = SyncSemanticIndexer(sync_config)
            print("   ✅ Indexer creation successful")
        except Exception as e:
            print(f"   ❌ Indexer creation error: {e}")
            return False
        
        # 测试索引
        print("4. Testing repository indexing...")
        try:
            stats = indexer.index_repository()
            print(f"   ✅ Indexing successful: {stats}")
        except Exception as e:
            print(f"   ❌ Indexing error: {e}")
            return False
        
        # 测试搜索器创建
        print("5. Testing searcher creation...")
        try:
            searcher = create_advanced_searcher(indexer)
            print("   ✅ Searcher creation successful")
        except Exception as e:
            print(f"   ❌ Searcher creation error: {e}")
            return False
        
        # 测试搜索
        print("6. Testing semantic search...")
        try:
            query = SearchQuery(
                query="find element in array",
                query_type="semantic",
                top_k=5
            )
            results = searcher.search(query)
            print(f"   ✅ Search successful: found {len(results)} results")
            
            for i, result in enumerate(results[:3], 1):
                context = result.context or {}
                print(f"      {i}. {context.get('function_name', 'Unknown')} "
                      f"(similarity: {result.similarity:.4f})")
        except Exception as e:
            print(f"   ❌ Search error: {e}")
            return False
        
        # 测试统计信息
        print("7. Testing statistics...")
        try:
            stats = indexer.get_stats()
            print(f"   ✅ Statistics: {stats.get('total_functions', 0)} functions indexed")
        except Exception as e:
            print(f"   ❌ Statistics error: {e}")
            return False
        
        # 清理
        indexer.close()
        
        print("\n🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False
    
    finally:
        # 清理测试目录
        shutil.rmtree(test_dir)


def test_hybrid_search():
    """测试混合搜索功能"""
    print("\n🔗 Testing Hybrid Search Functionality")
    print("=" * 60)
    
    # 创建测试目录和文件
    test_dir = tempfile.mkdtemp()
    test_file = os.path.join(test_dir, "test.py")
    
    test_code = '''
def complex_algorithm(data):
    """Complex algorithm for data processing"""
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
        else:
            result.append(item)
    return result

def simple_function(x):
    """Simple function"""
    return x + 1
'''
    
    with open(test_file, 'w') as f:
        f.write(test_code)
    
    try:
        # 测试混合搜索
        print("1. Testing hybrid search setup...")
        try:
            from sync_indexer import create_default_sync_config, SyncSemanticIndexer
            from semantic_search import create_advanced_searcher, SearchQuery
            from hybrid_search import create_hybrid_searcher, create_default_hybrid_config
            
            sync_config = create_default_sync_config(test_dir)
            indexer = SyncSemanticIndexer(sync_config)
            indexer.index_repository()
            
            semantic_searcher = create_advanced_searcher(indexer)
            hybrid_config = create_default_hybrid_config()
            hybrid_searcher = create_hybrid_searcher(semantic_searcher, hybrid_config)
            
            print("   ✅ Hybrid searcher setup successful")
        except Exception as e:
            print(f"   ❌ Hybrid searcher setup error: {e}")
            return False
        
        # 测试混合搜索
        print("2. Testing hybrid search...")
        try:
            query = SearchQuery(
                query="complex data processing",
                query_type="semantic",
                top_k=3
            )
            results = hybrid_searcher.search(query)
            print(f"   ✅ Hybrid search successful: found {len(results)} results")
            
            for i, result in enumerate(results[:2], 1):
                print(f"      {i}. {result.embedding.metadata.get('name', 'Unknown')} "
                      f"(vector: {result.vector_similarity:.4f}, "
                      f"graph: {result.graph_score:.4f}, "
                      f"combined: {result.combined_score:.4f})")
        except Exception as e:
            print(f"   ❌ Hybrid search error: {e}")
            return False
        
        # 清理
        hybrid_searcher.close()
        indexer.close()
        
        print("\n🎉 Hybrid search tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Hybrid search test failed with error: {e}")
        return False
    
    finally:
        # 清理测试目录
        shutil.rmtree(test_dir)


def test_cli_functionality():
    """测试CLI功能"""
    print("\n🖥️ Testing CLI Functionality")
    print("=" * 60)
    
    # 创建测试目录和文件
    test_dir = tempfile.mkdtemp()
    test_file = os.path.join(test_dir, "test.py")
    
    test_code = '''
def api_endpoint(request):
    """Handle API request"""
    return {"status": "success", "data": request}

def process_data(data):
    """Process incoming data"""
    return [item * 2 for item in data]
'''
    
    with open(test_file, 'w') as f:
        f.write(test_code)
    
    try:
        # 测试CLI创建
        print("1. Testing CLI creation...")
        try:
            from sync_indexer import create_default_sync_config, SyncSemanticIndexer
            from semantic_search import create_advanced_searcher
            from cli import create_cli
            
            sync_config = create_default_sync_config(test_dir)
            indexer = SyncSemanticIndexer(sync_config)
            indexer.index_repository()
            
            searcher = create_advanced_searcher(indexer)
            cli = create_cli(indexer, searcher)
            print("   ✅ CLI creation successful")
        except Exception as e:
            print(f"   ❌ CLI creation error: {e}")
            return False
        
        # 清理
        indexer.close()
        
        print("\n🎉 CLI tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ CLI test failed with error: {e}")
        return False
    
    finally:
        # 清理测试目录
        shutil.rmtree(test_dir)


def main():
    """主测试函数"""
    print("🚀 Semantic Search System Test Suite")
    print("=" * 80)
    
    all_passed = True
    
    # 运行基本功能测试
    if not test_basic_functionality():
        all_passed = False
    
    # 运行混合搜索测试
    if not test_hybrid_search():
        all_passed = False
    
    # 运行CLI测试
    if not test_cli_functionality():
        all_passed = False
    
    # 总结
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 All tests passed! The semantic search system is working correctly.")
        print("\nNext steps:")
        print("1. Install required dependencies: pip install -r requirements.txt")
        print("2. Index your repository: python main.py index --repo-path /path/to/repo")
        print("3. Start the search server: python main.py server --repo-path /path/to/repo --load-index ./index.json")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Make sure all required dependencies are installed")
        print("2. Check that the sync library is available")
        print("3. Verify Neo4j is running (for hybrid search)")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
