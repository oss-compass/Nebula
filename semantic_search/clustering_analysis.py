#!/usr/bin/env python3
"""
代码聚类和相似性分析测试脚本
专门用于测试语义搜索系统中的代码聚类、相似性分析、重复代码检测等功能
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_code_clustering():
    """测试代码聚类功能"""
    print("🔍 Testing Code Clustering Functionality")
    print("=" * 60)
    
    # 创建测试目录和文件
    test_dir = tempfile.mkdtemp()
    
    # 创建多个测试文件，包含不同类型的函数
    test_files = {
        "search_algorithms.py": '''
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

def interpolation_search(arr, target):
    """Interpolation search in sorted array"""
    left, right = 0, len(arr) - 1
    while left <= right and arr[left] <= target <= arr[right]:
        pos = left + int(((target - arr[left]) * (right - left)) / (arr[right] - arr[left]))
        if arr[pos] == target:
            return pos
        elif arr[pos] < target:
            left = pos + 1
        else:
            right = pos - 1
    return -1
''',
        "math_functions.py": '''
def fibonacci(n):
    """Calculate nth Fibonacci number"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def factorial(n):
    """Calculate factorial of n"""
    if n <= 1:
        return 1
    return n * factorial(n-1)

def power(base, exponent):
    """Calculate base raised to exponent"""
    if exponent == 0:
        return 1
    return base * power(base, exponent - 1)
''',
        "string_utils.py": '''
def reverse_string(s):
    """Reverse a string"""
    return s[::-1]

def is_palindrome(s):
    """Check if string is palindrome"""
    return s == s[::-1]

def count_vowels(s):
    """Count vowels in string"""
    vowels = 'aeiouAEIOU'
    return sum(1 for char in s if char in vowels)
''',
        "data_processing.py": '''
def filter_positive_numbers(numbers):
    """Filter positive numbers from list"""
    return [num for num in numbers if num > 0]

def map_double(numbers):
    """Double each number in list"""
    return [num * 2 for num in numbers]

def reduce_sum(numbers):
    """Sum all numbers in list"""
    return sum(numbers)
'''
    }
    
    # 写入测试文件
    for filename, content in test_files.items():
        filepath = os.path.join(test_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content)
    
    try:
        # 导入必要的模块
        print("1. Setting up indexer and searcher...")
        from sync_indexer import create_default_sync_config, SyncSemanticIndexer
        from semantic_search import create_advanced_searcher
        
        # 创建配置和索引器
        sync_config = create_default_sync_config(test_dir)
        indexer = SyncSemanticIndexer(sync_config)
        indexer.index_repository()
        
        # 创建高级搜索器
        searcher = create_advanced_searcher(indexer)
        print("   ✅ Setup successful")
        
        # 测试聚类功能
        print("2. Testing function clustering...")
        try:
            clusters = searcher.cluster_similar_functions(n_clusters=3)
            print(f"   ✅ Clustering successful: found {len(clusters)} clusters")
            
            # 显示聚类结果
            for cluster_id, functions in clusters.items():
                print(f"   Cluster {cluster_id}: {len(functions)} functions")
                for func in functions[:3]:  # 只显示前3个
                    context = func.context or {}
                    print(f"     - {context.get('function_name', 'Unknown')} "
                          f"({context.get('file_path', 'Unknown')})")
                if len(functions) > 3:
                    print(f"     ... and {len(functions) - 3} more")
                print()
        except Exception as e:
            print(f"   ❌ Clustering error: {e}")
            return False
        
        # 测试相似性分析
        print("3. Testing similarity analysis...")
        try:
            # 获取一个函数作为示例
            embeddings_list = list(indexer.embedding_manager.embeddings_index.values())
            if embeddings_list:
                example_function = embeddings_list[0]
                recommendations = searcher.get_recommendations(example_function, top_k=3)
                print(f"   ✅ Similarity analysis successful: found {len(recommendations)} similar functions")
                
                context = example_function.metadata
                print(f"   Example function: {context.get('name', 'Unknown')}")
                print("   Similar functions:")
                for rec in recommendations:
                    rec_context = rec.context or {}
                    print(f"     - {rec_context.get('function_name', 'Unknown')} "
                          f"(similarity: {rec.similarity:.4f})")
            else:
                print("   ⚠️ No functions found for similarity analysis")
        except Exception as e:
            print(f"   ❌ Similarity analysis error: {e}")
            return False
        
        # 测试重复代码检测
        print("4. Testing duplicate code detection...")
        try:
            duplicates = searcher.find_duplicate_code(threshold=0.8)
            print(f"   ✅ Duplicate detection successful: found {len(duplicates)} duplicate groups")
            
            for i, group in enumerate(duplicates[:2]):  # 只显示前2组
                print(f"   Duplicate group {i+1}: {len(group)} functions")
                for func in group:
                    context = func.context or {}
                    print(f"     - {context.get('function_name', 'Unknown')} "
                          f"({context.get('file_path', 'Unknown')})")
                print()
        except Exception as e:
            print(f"   ❌ Duplicate detection error: {e}")
            return False
        
        # 测试复杂度分析
        print("5. Testing complexity analysis...")
        try:
            complexity_analysis = searcher.get_code_complexity_analysis()
            print("   ✅ Complexity analysis successful")
            print(f"   Total functions: {complexity_analysis.get('total_functions', 0)}")
            print(f"   Average complexity: {complexity_analysis.get('average_complexity', 0):.2f}")
            print(f"   Max complexity: {complexity_analysis.get('max_complexity', 0)}")
            print(f"   Function types: {complexity_analysis.get('function_type_distribution', {})}")
        except Exception as e:
            print(f"   ❌ Complexity analysis error: {e}")
            return False
        
        # 清理
        indexer.close()
        
        print("\n🎉 Code clustering and analysis tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False
    
    finally:
        # 清理测试目录
        shutil.rmtree(test_dir)


def test_attrs_index_clustering():
    """使用attrs_index.json测试聚类功能"""
    print("\n📊 Testing Clustering with attrs_index.json")
    print("=" * 60)
    
    # 检查attrs_index.json是否存在
    attrs_index_path = "attrs_index.json"
    if not os.path.exists(attrs_index_path):
        print(f"❌ attrs_index.json not found at {attrs_index_path}")
        print("Please make sure the index file exists in the current directory")
        return False
    
    try:
        # 导入必要的模块
        print("1. Loading attrs index...")
        from sync_indexer import create_default_sync_config, SyncSemanticIndexer
        from semantic_search import create_advanced_searcher
        
        # 创建配置（使用attrs项目路径）
        attrs_repo_path = "F:\\aaa_school\\OSPP\\osscompass\\target_project\\attrs"
        if not os.path.exists(attrs_repo_path):
            print(f"❌ Attrs repository not found at {attrs_repo_path}")
            print("Please update the path in the script")
            return False
        
        sync_config = create_default_sync_config(attrs_repo_path)
        # 使用TF-IDF模型避免网络连接问题
        sync_config.embedding_config.model_type = "tfidf"
        indexer = SyncSemanticIndexer(sync_config)
        
        # 加载现有索引
        indexer.load_index(attrs_index_path)
        print("   ✅ Index loaded successfully")
        
        # 创建高级搜索器
        searcher = create_advanced_searcher(indexer)
        print("   ✅ Searcher created successfully")
        
        # 获取索引统计信息
        stats = indexer.get_stats()
        print(f"   Index contains {stats.get('total_functions', 0)} functions")
        
        # 测试聚类功能
        print("2. Testing function clustering on attrs codebase...")
        try:
            # 使用更多的聚类数量，因为attrs是一个较大的项目
            clusters = searcher.cluster_similar_functions(n_clusters=10)
            print(f"   ✅ Clustering successful: found {len(clusters)} clusters")
            
            # 显示聚类结果
            print("\n   Clustering Results:")
            print("   " + "="*50)
            for cluster_id, functions in clusters.items():
                print(f"   Cluster {cluster_id}: {len(functions)} functions")
                
                # 显示每个聚类中的前几个函数
                for i, func in enumerate(functions[:5]):  # 显示前5个
                    context = func.context or {}
                    func_name = context.get('function_name', 'Unknown')
                    file_path = context.get('file_path', 'Unknown')
                    # 只显示文件名，不显示完整路径
                    file_name = os.path.basename(file_path) if file_path != 'Unknown' else 'Unknown'
                    print(f"     {i+1}. {func_name} ({file_name})")
                
                if len(functions) > 5:
                    print(f"     ... and {len(functions) - 5} more functions")
                print()
        except Exception as e:
            print(f"   ❌ Clustering error: {e}")
            return False
        
        # 测试相似性分析
        print("3. Testing similarity analysis on attrs functions...")
        try:
            # 测试指定的三个函数
            test_functions = ["attrib (_make.py)", "optional (converters.py)", "convert (setters.py)"]
            
            for func_name in test_functions:
                print(f"\n   Function: {func_name}")
                try:
                    # 通过函数名在索引中查找对应的embedding
                    target_embedding = None
                    for embedding in indexer.embedding_manager.embeddings_index.values():
                        context = embedding.metadata
                        func_name_from_meta = context.get('name', '')
                        file_path = context.get('file_path', '')
                        file_name = os.path.basename(file_path) if file_path else ''
                        
                        # 检查是否匹配函数名和文件名
                        if func_name_from_meta in func_name and file_name in func_name:
                            target_embedding = embedding
                            break
                    
                    if target_embedding:
                        # 使用get_recommendations方法查找相似函数
                        recommendations = searcher.get_recommendations(target_embedding, top_k=5)
                        if recommendations:
                            print(f"   Similar functions ({len(recommendations)} found):")
                            for i, rec in enumerate(recommendations, 1):
                                # 从embedding的metadata中获取信息
                                rec_metadata = rec.embedding.metadata
                                
                                # 提取函数名和文件路径
                                rec_name = rec_metadata.get('name', 'Unknown')
                                rec_file_path = rec_metadata.get('filepath', 'Unknown')  # 注意：metadata中是'filepath'不是'file_path'
                                rec_file = os.path.basename(rec_file_path) if rec_file_path != 'Unknown' else 'Unknown'
                                
                                print(f"     {i}. {rec_name} ({rec_file}) - similarity: {rec.similarity:.4f}")
                        else:
                            print("   Similar functions (0 found):")
                    else:
                        print(f"   ❌ Function {func_name} not found in index")
                except Exception as e:
                    print(f"   ❌ Error finding similar functions for {func_name}: {e}")
        except Exception as e:
            print(f"   ❌ Similarity analysis error: {e}")
            return False
        
        # 测试重复代码检测
        print("4. Testing duplicate code detection on attrs codebase...")
        try:
            # 使用较高的阈值，因为attrs是经过良好维护的项目
            duplicates = searcher.find_duplicate_code(threshold=0.9)
            print(f"   ✅ Duplicate detection successful: found {len(duplicates)} duplicate groups")
            
            if duplicates:
                print("   Duplicate groups found:")
                for i, group in enumerate(duplicates[:3]):  # 只显示前3组
                    print(f"   Group {i+1}: {len(group)} functions")
                    for func in group:
                        context = func.context or {}
                        func_name = context.get('function_name', 'Unknown')
                        file_name = os.path.basename(context.get('file_path', 'Unknown'))
                        print(f"     - {func_name} ({file_name})")
                    print()
            else:
                print("   No significant duplicates found (good code quality!)")
        except Exception as e:
            print(f"   ❌ Duplicate detection error: {e}")
            return False
        
        # 测试复杂度分析
        print("5. Testing complexity analysis on attrs codebase...")
        try:
            complexity_analysis = searcher.get_code_complexity_analysis()
            print("   ✅ Complexity analysis successful")
            print(f"   Total functions: {complexity_analysis.get('total_functions', 0)}")
            print(f"   Average complexity: {complexity_analysis.get('average_complexity', 0):.2f}")
            print(f"   Max complexity: {complexity_analysis.get('max_complexity', 0)}")
            print(f"   Min complexity: {complexity_analysis.get('min_complexity', 0)}")
            print(f"   Std complexity: {complexity_analysis.get('std_complexity', 0):.2f}")
            
            # 显示复杂度分布
            distribution = complexity_analysis.get('complexity_distribution', {})
            print("   Complexity distribution:")
            print(f"     Low (0-2): {distribution.get('low', 0)} functions")
            print(f"     Medium (2-5): {distribution.get('medium', 0)} functions")
            print(f"     High (5-10): {distribution.get('high', 0)} functions")
            print(f"     Very High (10+): {distribution.get('very_high', 0)} functions")
            
            # 显示函数类型分布
            func_types = complexity_analysis.get('function_type_distribution', {})
            if func_types:
                print("   Function types:")
                for func_type, count in func_types.items():
                    print(f"     {func_type}: {count} functions")
        except Exception as e:
            print(f"   ❌ Complexity analysis error: {e}")
            return False
        
        # 清理
        indexer.close()
        
        print("\n🎉 Attrs index clustering and analysis tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Attrs index test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_advanced_search_features():
    """测试高级搜索功能"""
    print("\n🔬 Testing Advanced Search Features")
    print("=" * 60)
    
    # 创建测试目录和文件
    test_dir = tempfile.mkdtemp()
    test_file = os.path.join(test_dir, "test.py")
    
    test_code = '''
def calculate_fibonacci(n):
    """Calculate the nth Fibonacci number using recursion"""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def fibonacci_iterative(n):
    """Calculate the nth Fibonacci number using iteration"""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

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
        # 导入必要的模块
        print("1. Setting up advanced search...")
        from sync_indexer import create_default_sync_config, SyncSemanticIndexer
        from semantic_search import create_advanced_searcher, SearchQuery
        
        # 创建配置和索引器
        sync_config = create_default_sync_config(test_dir)
        indexer = SyncSemanticIndexer(sync_config)
        indexer.index_repository()
        
        # 创建高级搜索器
        searcher = create_advanced_searcher(indexer)
        print("   ✅ Setup successful")
        
        # 测试基于示例的搜索
        print("2. Testing example-based search...")
        try:
            example_code = "def search(arr, target):\n    for i, item in enumerate(arr):\n        if item == target:\n            return i\n    return -1"
            example_results = searcher.search_by_example(example_code, top_k=3)
            print(f"   ✅ Example-based search successful: found {len(example_results)} results")
            
            for i, result in enumerate(example_results, 1):
                context = result.context or {}
                print(f"     {i}. {context.get('function_name', 'Unknown')} "
                      f"(similarity: {result.similarity:.4f})")
        except Exception as e:
            print(f"   ❌ Example-based search error: {e}")
            return False
        
        # 测试基于函数签名的搜索
        print("3. Testing function signature search...")
        try:
            signature_results = searcher.search_by_function_signature(
                "search", 
                parameters=["arr", "target"], 
                top_k=3
            )
            print(f"   ✅ Function signature search successful: found {len(signature_results)} results")
            
            for i, result in enumerate(signature_results, 1):
                context = result.context or {}
                print(f"     {i}. {context.get('function_name', 'Unknown')} "
                      f"(similarity: {result.similarity:.4f})")
        except Exception as e:
            print(f"   ❌ Function signature search error: {e}")
            return False
        
        # 测试搜索建议
        print("4. Testing search suggestions...")
        try:
            suggestions = searcher.get_search_suggestions("fib", limit=5)
            print(f"   ✅ Search suggestions successful: found {len(suggestions)} suggestions")
            print(f"   Suggestions: {suggestions}")
        except Exception as e:
            print(f"   ❌ Search suggestions error: {e}")
            return False
        
        # 测试搜索分析
        print("5. Testing search analytics...")
        try:
            # 先执行一些搜索来生成历史
            queries = [
                SearchQuery("fibonacci calculation", "semantic", top_k=3),
                SearchQuery("search algorithm", "semantic", top_k=3),
                SearchQuery("binary search", "keyword", top_k=3)
            ]
            
            for query in queries:
                searcher.search(query)
            
            analytics = searcher.get_search_analytics()
            print("   ✅ Search analytics successful")
            print(f"   Total searches: {analytics.get('total_searches', 0)}")
            print(f"   Query types: {analytics.get('query_type_distribution', {})}")
            print(f"   Average result count: {analytics.get('average_result_count', 0):.2f}")
        except Exception as e:
            print(f"   ❌ Search analytics error: {e}")
            return False
        
        # 清理
        indexer.close()
        
        print("\n🎉 Advanced search features tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Advanced search test failed with error: {e}")
        return False
    
    finally:
        # 清理测试目录
        shutil.rmtree(test_dir)


def main():
    """主测试函数"""
    print("🚀 Code Clustering and Similarity Analysis Test Suite")
    print("=" * 80)
    
    all_passed = True
    
    # 运行基本聚类测试
    if not test_code_clustering():
        all_passed = False
    
    # 运行attrs索引聚类测试
    if not test_attrs_index_clustering():
        all_passed = False
    
    # 运行高级搜索功能测试
    if not test_advanced_search_features():
        all_passed = False
    
    # 总结
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 All clustering and analysis tests passed!")
        print("\nAvailable clustering and analysis features:")
        print("1. Function clustering - Group similar functions together")
        print("2. Similarity analysis - Find similar functions to a given function")
        print("3. Duplicate code detection - Find duplicate or near-duplicate code")
        print("4. Complexity analysis - Analyze code complexity distribution")
        print("5. Example-based search - Search using code examples")
        print("6. Function signature search - Search by function signatures")
        print("7. Search suggestions - Get intelligent search suggestions")
        print("8. Search analytics - Analyze search patterns and performance")
        
        print("\nUsage examples:")
        print("# Test clustering on your own codebase:")
        print("python test_clustering_analysis.py")
        print("\n# Use with attrs_index.json:")
        print("python test_clustering_analysis.py --use-attrs-index")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Make sure all required dependencies are installed")
        print("2. Check that attrs_index.json exists in the current directory")
        print("3. Verify the attrs repository path is correct")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test code clustering and similarity analysis")
    parser.add_argument("--use-attrs-index", action="store_true", 
                       help="Only test with attrs_index.json")
    parser.add_argument("--skip-attrs", action="store_true",
                       help="Skip attrs index tests")
    
    args = parser.parse_args()
    
    if args.use_attrs_index:
        # 只运行attrs索引测试
        success = test_attrs_index_clustering()
        sys.exit(0 if success else 1)
    elif args.skip_attrs:
        # 跳过attrs测试
        success = (test_code_clustering() and test_advanced_search_features())
        sys.exit(0 if success else 1)
    else:
        # 运行所有测试
        sys.exit(main())
