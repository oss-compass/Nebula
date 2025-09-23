import sys
import os
import logging

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search import Neo4jSearchEngine

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_database_connection():
    """测试数据库连�?""
    print("测试数据库连�?..")
    try:
        with Neo4jSearchEngine() as engine:
            info = engine.get_database_info()
            print(f"数据库信�? {info}")
            return True
    except Exception as e:
        print(f"数据库连接失�? {e}")
        return False

def test_basic_queries():
    """测试基础查询功能"""
    print("\n测试基础查询功能...")
    try:
        with Neo4jSearchEngine() as engine:
            # 测试函数查找
            functions = engine.find_function_by_name("test", exact_match=False)
            print(f"找到 {len(functions)} 个包�?test'的函�?)
            
            if functions:
                func_name = functions[0]['name']
                print(f"测试函数: {func_name}")
                
                # 测试调用者查�?
                callers = engine.find_api_callers(func_name, max_depth=2)
                print(f"调用者数�? {callers['total_callers']}")
                
                # 测试被调用者查�?
                callees = engine.find_api_callees(func_name, max_depth=2)
                print(f"被调用者数�? {callees['total_callees']}")
                
                # 测试依赖清单
                deps = engine.get_dependency_list(func_name, include_transitive=True, max_depth=3)
                print(f"依赖数量: {deps['total_dependencies']}")
            
            return True
    except Exception as e:
        print(f"基础查询测试失败: {e}")
        return False

def test_semantic_search():
    """测试语义搜索功能"""
    print("\n测试语义搜索功能...")
    try:
        with Neo4jSearchEngine() as engine:
            # 测试关键词搜�?
            results = engine.search_by_natural_language("test", limit=5, search_type="keyword")
            print(f"关键词搜索结�? {len(results)} �?)
            
            # 测试复杂度搜�?
            complex_funcs = engine.search_by_complexity(complexity_level="simple", limit=5)
            print(f"简单复杂度函数: {len(complex_funcs)} �?)
            
            return True
    except Exception as e:
        print(f"语义搜索测试失败: {e}")
        return False

def test_graph_analysis():
    """测试图分析功�?""
    print("\n测试图分析功�?..")
    try:
        with Neo4jSearchEngine() as engine:
            # 测试中心性分�?
            centrality = engine.calculate_centrality("pagerank", top_k=10)
            print(f"中心性分析结�? {len(centrality['results'])} 个节�?)
            
            # 测试社区发现
            communities = engine.find_communities("louvain", min_community_size=2)
            print(f"社区发现结果: {communities['total_communities']} 个社�?)
            
            # 测试图统�?
            stats = engine.get_graph_statistics()
            basic_stats = stats.get('basic_statistics', {})
            print(f"图统�? {basic_stats.get('total_nodes', 0)} 个节�? {basic_stats.get('total_edges', 0)} 条边")
            
            return True
    except Exception as e:
        print(f"图分析测试失�? {e}")
        return False

def test_comprehensive_analysis():
    """测试综合分析功能"""
    print("\n测试综合分析功能...")
    try:
        with Neo4jSearchEngine() as engine:
            # 获取一个函数进行测�?
            functions = engine.find_function_by_name("", exact_match=False)
            if functions:
                func_name = functions[0]['name']
                print(f"测试函数: {func_name}")
                
                # 测试重要性分�?
                importance = engine.analyze_function_importance(
                    func_name,
                    include_centrality=True,
                    include_community=True,
                    include_dependencies=True
                )
                
                if "error" not in importance:
                    print(f"重要性分析成功，组件: {importance.get('analysis_components', [])}")
                else:
                    print(f"重要性分析失�? {importance['error']}")
                
                # 测试综合搜索
                search_result = engine.comprehensive_search(
                    func_name,
                    search_type="keyword",
                    include_analysis=False,
                    limit=5
                )
                print(f"综合搜索结果: {search_result['total_semantic_results']} �?)
            
            return True
    except Exception as e:
        print(f"综合分析测试失败: {e}")
        return False

def main():
    """主测试函�?""
    print("Neo4j搜索功能测试")
    print("=" * 50)
    
    tests = [
        ("数据库连�?, test_database_connection),
        ("基础查询", test_basic_queries),
        ("语义搜索", test_semantic_search),
        ("图分�?, test_graph_analysis),
        ("综合分析", test_comprehensive_analysis),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"{test_name}: {'�?通过' if success else '�?失败'}")
        except Exception as e:
            print(f"{test_name}: �?异常 - {e}")
            results.append((test_name, False))
    
    # 输出测试总结
    print("\n" + "=" * 50)
    print("测试总结:")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "�?通过" if success else "�?失败"
        print(f"  {test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试通过�?)
    else:
        print("⚠️  部分测试失败，请检查配置和依赖")

if __name__ == "__main__":
    main()
