# Neo4j搜索模块使用指南

## 快速开始

### 1. 基本使用

```python
from search import Neo4jSearchEngine

# 创建搜索引擎
with Neo4jSearchEngine() as engine:
    # 查找API调用者
    callers = engine.find_api_callers("process_data")
    print(f"找到 {callers['total_callers']} 个调用者")
    
    # 自然语言搜索
    results = engine.search_by_natural_language("处理数据的函数")
    for result in results:
        print(f"- {result['name']}: {result.get('similarity_score', 0):.3f}")
```

### 2. 命令行使用

```bash
# 搜索函数
python search/cli.py search "处理数据的函数" --search-type hybrid --limit 10

# 查找调用者
python search/cli.py callers process_data --depth 3

# 中心性分析
python search/cli.py centrality --algorithm pagerank --top-k 20

# 社区发现
python search/cli.py communities --algorithm louvain --min-size 3

# 函数重要性分析
python search/cli.py analyze important_function --include-centrality --include-community
```

## 功能详解

### 基础查询功能

#### 1. API调用者查询
```python
# 查找某个API的所有调用者
callers = engine.find_api_callers("process_data", max_depth=3, include_external=True)
print(f"直接调用者: {len(callers['direct_callers'])}")
print(f"间接调用者: {len(callers['indirect_callers'])}")
```

#### 2. API被调用者查询
```python
# 查找某个API调用的其他函数
callees = engine.find_api_callees("process_data", max_depth=3)
print(f"直接被调用者: {len(callees['direct_callees'])}")
print(f"间接被调用者: {len(callees['indirect_callees'])}")
```

#### 3. 依赖清单
```python
# 获取函数的完整依赖关系
deps = engine.get_dependency_list("main_function", include_transitive=True, max_depth=5)
print(f"总依赖数: {deps['total_dependencies']}")
print(f"直接依赖: {len(deps['direct_dependencies'])}")
print(f"传递依赖: {len(deps['transitive_dependencies'])}")
```

### 语义搜索功能

#### 1. 自然语言搜索
```python
# 使用自然语言描述搜索
results = engine.search_by_natural_language(
    "处理用户输入并验证数据的函数",
    search_type="hybrid",  # semantic, keyword, hybrid, ai
    limit=10
)
```

#### 2. 按复杂度搜索
```python
# 查找复杂函数
complex_funcs = engine.search_by_complexity(
    complexity_level="complex",
    min_complexity=10,
    limit=20
)
```

#### 3. 相似函数搜索
```python
# 查找功能相似的函数
similar_funcs = engine.search_similar_functions("process_data", limit=10)
```

### 图分析功能

#### 1. 中心性分析
```python
# PageRank中心性
centrality = engine.calculate_centrality("pagerank", top_k=20)
print("最重要的函数:")
for result in centrality['results'][:10]:
    print(f"  {result['rank']}. {result['node_name']} ({result['centrality_score']:.4f})")

# 介数中心性
betweenness = engine.calculate_centrality("betweenness", top_k=20)
```

#### 2. 社区发现
```python
# Louvain社区发现
communities = engine.find_communities("louvain", min_community_size=3)
print(f"发现 {communities['total_communities']} 个社区")
for community in communities['communities'][:5]:
    print(f"社区 {community['community_id']}: {community['size']} 个函数")
```

#### 3. 相似度矩阵
```python
# 计算函数间的相似度矩阵
function_names = ["process_data", "validate_input", "format_output"]
similarity_matrix = engine.calculate_similarity_matrix(function_names, "structural")
```

### 综合分析功能

#### 1. 函数重要性分析
```python
# 综合分析函数重要性
importance = engine.analyze_function_importance(
    "critical_function",
    include_centrality=True,
    include_community=True,
    include_dependencies=True
)

print(f"函数: {importance['function_name']}")
print(f"分析组件: {', '.join(importance['analysis_components'])}")

# 中心性排名
if 'centrality_analysis' in importance:
    centrality = importance['centrality_analysis']
    print(f"中心性排名: {centrality['function_rank']}/{centrality['total_functions']}")
```

#### 2. 综合搜索
```python
# 结合多种搜索策略
search_result = engine.comprehensive_search(
    "数据处理相关函数",
    search_type="hybrid",
    include_analysis=True,
    limit=10
)
```

## 配置说明

### 环境变量
```bash
# Neo4j连接配置
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"
export NEO4J_DATABASE="your_database"

# OpenAI配置（可选）
export OPENAI_API_KEY="your_openai_key"
```

### 配置文件
修改 `config.py` 文件来自定义配置：
```python
class SearchConfig:
    # 搜索配置
    default_limit = 10
    max_depth = 5
    similarity_threshold = 0.3
    
    # 嵌入模型配置
    embedding_model = "all-MiniLM-L6-v2"
    embedding_device = "cpu"
    
    # 图分析配置
    centrality_algorithms = ["pagerank", "betweenness", "closeness", "eigenvector"]
    community_algorithms = ["louvain", "leiden", "label_propagation"]
```

## 运行示例

### 1. 运行测试
```bash
python search/test_search.py
```

### 2. 运行示例
```bash
python search/examples.py
```

### 3. 命令行工具
```bash
# 显示帮助
python search/cli.py --help

# 搜索函数
python search/cli.py search "处理数据" --limit 5

# 查找调用者
python search/cli.py callers process_data --depth 2

# 中心性分析
python search/cli.py centrality --algorithm pagerank --top-k 10

# 社区发现
python search/cli.py communities --algorithm louvain --min-size 2

# 函数分析
python search/cli.py analyze important_function --include-centrality --include-dependencies

# 数据库信息
python search/cli.py info
```

## 结果导出

### 1. 导出为JSON
```python
results = engine.search_by_natural_language("数据处理")
engine.export_results(results, "results.json", "json")
```

### 2. 导出为Markdown
```python
results = engine.search_by_natural_language("数据处理")
engine.export_results(results, "results.md", "markdown")
```

### 3. 命令行导出
```bash
python search/cli.py search "数据处理" --output results.json --format json
python search/cli.py centrality --output centrality.md --format markdown
```

## 注意事项

1. **数据库连接**: 确保Neo4j数据库已启动并包含相应的函数和关系数据
2. **依赖安装**: 某些功能需要额外的依赖包，请根据需要安装
3. **性能考虑**: 大规模图分析可能需要较长时间，建议设置合理的限制参数
4. **内存使用**: 语义搜索和图分析可能消耗较多内存，请确保系统资源充足

## 故障排除

### 常见问题

1. **连接失败**: 检查Neo4j连接配置和数据库状态
2. **嵌入模型加载失败**: 确保网络连接正常，或使用本地模型
3. **图分析算法不可用**: 安装相应的依赖包（如networkx）
4. **内存不足**: 减少搜索范围或使用更小的图

### 日志配置
```python
import logging
logging.basicConfig(level=logging.INFO)
```
