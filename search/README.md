# Neo4j搜索模块

基于Neo4j图数据库的代码搜索和分析模块，提供丰富的搜索和分析功能。

## 功能特性

### 1. 基础查询功能
- **API调用者查询**: 查找某个API的所有调用者
- **API被调用者查询**: 查找某个API调用的其他函数
- **依赖清单**: 获取函数的完整依赖关系
- **函数查找**: 根据函数名精确或模糊查找
- **调用图**: 生成函数的调用关系图
- **文件依赖**: 分析文件间的依赖关系

### 2. 语义搜索功能
- **自然语言搜索**: 使用自然语言描述搜索相关函数
- **关键词搜索**: 基于关键词的精确搜索
- **混合搜索**: 结合语义和关键词的智能搜索
- **AI增强搜索**: 使用OpenAI API进行智能排序和解释
- **复杂度搜索**: 按代码复杂度筛选函数
- **相似函数搜索**: 查找功能相似的函数

### 3. 图分析功能
- **中心性分析**: PageRank、介数中心性、接近中心性、特征向量中心性
- **社区发现**: Louvain、Leiden、标签传播算法
- **相似度分析**: 结构相似度、语义相似度、Jaccard相似度
- **图统计**: 节点数、边数、连通性等统计信息

### 4. 综合分析功能
- **函数重要性分析**: 综合中心性、社区、依赖等因素
- **综合搜索**: 结合多种搜索策略的智能搜索
- **结果导出**: 支持JSON和Markdown格式导出

## 安装依赖

```bash
# 基础依赖
pip install neo4j

# 语义搜索依赖（可选）
pip install sentence-transformers

# 图分析依赖（可选）
pip install networkx

# 中文分词依赖（可选）
pip install jieba

# AI增强搜索依赖（可选）
pip install openai
```

## 快速开始

### 1. 基础使用

```python
from search import Neo4jSearchEngine

# 创建搜索引擎
with Neo4jSearchEngine() as engine:
    # 查找API调用者
    callers = engine.find_api_callers("process_data", max_depth=3)
    print(f"找到 {callers['total_callers']} 个调用者")
    
    # 自然语言搜索
    results = engine.search_by_natural_language("处理数据的函数", limit=10)
    for result in results:
        print(f"- {result['name']}: {result.get('similarity_score', 0):.3f}")
    
    # 中心性分析
    centrality = engine.calculate_centrality("pagerank", top_k=20)
    print("最重要的函数:")
    for result in centrality['results'][:5]:
        print(f"  {result['rank']}. {result['node_name']} ({result['centrality_score']:.4f})")
```

### 2. 高级配置

```python
from search import Neo4jSearchEngine

# 自定义配置
engine = Neo4jSearchEngine(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="your_password",
    neo4j_database="your_database",
    embedding_model="all-MiniLM-L6-v2",
    openai_api_key="your_openai_key"
)

try:
    # 综合分析
    analysis = engine.analyze_function_importance(
        "important_function",
        include_centrality=True,
        include_community=True,
        include_dependencies=True
    )
    
    # 导出结果
    engine.export_results(analysis, "analysis_result.json", "json")
    
finally:
    engine.close()
```

## API参考

### Neo4jSearchEngine

主要的搜索引擎类，提供统一的接口。

#### 基础查询方法

- `find_api_callers(api_name, max_depth=3, include_external=True)`: 查找API调用者
- `find_api_callees(api_name, max_depth=3, include_external=True)`: 查找API被调用者
- `get_dependency_list(function_name, include_transitive=True, max_depth=5)`: 获取依赖清单
- `find_function_by_name(function_name, exact_match=True)`: 根据函数名查找
- `get_function_call_graph(function_name, max_depth=3, direction="both")`: 获取调用图
- `get_file_dependencies(file_path)`: 获取文件依赖

#### 语义搜索方法

- `search_by_natural_language(query, limit=10, search_type="hybrid", similarity_threshold=None)`: 自然语言搜索
- `search_by_complexity(complexity_level=None, min_lines=None, max_lines=None, min_complexity=None, max_complexity=None, limit=100)`: 按复杂度搜索
- `search_similar_functions(function_name, limit=10, similarity_threshold=None)`: 查找相似函数

#### 图分析方法

- `calculate_centrality(algorithm="pagerank", top_k=20, include_weights=True)`: 计算中心性
- `find_communities(algorithm="louvain", min_community_size=2, resolution=1.0)`: 社区发现
- `calculate_similarity_matrix(function_names, similarity_type="structural")`: 计算相似度矩阵
- `get_graph_statistics()`: 获取图统计信息

#### 综合分析方法

- `analyze_function_importance(function_name, include_centrality=True, include_community=True, include_dependencies=True)`: 函数重要性分析
- `comprehensive_search(query, search_type="hybrid", include_analysis=False, limit=10)`: 综合搜索
- `export_results(results, output_file, format="json")`: 导出结果

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

可以通过修改 `config.py` 文件来自定义配置：

```python
class SearchConfig:
    # Neo4j连接配置
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "your_password"
    
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

## 使用示例

### 1. 查找API使用情况

```python
# 查找某个API的所有调用者
callers = engine.find_api_callers("process_data")
print(f"API 'process_data' 被 {callers['total_callers']} 个函数调用")

# 查看调用者分布
for file_path, callers_in_file in callers['callers_by_file'].items():
    print(f"文件 {file_path}: {len(callers_in_file)} 个调用者")
```

### 2. 分析函数依赖

```python
# 获取函数的完整依赖关系
deps = engine.get_dependency_list("main_function", include_transitive=True, max_depth=5)
print(f"函数 'main_function' 有 {deps['total_dependencies']} 个依赖")

# 查看复杂度统计
complexity_stats = deps['complexity_statistics']
print(f"平均复杂度: {complexity_stats['average_complexity']:.2f}")
print(f"高复杂度依赖: {len(complexity_stats['high_complexity_deps'])} 个")
```

### 3. 语义搜索

```python
# 使用自然语言搜索
results = engine.search_by_natural_language(
    "处理用户输入并验证数据的函数",
    search_type="hybrid",
    limit=10
)

for result in results:
    print(f"函数: {result['name']}")
    print(f"相似度: {result.get('similarity_score', 0):.3f}")
    print(f"描述: {result.get('docstring_description', 'N/A')}")
    print("---")
```

### 4. 图分析

```python
# 计算PageRank中心性
centrality = engine.calculate_centrality("pagerank", top_k=20)
print("最重要的函数:")
for result in centrality['results'][:10]:
    print(f"  {result['rank']}. {result['node_name']} ({result['centrality_score']:.4f})")

# 社区发现
communities = engine.find_communities("louvain", min_community_size=3)
print(f"发现 {communities['total_communities']} 个社区")
for community in communities['communities'][:5]:
    print(f"社区 {community['community_id']}: {community['size']} 个函数")
```

### 5. 综合分析

```python
# 分析函数重要性
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

# 社区信息
if 'community_analysis' in importance:
    community = importance['community_analysis']
    if community.get('function_community'):
        func_community = community['function_community']
        print(f"所属社区: 社区 {func_community['community_id']} (大小: {func_community['size']})")
```

## 运行示例

```bash
# 运行所有示例
python search/examples.py

# 或者导入使用
python -c "from search.examples import example_basic_queries; example_basic_queries()"
```

## 注意事项

1. **Neo4j数据库**: 确保Neo4j数据库已启动并包含相应的函数和关系数据
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

## 贡献

欢迎提交Issue和Pull Request来改进这个模块。

## 许可证

MIT License
