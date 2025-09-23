# Neo4j图数据库集成使用指南

## 概述

extract模块现在支持直接将分析结果写入Neo4j图数据库，无需生成中间JSON文件。这样可以：

- **提高效率**：避免大型JSON文件的I/O操作
- **节省存储空间**：数据直接存储在数据库中
- **实时分析**：数据立即可用于图查询和分析
- **更好的可视化**：利用Neo4j Browser进行图形化展示

## 安装Neo4j

### 1. 安装Neo4j数据库

#### 使用Docker（推荐）
```bash
# 启动Neo4j容器
docker run \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -d \
    -v $HOME/neo4j/data:/data \
    -v $HOME/neo4j/logs:/logs \
    -v $HOME/neo4j/import:/var/lib/neo4j/import \
    -v $HOME/neo4j/plugins:/plugins \
    --env NEO4J_AUTH=neo4j/90879449Drq \
    neo4j:latest
```

#### 使用Neo4j Desktop
1. 下载并安装 [Neo4j Desktop](https://neo4j.com/download/)
2. 创建新项目
3. 创建新数据库
4. 设置用户名和密码

### 2. 安装Python依赖

```bash
pip install neo4j>=5.0.0
```

## 使用方法

### 基本用法

```bash
# 直接写入Neo4j数据库
python -m extract.main https://github.com/user/repo --neo4j

# 使用自定义Neo4j配置
python -m extract.main https://github.com/user/repo --neo4j \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-user myuser \
    --neo4j-password mypass \
    --neo4j-database mydb
```

### 高级选项

```bash
# 清空数据库后写入
python -m extract.main https://github.com/user/repo --neo4j --clear-db

# 结合其他选项
python -m extract.main https://github.com/user/repo src --neo4j --filter-large-functions

# 写入远程Neo4j数据库
python -m extract.main https://github.com/user/repo --neo4j \
    --neo4j-uri bolt://remote-server:7687 \
    --neo4j-user admin \
    --neo4j-password secret123
```

## 数据模型

### 节点类型

#### 1. Repository（仓库）
```cypher
(:Repository {
    name: "仓库名称",
    analysis_timestamp: "2024-01-01T00:00:00+00:00",
    total_functions: 100,
    created_at: datetime()
})
```

#### 2. File（文件）
```cypher
(:File {
    path: "文件相对路径",
    created_at: datetime()
})
```

#### 3. Class（类）
```cypher
(:Class {
    name: "类名",
    created_at: datetime()
})
```

#### 4. Function（函数）
```cypher
(:Function {
    name: "函数名",
    source_code: "完整源代码",
    start_line: 10,
    end_line: 25,
    start_column: 0,
    end_column: 0,
    comments: ["注释1", "注释2"],
    parameters: [{"name": "param1", "type": "int"}],
    return_type: "void",
    parent_class: "所属类名",
    file_path: "文件路径",
    imports: ["导入模块1", "导入模块2"],
    internal_calls: ["内部调用函数1"],
    external_calls: ["外部调用函数1"],
    cyclomatic_complexity: 5,
    lines_of_code: 15,
    parameters_count: 3,
    token_count: 120,
    complexity_score: 7.5,
    return_types: 1,
    external_dependencies: 5,
    syntax_depth: 3,
    branch_count: 4,
    total_score: 85.5,
    importance_level: "High",
    is_hub: true,
    is_leaf: false,
    is_coordinator: false,
    is_foundation: true,
    created_at: datetime()
})
```

### 关系类型

#### 1. BELONGS_TO（属于）
- `(Function)-[:BELONGS_TO]->(Repository)`
- `(File)-[:BELONGS_TO]->(Repository)`
- `(Class)-[:BELONGS_TO]->(Repository)`

#### 2. CALLS（调用）
- `(Function)-[:CALLS {type: 'internal'}]->(Function)`

#### 3. TRANSITIVE_CALLS（传递调用）
- `(Function)-[:TRANSITIVE_CALLS {depth: 2}]->(Function)`

#### 4. DEPENDS_ON（依赖）
- `(File)-[:DEPENDS_ON {level: 'file'}]->(File)`
- `(Class)-[:DEPENDS_ON {level: 'class'}]->(Class)`

## 常用查询示例

### 1. 查看所有函数
```cypher
MATCH (f:Function)
RETURN f.name, f.importance_level, f.complexity_score
ORDER BY f.complexity_score DESC
LIMIT 10
```

### 2. 查找最重要的函数
```cypher
MATCH (f:Function)
WHERE f.importance_level = 'Critical'
RETURN f.name, f.total_score, f.complexity_score
ORDER BY f.total_score DESC
```

### 3. 分析函数调用关系
```cypher
MATCH (caller:Function)-[r:CALLS]->(callee:Function)
RETURN caller.name, callee.name, r.type
LIMIT 20
```

### 4. 查找枢纽函数
```cypher
MATCH (f:Function)
WHERE f.is_hub = true
RETURN f.name, f.total_score, f.importance_level
ORDER BY f.total_score DESC
```

### 5. 分析传递调用链
```cypher
MATCH path = (start:Function)-[:TRANSITIVE_CALLS*1..5]->(end:Function)
WHERE start.name = 'main'
RETURN path
LIMIT 10
```

### 6. 查找最复杂的函数
```cypher
MATCH (f:Function)
RETURN f.name, f.cyclomatic_complexity, f.lines_of_code, f.complexity_score
ORDER BY f.complexity_score DESC
LIMIT 10
```

### 7. 分析文件依赖关系
```cypher
MATCH (f1:File)-[r:DEPENDS_ON]->(f2:File)
RETURN f1.path, f2.path, r.level
```

### 8. 查找孤立函数
```cypher
MATCH (f:Function)
WHERE f.is_leaf = true AND NOT (f)-[:CALLS]->()
RETURN f.name, f.file_path
```

### 9. 分析类的函数分布
```cypher
MATCH (c:Class)<-[:BELONGS_TO]-(f:Function)
RETURN c.name, count(f) as function_count
ORDER BY function_count DESC
```

### 10. 查找循环依赖
```cypher
MATCH (f1:Function)-[:CALLS*1..10]->(f2:Function)-[:CALLS*1..10]->(f1)
WHERE f1 <> f2
RETURN f1.name, f2.name
```

## 性能优化

### 1. 创建索引
系统会自动创建以下索引：
- `function_name_index`: 函数名索引
- `function_importance_index`: 重要度索引
- `function_complexity_index`: 复杂度索引
- `file_path_index`: 文件路径索引
- `class_name_index`: 类名索引
- `repository_name_index`: 仓库名索引

### 2. 查询优化建议
- 使用 `LIMIT` 限制结果数量
- 在 `WHERE` 子句中使用索引字段
- 避免过深的路径查询（如 `*1..20`）
- 使用 `EXPLAIN` 分析查询计划

## 故障排除

### 1. 连接问题
```bash
# 检查Neo4j是否运行
docker ps | grep neo4j

# 检查端口是否开放
telnet localhost 7687
```

### 2. 认证问题
```bash
# 重置密码
docker exec -it neo4j cypher-shell -u neo4j -p neo4j
# 在cypher-shell中执行：
ALTER USER neo4j SET PASSWORD '90879449Drq';
```

### 3. 内存不足
```bash
# 增加Neo4j内存限制
docker run --env NEO4J_dbms_memory_heap_initial__size=2G --env NEO4J_dbms_memory_heap_max__size=4G ...
```

### 4. 数据清理
```cypher
# 清空所有数据
MATCH (n) DETACH DELETE n;

# 删除特定仓库的数据
MATCH (r:Repository {name: 'repo_name'})-[*]-(n)
DETACH DELETE r, n;
```

## 可视化

### 使用Neo4j Browser
1. 打开 http://localhost:7474
2. 登录到Neo4j Browser
3. 使用上述查询语句进行可视化分析

### 示例可视化查询
```cypher
// 显示函数调用关系图
MATCH (f1:Function)-[r:CALLS]->(f2:Function)
WHERE f1.importance_level IN ['Critical', 'High']
RETURN f1, r, f2
LIMIT 50
```

```cypher
// 显示重要度分布
MATCH (f:Function)
RETURN f.importance_level, count(f) as count
ORDER BY count DESC
```

## 最佳实践

1. **定期备份**：使用 `neo4j-admin dump` 备份数据库
2. **监控性能**：使用Neo4j Browser的查询分析功能
3. **合理分页**：大数据集查询时使用 `SKIP` 和 `LIMIT`
4. **索引优化**：根据查询模式创建合适的索引
5. **数据清理**：定期清理不需要的历史数据

## 扩展功能

### 1. 自定义查询
可以基于Neo4j的Cypher查询语言编写复杂的分析查询：

```cypher
// 分析函数影响范围
MATCH (f:Function {name: 'target_function'})
MATCH path = (f)-[:TRANSITIVE_CALLS*1..5]->(impacted:Function)
RETURN impacted.name, length(path) as impact_depth
ORDER BY impact_depth
```

### 2. 集成其他工具
- **Neo4j Bloom**：商业可视化工具
- **Gephi**：开源图可视化工具
- **Cytoscape**：生物信息学图分析工具

### 3. 自动化分析
可以编写脚本定期运行分析并更新Neo4j数据库：

```python
import subprocess
import time

def schedule_analysis():
    while True:
        # 运行分析
        subprocess.run([
            "python", "-m", "extract.main", 
            "https://github.com/user/repo", 
            "--neo4j", "--clear-db"
        ])
        
        # 等待24小时
        time.sleep(24 * 60 * 60)
```
