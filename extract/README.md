# Extract - 代码分析工具包

这是一个功能强大的代码分析工具，主要用于从GitHub仓库中提取和分析API信息。该工具已从原始的 `extract5.py` 拆分为多个模块，便于维护和扩展。

## 功能特性

1. **代码复杂度分析**：语义、语法、结构复杂度分析
2. **API信息提取**：函数、类、导入、调用关系提取
3. **调用关系图构建**：构建和分析API调用关系图
4. **传递调用分析**：分析间接调用关系和影响范围
5. **函数重要度计算**：基于多维度指标计算函数重要度
6. **依赖关系分析**：分析文件、类、模块级依赖关系
7. **循环依赖检测**：检测各种级别的循环依赖

## 模块结构

```
extract/
├── __init__.py              # 包初始化文件
├── config.py                # 配置和常量定义
├── utils.py                 # 基础工具函数
├── complexity_analyzer.py   # 复杂度分析模块
├── info_extractor.py        # 信息提取模块
├── call_graph_analyzer.py   # 调用图分析模块
├── importance_calculator.py # 重要度计算模块
├── main.py                  # 主入口文件
└── README.md               # 说明文档
```

### 模块说明

- **config.py**: 包含语言映射、节点类型配置、阈值设置等常量
- **utils.py**: 提供仓库克隆、语言检测、节点处理等基础工具函数
- **complexity_analyzer.py**: 实现语义、语法、结构复杂度分析功能
- **info_extractor.py**: 负责函数信息、类信息、导入、调用关系提取
- **call_graph_analyzer.py**: 构建调用关系图、分析依赖关系、传递调用
- **importance_calculator.py**: 实现函数重要度计算算法
- **main.py**: 整合所有模块，提供完整的分析流程

## 安装依赖

```bash
pip install tree-sitter tree-sitter-languages lizard
```

## 使用方法

### 基本用法

```bash
# 完整克隆仓库，生成JSON文件
python -m extract.main https://github.com/user/repo

# 完整克隆 + 过滤大函数（代码行数>200）
python -m extract.main https://github.com/user/repo --filter-large-functions

# 稀疏检出指定目录
python -m extract.main https://github.com/user/repo src

# 稀疏检出多个目录
python -m extract.main https://github.com/user/repo "src/,lib/,core/"

# 稀疏检出 + 过滤大函数
python -m extract.main https://github.com/user/repo src --filter-large-functions

# 直接写入Neo4j数据库（不生成JSON文件）
python -m extract.main https://github.com/user/repo --neo4j

# 使用自定义Neo4j配置
python -m extract.main https://github.com/user/repo --neo4j \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-user myuser \
    --neo4j-password mypass \
    --neo4j-database mydb

# 清空Neo4j数据库后写入
python -m extract.main https://github.com/user/repo --neo4j --clear-db
```

### 参数说明

- `<GitHub仓库链接>`: 要分析的GitHub仓库URL
- `--filter-large-functions`: 可选参数，只保留代码行数>200的函数
- `--neo4j`: 直接写入Neo4j数据库，不生成JSON文件
- `--neo4j-uri <URI>`: Neo4j数据库URI (默认: bolt://localhost:7687)
- `--neo4j-user <用户名>`: Neo4j用户名 (默认: neo4j)
- `--neo4j-password <密码>`: Neo4j密码 (默认: 90879449Drq)
- `--neo4j-database <数据库>`: Neo4j数据库名 (默认: neo4j)
- `--clear-db`: 清空Neo4j数据库后写入
- `<稀疏检出路径>`: 指定要检出的子目录路径，多个路径用逗号分隔

## Neo4j图数据库集成

### 概述

extract模块支持直接将分析结果写入Neo4j图数据库，无需生成中间JSON文件。这样可以：

### 修复说明

**v1.1 更新**: 修复了Neo4j写入器中的问题，现在可以正确创建：
- ✅ 函数节点及其完整属性
- ✅ 类节点及其属性  
- ✅ 文件节点及其属性
- ✅ 函数调用关系
- ✅ 传递调用关系
- ✅ 依赖关系

**新增功能**:
- 详细的调试日志输出
- 数据验证和错误处理
- 进度跟踪（每100个函数显示一次进度）
- 异常处理和错误恢复

### 测试Neo4j写入器

使用测试脚本验证Neo4j写入器功能：

```bash
# 运行测试脚本
python extract/test_neo4j_writer.py
```

测试脚本会：
1. 创建包含2个函数和1个类的测试数据
2. 清空Neo4j数据库
3. 写入测试数据
4. 验证所有节点和关系是否正确创建

- **提高效率**：避免大型JSON文件的I/O操作
- **节省存储空间**：数据直接存储在数据库中
- **实时分析**：数据立即可用于图查询和分析
- **更好的可视化**：利用Neo4j Browser进行图形化展示

### 安装Neo4j

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
    --env NEO4J_AUTH=neo4j/password \
    neo4j:latest
```

#### 安装Python依赖
```bash
pip install neo4j>=5.0.0
```

### Neo4j数据模型

#### 节点类型
- **Repository**: 仓库信息
- **File**: 文件信息
- **Class**: 类信息
- **Function**: 函数信息（包含所有分析结果）

#### 关系类型
- **BELONGS_TO**: 归属关系
- **CALLS**: 函数调用关系
- **TRANSITIVE_CALLS**: 传递调用关系
- **DEPENDS_ON**: 依赖关系

### 常用查询示例

```cypher
// 查看最重要的函数
MATCH (f:Function)
WHERE f.importance_level = 'Critical'
RETURN f.name, f.total_score, f.complexity_score
ORDER BY f.total_score DESC

// 分析函数调用关系
MATCH (caller:Function)-[r:CALLS]->(callee:Function)
RETURN caller.name, callee.name, r.type
LIMIT 20

// 查找枢纽函数
MATCH (f:Function)
WHERE f.is_hub = true
RETURN f.name, f.total_score, f.importance_level
ORDER BY f.total_score DESC
```

### 详细使用指南

更多Neo4j使用信息请参考：[NEO4J_USAGE.md](NEO4J_USAGE.md)

## 输出结果

### JSON文件输出

当不使用`--neo4j`选项时，分析完成后会生成一个JSON文件，包含以下信息：

### JSON输出模板结构

```json
{
  "repository": "仓库名称",
  "analysis_timestamp": "2024-01-01T00:00:00+00:00",
  "total_functions": 100,
  "function_names": ["function1", "function2", ...],
  "functions": [
    {
      "basic_info": {
        "function_name": "函数名",
        "source_code": "完整的函数源代码",
        "code_location": {
          "start_line": 10,
          "end_line": 25,
          "start_column": 0,
          "end_column": 0
        },
        "comments": ["函数注释1", "函数注释2"],
        "parameters": [
          {
            "name": "参数名",
            "type": "参数类型",
            "default": "默认值"
          }
        ],
        "return_type": "返回类型"
      },
      "complexity": {
        "semantic_complexity": {
          "cyclomatic_complexity": 5,
          "lines_of_code": 15,
          "parameters": 3,
          "token_count": 120,
          "complexity_score": 7.5
        },
        "syntactic_complexity": {
          "return_types": 1,
          "external_dependencies": 5,
          "syntax_depth": 3
        },
        "structural_complexity": {
          "branch_count": 4,
          "syntax_depth": 2
        }
      },
      "context": {
        "parent_class": "所属类名",
        "file_path": "文件相对路径",
        "imports": ["导入的模块1", "导入的模块2"],
        "function_calls": {
          "internal_calls": ["内部调用函数1", "内部调用函数2"],
          "external_calls": ["外部调用函数1", "外部调用函数2"]
        },
        "return_value_usage": []
      },
      "importance": {
        "total_score": 85.5,
        "importance_level": "High",
        "breakdown": {
          "direct_call_weight": 20.0,
          "transitive_call_weight": 30.0,
          "depth_impact": 15.0,
          "hub_effect": 10.0,
          "incoming_call_frequency": 5.0,
          "special_weight": 0.0,
          "complexity_adjustment": 1.2
        },
        "metrics": {
          "is_hub": true,
          "is_leaf": false,
          "is_coordinator": false,
          "is_foundation": true
        }
      }
    }
  ],
  "class_function_relationship": {
    "description": "类和函数的关联关系分析",
    "classes_with_methods": 10,
    "standalone_functions": 90
  },
  "api_call_relationships": {
    "call_graph": {
      "nodes": [
        {
          "id": "function1",
          "type": "function",
          "file_path": "文件路径"
        }
      ],
      "edges": [
        {
          "source": "function1",
          "target": "function2",
          "type": "internal_call"
        }
      ],
      "statistics": {
        "total_nodes": 100,
        "total_edges": 150,
        "isolated_functions": 5,
        "most_called_functions": [
          {
            "function_name": "function1",
            "call_count": 10
          }
        ]
      }
    },
    "dependencies": {
      "file_level": {
        "dependencies": {},
        "circular_dependencies": []
      },
      "class_level": {
        "dependencies": {},
        "circular_dependencies": []
      },
      "module_level": {
        "dependencies": {},
        "circular_dependencies": []
      }
    },
    "metrics": {
      "overall_metrics": {
        "total_functions": 100,
        "total_calls": 150,
        "average_complexity": 5.2,
        "most_complex_functions": [
          {
            "function_name": "complex_function",
            "complexity": 15.5
          }
        ]
      }
    }
  },
  "transitive_calls": {
    "transitive_graph": {
      "nodes": [],
      "edges": [],
      "statistics": {
        "total_nodes": 100,
        "max_transitive_depth": 5,
        "functions_with_transitive_calls": 80
      }
    },
    "impact_analysis": {
      "most_transitive_functions": [
        {
          "function_name": "function1",
          "transitive_call_count": 20
        }
      ],
      "transitive_call_chains": [
        {
          "description": "function1 -> function2 -> function3",
          "length": 3
        }
      ],
      "high_impact_functions": [
        {
          "function_name": "function1",
          "impact_count": 15
        }
      ],
      "isolated_functions": ["function1", "function2"],
      "hub_functions": [
        {
          "function_name": "function1",
          "total_influence": 25.0
        }
      ],
      "leaf_functions": ["function1", "function2"],
      "importance_scores": {
        "scores": {},
        "importance_distribution": {
          "Critical": 5,
          "High": 15,
          "Medium": 30,
          "Low": 40,
          "Minimal": 10
        },
        "top_functions": [
          {
            "function_name": "function1",
            "total_score": 95.5,
            "importance_level": "Critical"
          }
        ]
      }
    }
  },
  "filter_info": {
    "filter_enabled": true,
    "filter_condition": "lines_of_code > 200",
    "total_functions_before_filter": 150,
    "total_functions_after_filter": 100,
    "filtered_out_count": 50
  }
}
```

### 主要字段说明

#### 1. 基础信息 (basic_info)
- `function_name`: 函数名称
- `source_code`: 完整的函数源代码
- `code_location`: 代码位置信息（行号、列号）
- `comments`: 函数注释列表
- `parameters`: 参数信息（名称、类型、默认值）
- `return_type`: 返回类型

#### 2. 复杂度信息 (complexity)
- **语义复杂度**：圈复杂度、代码行数、参数数量、token数量、复杂度分数
- **语法复杂度**：返回类型数量、外部依赖数量、语法深度
- **结构复杂度**：分支数量、语法深度

#### 3. 上下文信息 (context)
- `parent_class`: 所属类名
- `file_path`: 文件相对路径
- `imports`: 导入的模块列表
- `function_calls`: 函数调用关系（内部调用、外部调用）

#### 4. 重要度信息 (importance)
- `total_score`: 总重要度分数
- `importance_level`: 重要度等级（Critical/High/Medium/Low/Minimal）
- `breakdown`: 各维度分数分解
- `metrics`: 函数特征指标

#### 5. API调用关系 (api_call_relationships)
- `call_graph`: 调用关系图（节点、边、统计信息）
- `dependencies`: 依赖关系分析（文件级、类级、模块级）
- `metrics`: API指标统计

#### 6. 传递调用关系 (transitive_calls)
- `transitive_graph`: 传递调用图
- `impact_analysis`: 影响范围分析
- `importance_scores`: 重要度分析结果

## 支持的语言

- Python (.py)
- Java (.java)
- C/C++ (.c, .h)
- JavaScript (.js)
- TypeScript (.ts)

## 重要度计算算法

函数重要度基于以下维度计算：

1. **直接调用权重** (DCW): 函数直接调用其他函数的数量
2. **传递调用权重** (TCW): 通过传递调用影响的函数数量
3. **调用深度影响** (DI): 考虑调用链深度，使用对数增长
4. **枢纽效应** (HE): 既是调用者又是被调用者的函数权重更高
5. **被调用频率** (ICF): 被其他函数调用的次数
6. **复杂度调整** (CA): 基于函数复杂度进行调整

最终重要度 = (DCW + TCW + DI + HE + ICF) * CA

## 扩展开发

### 添加新语言支持

1. 在 `config.py` 中添加文件扩展名映射
2. 在 `NODE_TYPES` 中添加对应的节点类型
3. 在 `info_extractor.py` 中添加语言特定的解析逻辑

### 添加新的分析功能

1. 创建新的分析模块
2. 在 `main.py` 中集成新功能
3. 更新 `__init__.py` 导出新功能

### 自定义重要度算法

修改 `importance_calculator.py` 中的 `calculate_function_importance` 函数，调整权重系数或添加新的计算维度。

## 注意事项

1. 确保安装了所有必要的依赖
2. 大型仓库分析可能需要较长时间
3. 稀疏检出可以显著减少分析时间
4. 大函数过滤可以帮助聚焦于关键函数
5. 分析结果以JSON格式保存，便于后续处理

## 故障排除

### 常见问题

1. **Tree-sitter解析器错误**: 确保安装了对应语言的tree-sitter解析器
2. **内存不足**: 对于大型仓库，建议使用稀疏检出或大函数过滤
3. **网络问题**: 确保能够访问GitHub仓库

### 调试模式

可以在代码中添加更多print语句来跟踪分析过程，或者使用Python的logging模块。

