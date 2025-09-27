# Nebula

Nebula 是一个开源软件 API 图谱构建与基于图谱的开源软件生态评估系统，通过 AI 技术提供 API 识别、分析和图谱构建功能，为开源软件生态评估提供数据支持。

本项目主要针对开源软件的 API 图谱构建和生态评估，构建了多阶段分析流水线，包括 API 识别与提取、Agent 签名生成、API 功能分析、相似 API 匹配和 API 图谱建模等核心服务。通过这些智能体，可以实现对开源软件 API 的自动化识别、分析和图谱构建，从而为开源软件生态评估提供支持。

目前主要支持包括 Python、JavaScript、Java、C/C++、Rust、Go、Ruby 和 C# 等多种编程语言的 API 分析，使用各种 AI 模型进行 API 功能分析和相似性匹配，未来将添加更多语言和模型支持。

## 使用说明

### 克隆项目仓库
```bash
git clone https://github.com/yourusername/nebula.git
cd nebula/github
```

### 创建环境
建议使用 conda 创建虚拟环境：
```bash
conda create -n nebula python=3.10
conda activate nebula
pip install -r requirements.txt
```

### 配置数据库、API 密钥和模型设置
在 `config.py` 文件中添加您所需的配置：

```python
# Neo4j 配置
NEO4J_CONFIG = {
    "uri": "bolt://localhost:7687",
    "user": "neo4j", 
    "password": "your_password_here",
    "database": "neo4j"
}

# AI 配置
AI_CONFIG = {
    "model": "gpt-3.5-turbo",
    "api_key": "YOUR_API_KEY_HERE",
    "max_tokens": 1000
}
```

请将 `YOUR_API_KEY_HERE` 替换为实际的 API 密钥，并配置您的 Neo4j 数据库设置。

此外，需要解决配置文件编码问题。请确保您的配置文件编码为 utf-8，并且文件中没有 BOM 头。

## 使用项目

### 从命令行使用

#### 运行完整的分析流水线：
```bash
python batch_processor.py --repos repo1,repo2,repo3
```

#### 运行各个阶段：
```bash
# API 识别与提取
python extract/main.py --repo-path /path/to/repo

# Agent 签名生成  
python description/main.py --input-file extracted_apis.json

# API 功能分析
python semantic_search/vector_embedding.py --input-file api_signatures.json

# API 图谱导入到 Neo4j
python ingest.py api_graph_data.json
```

### 使用 Gradio GUI

运行 `demo.py`，这将在本地端口 8003 上启动 gradio 服务。在浏览器中打开 http://127.0.0.1:8003/。


#### 开始 API 图谱构建
在上面的文本框中输入您要处理的开源项目仓库路径。点击开始进行 API 识别和图谱构建。完成后，详细结果将显示在下面的各个部分中：

#### API 识别与提取结果
提取结果将显示：
- API 函数识别
- API 签名提取
- API 参数分析
- API 依赖关系

#### API 功能分析
点击"API 功能分析"按钮开始 API 功能分析。进度将在分析部分显示。完成后，结果将显示：

#### 相似 API 匹配
点击"相似 API 匹配"按钮开始相似 API 匹配功能。完成后，您可以查看相似 API 的匹配结果：

#### API 图谱可视化
点击"显示 API 图谱"按钮显示 API 关系图谱：

## 项目结构

```
github/
├── extract/                 # API 识别与提取模块
│   ├── main.py             # 主入口点
│   ├── api_extractor.py    # API 提取器
│   ├── signature_analyzer.py  # 签名分析器
│   └── ...
├── description/            # Agent 签名生成模块
│   ├── main.py            # 主入口点
│   ├── ai_client.py       # AI 客户端
│   ├── batch_processor.py # 批处理器
│   └── ...
├── graph_search/          # 相似 API 匹配模块
│   ├── main_interface.py  # 主界面
│   ├── api_matcher.py     # API 匹配器
│   └── ...
├── semantic_search/       # API 功能分析核心
│   └── vector_embedding.py # 向量嵌入
├── ingest.py             # Neo4j API 图谱导入
├── demo.py               # Web 演示界面
├── batch_processor.py   # 批处理工具
└── requirements.txt      # 依赖项
```

## 支持的语言

- Python
- JavaScript/TypeScript  
- Java
- C/C++
- Rust
- Go
- Ruby
- C#

## 配置

### AI 模型配置
在 `description/config.py` 中配置 AI 模型：

```python
AI_CONFIG = {
    "model": "gpt-3.5-turbo",
    "api_key": "your-api-key",
    "max_tokens": 1000
}
```

### 向量嵌入配置
在 `semantic_search/vector_embedding.py` 中配置嵌入模型：

```python
config = EmbeddingConfig(
    model_name="all-MiniLM-L6-v2",
    model_type="sentence_transformers"
)
```

### Neo4j 配置
在 `config.py` 中配置数据库连接：

```python
NEO4J_CONFIG = {
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "your_password",
    "database": "neo4j"
}
```

## 联系我们

如果您有任何问题或建议，请通过以下方式联系我们：

邮箱：348351928@qq.com

## 许可证

本项目采用 MIT 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件。

## 致谢

- [Tree-sitter](https://tree-sitter.github.io/) - 代码解析
- [Neo4j](https://neo4j.com/) - 图数据库
- [Gradio](https://gradio.app/) - Web 界面
- [Sentence Transformers](https://www.sbert.net/) - 语义嵌入
