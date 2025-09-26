# Nebula

Nebula is an open source software API graph construction and ecosystem evaluation system based on graph analysis. It provides API identification, analysis, and graph construction capabilities through AI technology, offering data support for open source software ecosystem evaluation.

This project mainly targets API graph construction and ecosystem evaluation of open source software, and constructs a multi-stage analysis pipeline, including API Identification & Extraction, Agent Signature Generation, API Function Analysis, Similar API Matching, and API Graph Modeling. Through these agents, automated API identification, analysis, and graph construction can be achieved, thereby providing support for open source software ecosystem evaluation.

Currently, it mainly supports API analysis of multiple programming languages including Python, JavaScript, Java, C/C++, Rust, Go, Ruby, and C#, using various AI models for API function analysis and similarity matching, with more language and model support to be added in the future.

## Usage Instructions

### Clone the project repository
```bash
git clone https://github.com/yourusername/nebula.git
cd nebula/github
```

### Create Environment
It is recommended to use conda to create a virtual environment:
```bash
conda create -n nebula python=3.10
conda activate nebula
pip install -r requirements.txt
```

### Configure Database, API Key, and Model Settings
Add your desired configuration in the `config.py` file:

```python
# Neo4j Configuration
NEO4J_CONFIG = {
    "uri": "bolt://localhost:7687",
    "user": "neo4j", 
    "password": "your_password_here",
    "database": "neo4j"
}

# AI Configuration
AI_CONFIG = {
    "model": "gpt-3.5-turbo",
    "api_key": "YOUR_API_KEY_HERE",
    "max_tokens": 1000
}
```

Please replace `YOUR_API_KEY_HERE` with the actual API key and configure your Neo4j database settings.

In addition, it is necessary to address the encoding issue of the config file. Please ensure that the encoding of your config file is utf-8 and that there are no BOM headers in the file.

## Using the Project

### Using from Command Line

#### Run the complete analysis pipeline:
```bash
python batch_processor.py --repos repo1,repo2,repo3
```

#### Run individual stages:
```bash
# API identification and extraction
python extract/main.py --repo-path /path/to/repo

# Agent signature generation  
python description/main.py --input-file extracted_apis.json

# API function analysis
python semantic_search/vector_embedding.py --input-file api_signatures.json

# API graph ingestion to Neo4j
python ingest.py api_graph_data.json
```

### Using the Gradio GUI

Run `demo.py`, which will start the gradio service on local port 8002. Open http://127.0.0.1:8002/ in your browser.


#### Start API Graph Construction
Enter the open source project repository path you want to process in the text box above. Click start to begin API identification and graph construction. After completion, detailed results will be displayed in the various sections below:

#### API Identification and Extraction Results
The extraction results will show:
- API function identification
- API signature extraction
- API parameter analysis
- API dependency relationships

#### API Function Analysis
Click the "API Function Analysis" button to start API function analysis. The progress will be shown in the analysis section. After completion, the results will be displayed:

#### Similar API Matching
Click the "Similar API Matching" button to start similar API matching functionality. After completion, you can view similar API matching results:

#### API Graph Visualization
Click the "Show API Graph" button to display the API relationship graph:

## Project Structure

```
github/
├── extract/                 # API identification and extraction module
│   ├── main.py             # Main entry point
│   ├── api_extractor.py    # API extractor
│   ├── signature_analyzer.py  # Signature analyzer
│   └── ...
├── description/            # Agent signature generation module
│   ├── main.py            # Main entry point
│   ├── ai_client.py       # AI client
│   ├── batch_processor.py # Batch processor
│   └── ...
├── graph_search/          # Similar API matching module
│   ├── main_interface.py  # Main interface
│   ├── api_matcher.py     # API matcher
│   └── ...
├── semantic_search/       # API function analysis core
│   └── vector_embedding.py # Vector embedding
├── ingest.py             # Neo4j API graph import
├── demo.py               # Web demo interface
├── batch_processor.py   # Batch processing tool
└── requirements.txt      # Dependencies
```

## Supported Languages

- Python
- JavaScript/TypeScript  
- Java
- C/C++
- Rust
- Go
- Ruby
- C#

## Configuration

### AI Model Configuration
Configure AI models in `description/config.py`:

```python
AI_CONFIG = {
    "model": "gpt-3.5-turbo",
    "api_key": "your-api-key",
    "max_tokens": 1000
}
```

### Vector Embedding Configuration
Configure embedding models in `semantic_search/vector_embedding.py`:

```python
config = EmbeddingConfig(
    model_name="all-MiniLM-L6-v2",
    model_type="sentence_transformers"
)
```

### Neo4j Configuration
Configure database connection in `config.py`:

```python
NEO4J_CONFIG = {
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "your_password",
    "database": "neo4j"
}
```

## Contact Us

If you have any questions or suggestions, please contact us via the following methods:

Email: 348351928@qq.com

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Tree-sitter](https://tree-sitter.github.io/) - Code parsing
- [Neo4j](https://neo4j.com/) - Graph database
- [Gradio](https://gradio.app/) - Web interface
- [Sentence Transformers](https://www.sbert.net/) - Semantic embeddings