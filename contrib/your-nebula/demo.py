#!/usr/bin/env python3

import os
import sys
import json
import time
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import gradio as gr
import pandas as pd
import numpy as np
from datetime import datetime

# Add project path
sys.path.append('.')
sys.path.append('..')  # Add parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # Add project root directory

# Set logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OSSCompassAppNew:
    
    def __init__(self):
        self.batch_processor = None
        
        # Configuration
        self.neo4j_config = {
            'uri': 'bolt://localhost:7687',
            'user': 'neo4j',
            'password': '90879449Drq',
            'database': None
        }
        
        # Processing status
        self.processing_status = {}
        self.results_cache = {}
        
    def initialize_components(self):
        try:
            # Try to import BatchProcessor
            try:
                from batch_processor import BatchProcessor
                self.BatchProcessor = BatchProcessor
                logger.info("BatchProcessor imported successfully")
            except ImportError as e:
                logger.warning(f"BatchProcessor import failed: {e}")
                self.BatchProcessor = None
            
            logger.info("Component initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            traceback.print_exc()
            return False
    
    def process_projects(self, project_urls: str, clean_db: bool = False) -> str:
        try:
            if not project_urls.strip():
                return "请输入项目链接"
            
            if self.BatchProcessor is None:
                return "BatchProcessor模块未正确导入，无法处理项目"
            
            # Parse project links
            urls = [url.strip() for url in project_urls.split('\n') if url.strip()]
            if not urls:
                return "未找到有效的项目链接"
            
            # Create temporary file
            temp_file = f"temp_repos_{int(time.time())}.txt"
            with open(temp_file, 'w', encoding='utf-8') as f:
                for url in urls:
                    f.write(url + '\n')
            
            # Initialize batch processor
            self.batch_processor = self.BatchProcessor(
                txt_file=temp_file,
                neo4j_uri=self.neo4j_config['uri'],
                neo4j_user=self.neo4j_config['user'],
                neo4j_password=self.neo4j_config['password'],
                neo4j_database=self.neo4j_config['database'],
                clean_db=clean_db,
                use_cache=True
            )
            
            # Start processing
            result = []
            repo_urls = self.batch_processor.read_repo_urls()
            
            for i, repo_url in enumerate(repo_urls):
                try:
                    result.append(f"正在处理 {i+1}/{len(repo_urls)}: {repo_url}")
                    
                    # Process single repository
                    self.batch_processor.process_single_repo(repo_url)
                    result.append(f"✓ 完成: {repo_url}")
                    
                except Exception as e:
                    result.append(f"✗ 失败: {repo_url} - {str(e)}")
            
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            result.append(f"\n🎉 批量处理完成！已处理 {len(repo_urls)} 个项目")
            return "\n".join(result)
            
        except Exception as e:
            logger.error(f"Project processing failed: {e}")
            traceback.print_exc()
            return f"处理失败: {str(e)}"
    
    def get_project_status(self) -> str:
        try:
            status_info = []
            status_info.append("项目处理状态报告")
            status_info.append("=" * 50)
            
            # Check output directories
            output_dirs = ['output/extract_output', 'output/description_output', 'output/vector_embedding_output', 'output/ingest_output']
            for dir_name in output_dirs:
                if os.path.exists(dir_name):
                    files = list(Path(dir_name).glob('*'))
                    status_info.append(f"{dir_name}: {len(files)} 个文件")
                else:
                    status_info.append(f"{dir_name}: 目录不存在")
            
            # Check Neo4j connection
            try:
                from neo4j import GraphDatabase
                driver = GraphDatabase.driver(
                    self.neo4j_config['uri'],
                    auth=(self.neo4j_config['user'], self.neo4j_config['password'])
                )
                with driver.session() as session:
                    result = session.run("MATCH (n) RETURN count(n) as count")
                    count = result.single()["count"]
                    status_info.append(f"🔗 Neo4j数据库: 连接成功，包含 {count} 个节点")
                driver.close()
            except Exception as e:
                status_info.append(f"🔗 Neo4j数据库: 连接失败 - {str(e)}")
            
            # Check recent processing logs
            log_files = list(Path('.').glob('*.log'))
            if log_files:
                latest_log = max(log_files, key=os.path.getctime)
                status_info.append(f"📝 最新日志: {latest_log.name}")
            
            return "\n".join(status_info)
            
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return f"获取状态失败: {str(e)}"
    
    def semantic_search(self, query: str, limit: int = 10) -> str:
        try:
            if not query.strip():
                return "请输入搜索查询"
            
            # Try to import semantic search module with multiple fallback paths
            searcher = None
            try:
                # Try importing from semantic_search directory
                try:
                    from semantic_search.semantic_search import SemanticSearcher
                    from semantic_search.sync_indexer import create_default_sync_config, SyncSemanticIndexer
                    
                    # Create indexer configuration
                    sync_config = create_default_sync_config(".")
                    sync_config.index_path = "output/vector_embedding_output"
                    
                    # Try to load existing index
                    try:
                        # Check if index files exist
                        import os
                        index_files = []
                        embeddings_dir = os.path.join(sync_config.index_path, "embeddings")
                        if os.path.exists(embeddings_dir):
                            for file in os.listdir(embeddings_dir):
                                if file.endswith("_embeddings.json"):
                                    index_files.append(os.path.join(embeddings_dir, file))
                        
                        if not index_files:
                            # No existing index, create new one
                            logger.warning("No existing embeddings found, creating new index...")
                            # Create a simple index for testing
                            pass
                    except Exception as e:
                        logger.warning(f"Could not check for existing index: {e}")
                    
                    # Create indexer
                    indexer = SyncSemanticIndexer(sync_config)
                    
                    # Try to load existing embeddings
                    try:
                        import os
                        embeddings_dir = os.path.join(sync_config.index_path, "embeddings")
                        if os.path.exists(embeddings_dir):
                            for file in os.listdir(embeddings_dir):
                                if file.endswith("_embeddings.json"):
                                    # Extract the base filename without _embeddings.json
                                    base_name = file.replace("_embeddings.json", "")
                                    # Try to find the corresponding index file
                                    index_file = os.path.join(sync_config.index_path, f"{base_name}_index.json")
                                    if os.path.exists(index_file):
                                        logger.info(f"Loading index from {index_file}")
                                        indexer.load_index(index_file)
                                        break
                                    else:
                                        # If no index file, try to load embeddings directly
                                        embedding_file = os.path.join(embeddings_dir, file)
                                        logger.info(f"Loading embeddings directly from {embedding_file}")
                                        indexer.embedding_manager.load_index(embedding_file)
                                        break
                    except Exception as e:
                        logger.warning(f"Could not load existing embeddings: {e}")
                    
                    # Create searcher with indexer
                    searcher = SemanticSearcher(indexer)
                except ImportError:
                    # Try alternative import path
                    try:
                        from graph_search.semantic_search import SemanticSearch
                        searcher = SemanticSearch(
                            neo4j_uri=self.neo4j_config['uri'],
                            neo4j_user=self.neo4j_config['user'],
                            neo4j_password=self.neo4j_config['password'],
                            neo4j_database=self.neo4j_config['database']
                        )
                    except ImportError:
                        # Try direct Neo4j query as fallback
                        from neo4j import GraphDatabase
                        driver = GraphDatabase.driver(
                            self.neo4j_config['uri'],
                            auth=(self.neo4j_config['user'], self.neo4j_config['password'])
                        )
                        
                        with driver.session() as session:
                            # Simple function search query
                            query_result = session.run("""
                                MATCH (f:Function)
                                WHERE f.name CONTAINS $query OR f.description CONTAINS $query
                                RETURN f.name as name, f.description as description, f.file_path as file_path
                                LIMIT $limit
                            """, query=query, limit=limit)
                            
                            results = []
                            for record in query_result:
                                results.append({
                                    'name': record.get('name', 'Unknown'),
                                    'description': record.get('description', 'No description'),
                                    'file_path': record.get('file_path', 'Unknown')
                                })
                        
                        driver.close()
                        
                        if not results:
                            return f"未找到与 '{query}' 相关的函数"
                        
                        # Format results
                        formatted_results = []
                        formatted_results.append(f"搜索结果 (查询: '{query}')")
                        formatted_results.append("=" * 60)
                        
                        for i, result in enumerate(results, 1):
                            formatted_results.append(f"\n{i}. {result['name']}")
                            formatted_results.append(f"   文件: {result['file_path']}")
                            formatted_results.append(f"   描述: {result['description'][:100]}...")
                        
                        return "\n".join(formatted_results)
                
                if searcher:
                    # Execute search using the searcher
                    from semantic_search.semantic_search import SearchQuery
                    
                    # Create search query
                    search_query = SearchQuery(
                        query=query,
                        query_type="semantic",
                        top_k=limit,
                        threshold=0.0
                    )
                    
                    results = searcher.search(search_query)
                    
                    if not results:
                        return f"未找到与 '{query}' 相关的函数"
                    
                    # Format results
                    formatted_results = []
                    formatted_results.append(f"语义搜索结果 (查询: '{query}')")
                    formatted_results.append("=" * 60)
                    
                    for i, result in enumerate(results, 1):
                        # Access SearchResult attributes
                        embedding = result.embedding
                        similarity = result.similarity
                        rank = result.rank
                        
                        # Get function information from embedding metadata
                        metadata = embedding.metadata
                        func_name = metadata.get('name', 'Unknown')
                        file_path = metadata.get('file_path', 'Unknown')
                        content = embedding.content
                        
                        formatted_results.append(f"\n{i}. {func_name}")
                        formatted_results.append(f"   文件: {file_path}")
                        formatted_results.append(f"   相似度: {similarity:.4f}")
                        formatted_results.append(f"   内容: {content[:100]}...")
                    
                    return "\n".join(formatted_results)
                
            except Exception as e:
                logger.error(f"Semantic search error: {e}")
                return f"搜索失败: {str(e)}"
                
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return f"搜索失败: {str(e)}"
    
    def generate_visualization(self, viz_type: str = "interactive"):
        try:
            if viz_type == "interactive":
                return self._generate_draggable_plotly_graph()
            elif viz_type == "neo4j":
                return self._generate_draggable_plotly_graph()
            elif viz_type == "statistics":
                return self._generate_statistics_plotly()
            elif viz_type == "graph":
                # Compatible with old "graph" option, map to "interactive"
                return self._generate_draggable_plotly_graph()
            else:
                return self._generate_draggable_plotly_graph()
                
        except Exception as e:
            logger.error(f"Failed to generate visualization: {e}")
            return self._generate_draggable_plotly_graph()
    
    def _generate_draggable_plotly_graph(self):
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from neo4j import GraphDatabase
            import numpy as np
            
            # Connect to Neo4j database
            driver = GraphDatabase.driver(
                self.neo4j_config['uri'],
                auth=(self.neo4j_config['user'], self.neo4j_config['password'])
            )
            
            with driver.session(database=self.neo4j_config['database']) as session:
                # Get node data
                node_query = """
                MATCH (n)
                RETURN n.id as id, n.name as name, labels(n) as labels,
                       n.lines_of_code as size, n.importance_score as importance
                LIMIT 30
                """
                nodes_result = session.run(node_query)
                nodes = []
                for record in nodes_result:
                    node_type = record["labels"][0] if record["labels"] else "Node"
                    size = record["importance"] or record["size"] or 1
                    nodes.append({
                        "id": record["id"] or f"node_{len(nodes)}",
                        "name": record["name"] or "Unknown",
                        "type": node_type,
                        "size": max(5, int(size)),
                        "cluster": 0
                    })
                
                # Get edge data
                edge_query = """
                MATCH (a)-[r]->(b)
                WHERE a.id IS NOT NULL AND b.id IS NOT NULL
                RETURN a.id as source, b.id as target, type(r) as type, 
                       COALESCE(r.weight, 1.0) as weight
                LIMIT 50
                """
                edges_result = session.run(edge_query)
                edges = []
                for record in edges_result:
                    edges.append({
                        "source": record["source"],
                        "target": record["target"],
                        "type": record["type"] or "RELATION",
                        "weight": float(record["weight"]) if record["weight"] is not None else 1.0
                    })
                
                # If no data, use test data
                if not nodes:
                    nodes = [
                        {"id": "node1", "name": "Project A", "type": "Project", "size": 5, "cluster": 0},
                        {"id": "node2", "name": "Project B", "type": "Project", "size": 3, "cluster": 1},
                        {"id": "node3", "name": "Project C", "type": "Project", "size": 4, "cluster": 0},
                        {"id": "node4", "name": "Project D", "type": "Project", "size": 2, "cluster": 2},
                        {"id": "node5", "name": "Project E", "type": "Project", "size": 6, "cluster": 1},
                        {"id": "node6", "name": "Project F", "type": "Project", "size": 3, "cluster": 2}
                    ]
                    edges = [
                        {"source": "node1", "target": "node2", "type": "RELATED", "weight": 0.8},
                        {"source": "node1", "target": "node3", "type": "RELATED", "weight": 0.6},
                        {"source": "node2", "target": "node4", "type": "RELATED", "weight": 0.4},
                        {"source": "node3", "target": "node5", "type": "RELATED", "weight": 0.7},
                        {"source": "node4", "target": "node6", "type": "RELATED", "weight": 0.5},
                        {"source": "node5", "target": "node6", "type": "RELATED", "weight": 0.3}
                    ]
            
            driver.close()
            
            # Create node positions (using force-directed layout)
            node_positions = self._calculate_node_positions(nodes, edges)
            
            # Prepare edge data
            edge_x = []
            edge_y = []
            edge_info = []
            
            for edge in edges:
                source_pos = node_positions.get(edge['source'])
                target_pos = node_positions.get(edge['target'])
                if source_pos and target_pos:
                    edge_x.extend([source_pos[0], target_pos[0], None])
                    edge_y.extend([source_pos[1], target_pos[1], None])
                    edge_info.append(f"{edge['source']} → {edge['target']}<br>Type: {edge['type']}<br>Weight: {edge['weight']:.2f}")
            
            # Create edge trace
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='rgba(100,200,255,0.8)'),  # 亮蓝色，在暗色背景下更清晰
                hoverinfo='none',
                mode='lines',
                name='Connections'
            )
            
            # Prepare node data
            node_x = [pos[0] for pos in node_positions.values()]
            node_y = [pos[1] for pos in node_positions.values()]
            node_text = [node['name'] for node in nodes]
            node_sizes = [max(10, node['size'] * 3) for node in nodes]
            node_colors = [hash(node['type']) % 10 for node in nodes]
            
            # Create node trace
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="middle center",
                hovertext=[f"Name: {node['name']}<br>Type: {node['type']}<br>Size: {node['size']}" for node in nodes],
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    colorscale='Viridis',  # 保持Viridis色彩方案，在暗色背景下效果更好
                    line=dict(width=3, color='rgba(255,255,255,0.9)'),  # 更粗的白色边框，增强对比度
                    opacity=0.9  # 提高不透明度
                ),
                name='Nodes',
                textfont=dict(color='white', size=12, family='Arial Black')  # 白色文字，增强可读性
            )
            
            # Create figure
            fig = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              title=dict(
                                  text='Neo4j 交互式可拖拽图谱',
                                  font=dict(size=20, color='white')  # 白色标题，在暗色背景下更清晰
                              ),
                              showlegend=True,
                              hovermode='closest',
                              margin=dict(b=20,l=5,r=5,t=40),
                              annotations=[ dict(
                                  text="拖拽节点重新布局，悬停查看详情",
                                  showarrow=False,
                                  xref="paper", yref="paper",
                                  x=0.005, y=-0.002,
                                  xanchor='left', yanchor='bottom',
                                  font=dict(color='rgba(200,200,200,0.8)', size=12)  # 浅灰色说明文字
                              )],
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              plot_bgcolor='rgba(0,0,0,0)',
                              paper_bgcolor='rgba(0,0,0,0)',
                              autosize=True,
                              width=None,
                              height=600,
                              # 添加暗色主题的图例样式
                              legend=dict(
                                  bgcolor='rgba(0,0,0,0.8)',
                                  bordercolor='rgba(255,255,255,0.3)',
                                  borderwidth=1,
                                  font=dict(color='white', size=12)
                              )
                          ))
            
            # Add drag functionality
            fig.update_layout(
                dragmode='pan',  # Allow dragging
                xaxis=dict(scaleanchor="y", scaleratio=1),  # Maintain aspect ratio
                yaxis=dict(scaleanchor="x", scaleratio=1)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to generate draggable graph: {str(e)}")
            # Return a simple test chart
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[1, 2, 3, 4],
                y=[1, 2, 3, 4],
                mode='markers+text',
                text=['Node1', 'Node2', 'Node3', 'Node4'],
                marker=dict(size=20, color='blue')
            ))
            fig.update_layout(
                title="测试图表 - 可拖拽节点",
                xaxis=dict(scaleanchor="y", scaleratio=1),
                yaxis=dict(scaleanchor="x", scaleratio=1),
                dragmode='pan'
            )
            return fig
    
    def _calculate_node_positions(self, nodes, edges):
        import numpy as np
        import random
        
        # Initialize random positions
        positions = {}
        for i, node in enumerate(nodes):
            angle = 2 * np.pi * i / len(nodes)
            radius = 50 + random.uniform(-20, 20)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions[node['id']] = (x, y)
        
        # Simple force-directed iteration
        for _ in range(50):
            forces = {node_id: [0, 0] for node_id in positions.keys()}
            
            # Calculate repulsive forces (between nodes)
            for i, node1 in enumerate(nodes):
                for j, node2 in enumerate(nodes):
                    if i != j:
                        pos1 = np.array(positions[node1['id']])
                        pos2 = np.array(positions[node2['id']])
                        diff = pos1 - pos2
                        dist = np.linalg.norm(diff)
                        if dist > 0:
                            force = 100 / (dist ** 2)  # Repulsive force
                            forces[node1['id']] += force * diff / dist
            
            # Calculate attractive forces (edge connections)
            for edge in edges:
                if edge['source'] in positions and edge['target'] in positions:
                    pos1 = np.array(positions[edge['source']])
                    pos2 = np.array(positions[edge['target']])
                    diff = pos2 - pos1
                    dist = np.linalg.norm(diff)
                    if dist > 0:
                        force = dist * 0.1  # Attractive force
                        forces[edge['source']] += force * diff / dist
                        forces[edge['target']] -= force * diff / dist
            
            # Update positions
            for node_id in positions:
                pos = np.array(positions[node_id])
                force = np.array(forces[node_id])
                pos += force * 0.1  # Step size
                positions[node_id] = tuple(pos)
        
        return positions
    
    def _generate_statistics_plotly(self):
        try:
            import plotly.graph_objects as go
            from neo4j import GraphDatabase
            
            # Connect to Neo4j database
            driver = GraphDatabase.driver(
                self.neo4j_config['uri'],
                auth=(self.neo4j_config['user'], self.neo4j_config['password'])
            )
            
            with driver.session(database=self.neo4j_config['database']) as session:
                # Get function count by file
                file_query = """
                MATCH (f:Function)
                RETURN f.file_path as file_path, count(f) as function_count
                ORDER BY function_count DESC
                LIMIT 20
                """
                file_result = session.run(file_query)
                files = []
                counts = []
                for record in file_result:
                    files.append(record["file_path"] or "Unknown")
                    counts.append(record["function_count"])
            
            driver.close()
            
            if not files:
                # Return empty chart if no data
                fig = go.Figure()
                fig.add_annotation(
                    text="暂无数据",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return fig
            
            # Create bar chart with dark theme colors
            fig = go.Figure(data=[
                go.Bar(
                    x=files, 
                    y=counts, 
                    name='函数数量',
                    marker=dict(
                        color='rgba(100,200,255,0.8)',  # 亮蓝色，在暗色背景下更清晰
                        line=dict(color='rgba(255,255,255,0.8)', width=1)  # 白色边框
                    )
                )
            ])
            
            fig.update_layout(
                title=dict(
                    text='各文件函数数量统计',
                    font=dict(color='white', size=16)  # 白色标题
                ),
                xaxis=dict(
                    title='文件路径',
                    titlefont=dict(color='white'),
                    tickfont=dict(color='white'),
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                yaxis=dict(
                    title='函数数量',
                    titlefont=dict(color='white'),
                    tickfont=dict(color='white'),
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                autosize=True,
                width=None,
                height=600,
                legend=dict(
                    bgcolor='rgba(0,0,0,0.8)',
                    bordercolor='rgba(255,255,255,0.3)',
                    borderwidth=1,
                    font=dict(color='white', size=12)
                )
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to generate statistics: {e}")
            # Return empty chart
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_annotation(
                text="统计图表生成失败",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def build_interface(self):
        with gr.Blocks(
            title="OSSCompass 开源项目分析工具 - 新布局", 
            theme=gr.themes.Soft(),
            css="""
            .main-container {
                max-width: 1400px;
                margin: 0 auto;
            }
            .section-header {
                margin-bottom: 1rem;
                padding: 0.5rem;
                border-radius: 8px;
                border-left: 4px solid #007bff;
            }
            .status-section {
                height: 50%;
                display: flex;
                flex-direction: column;
            }
            .result-section {
                height: 50%;
                display: flex;
                flex-direction: column;
            }
            .right-column {
                height: 100%;
                display: flex;
                flex-direction: column;
            }
            .left-column {
                height: 100%;
                display: flex;
                flex-direction: column;
            }
            .fixed-height {
                min-height: 200px !important;
                height: 200px !important;
            }
            .status-container {
                height: 100%;
                display: flex;
                flex-direction: column;
            }
            .status-textbox {
                flex: 1;
                min-height: 0;
            }
            .status-textbox textarea {
                overflow-y: auto !important;
                resize: none !important;
            }
            .result-container {
                height: 100%;
                display: flex;
                flex-direction: column;
            }
            .result-textbox {
                flex: 1;
                min-height: 0;
            }
            .result-textbox textarea {
                overflow-y: auto !important;
                resize: none !important;
            }
            """
        ) as demo:
            gr.Markdown("# API图谱与生态评估分析工具")
            gr.Markdown("输入开源项目链接进行语义搜索、相似性比较、API推荐、可视化和聚类分析")
            
            # Top section: Batch processing
            gr.Markdown("## 批量处理")
            
            with gr.Row():
                with gr.Column(scale=4):
                    project_input = gr.Textbox(
                        label="项目链接",
                        placeholder="输入GitHub项目链接，每行一个\n示例:\nhttps://github.com/user/repo1\nhttps://github.com/user/repo2",
                        lines=10
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            clean_db_checkbox = gr.Checkbox(
                                label="清空数据库",
                                value=False,
                                # info="Clear Neo4j database before processing"
                            )
                        with gr.Column(scale=2):
                            process_btn = gr.Button("开始处理", variant="primary")
                
                
            # Bottom section: Processing results and project status
            with gr.Row():
                # Left: Processing results
                with gr.Column(scale=1, elem_classes="result-container"):
                    gr.Markdown("## 处理结果")
                    
                    process_output = gr.Textbox(
                        label="处理结果",
                        lines=11,
                        max_lines=25,
                        interactive=False,
                        show_copy_button=True,
                        elem_classes="result-textbox"
                    )
                
                # Right: Project status
                with gr.Column(scale=1, elem_classes="status-container"):
                    gr.Markdown("## 项目状态")
                    
                    status_output = gr.Textbox(
                        label="系统状态",
                        lines=8,
                        interactive=False,
                        show_copy_button=True,
                        elem_classes="status-textbox"
                    )
                    
                    with gr.Row():
                        status_btn = gr.Button("刷新状态", variant="primary")
            
            # Bind events
            process_btn.click(
                fn=self.process_projects,
                inputs=[project_input, clean_db_checkbox],
                outputs=process_output
            )
            
            status_btn.click(
                fn=self.get_project_status,
                outputs=status_output
            )
            
            # Separator
            gr.Markdown("---")
            
            # Bottom section: Tab-based functionality
            with gr.Tabs():
                # Semantic search tab
                with gr.Tab("语义搜索"):
                    gr.Markdown("## 智能语义搜索")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            search_query = gr.Textbox(
                                label="搜索查询",
                                placeholder="输入函数名、功能描述或关键词",
                                lines=2
                            )
                            search_limit = gr.Slider(
                                minimum=1,
                                maximum=50,
                                value=10,
                                step=1,
                                label="结果数量"
                            )
                            search_btn = gr.Button("搜索", variant="primary")
                        
                       
                    
                    search_output = gr.Textbox(
                        label="搜索结果",
                        lines=15,
                        interactive=False
                    )
                    
                    search_btn.click(
                        fn=self.semantic_search,
                        inputs=[search_query, search_limit],
                        outputs=search_output
                    )
                
                # Visualization tab
                with gr.Tab("可视化"):
                    gr.Markdown("## 交互式节点图谱可视化")
                    
                    # Control panel
                    with gr.Row():
                        with gr.Column(scale=1):
                            viz_type = gr.Radio(
                                choices=["interactive", "neo4j", "statistics", "graph"],
                                value="interactive",
                                label="可视化类型",
                                info="交互式图谱支持拖拽和选择功能"
                            )
                            viz_btn = gr.Button("生成可视化", variant="primary")
                        
                       
                    # Chart area - independent full row
                    viz_output = gr.Plot(
                        label="交互式可视化",
                        value=None,
                        container=True,
                        show_label=True,
                        scale=1,
                        elem_classes=["full-width-plot"],
                        # Add more layout parameters
                        elem_id="main-plot"
                    )
                    
                    viz_btn.click(
                        fn=self.generate_visualization,
                        inputs=[viz_type],
                        outputs=viz_output
                    )
                
            
            # Auto-refresh status on page load
            demo.load(
                fn=self.get_project_status,
                outputs=status_output
            )
            
            # Add custom CSS to ensure charts take full width
            demo.css = """
            .full-width-plot {
                width: 100% !important;
                max-width: 100% !important;
                min-width: 100% !important;
                display: block !important;
            }
            
            .full-width-plot .plotly {
                width: 100% !important;
                max-width: 100% !important;
                min-width: 100% !important;
            }
            
            .full-width-plot .js-plotly-plot {
                width: 100% !important;
                max-width: 100% !important;
                min-width: 100% !important;
            }
            
            #main-plot {
                width: 100% !important;
                max-width: 100% !important;
                min-width: 100% !important;
                display: block !important;
            }
            
            #main-plot .plotly {
                width: 100% !important;
                max-width: 100% !important;
                min-width: 100% !important;
            }
            
            #main-plot .js-plotly-plot {
                width: 100% !important;
                max-width: 100% !important;
                min-width: 100% !important;
            }
            
            /* 确保Plot组件容器占满宽度 */
            .gradio-plot {
                width: 100% !important;
                max-width: 100% !important;
                min-width: 100% !important;
            }
            
            .gradio-plot .plotly {
                width: 100% !important;
                max-width: 100% !important;
                min-width: 100% !important;
            }
            
            /* 响应式设计 */
            @media (max-width: 768px) {
                .full-width-plot, #main-plot {
                    width: 100% !important;
                    max-width: 100% !important;
                }
            }
            """
        
        return demo
    
    def launch(self, server_port=8003, share=False):
        if not self.initialize_components():
            print("Component initialization failed, but can continue launching application")
            print("Some features may not work properly")
        
        demo = self.build_interface()
        demo.launch(
            server_port=server_port,
            share=share,
            show_error=True,
            quiet=False
        )

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='OSSCompass Demo Application')
    parser.add_argument('--port', type=int, default=8003, help='Server port (default: 8003)')
    parser.add_argument('--share', action='store_true', help='Share the application publicly')
    args = parser.parse_args()
    
    app = OSSCompassAppNew()
    app.launch(server_port=args.port, share=args.share)

if __name__ == "__main__":
    main()
