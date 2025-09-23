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

# 添加项目路径
sys.path.append('.')
sys.path.append('..')  # 添加父目�?sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 添加项目根目�?
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OSSCompassAppNew:
    """开源项目分析应�?- 新布局版本"""
    
    def __init__(self):
        self.batch_processor = None
        
        # 配置
        self.neo4j_config = {
            'uri': 'bolt://localhost:7687',
            'user': 'neo4j',
            'password': '90879449Drq',
            'database': None
        }
        
        # 处理状�?        self.processing_status = {}
        self.results_cache = {}
        
    def initialize_components(self):
        """初始化各个组�?""
        try:
            # 尝试导入BatchProcessor
            try:
                from batch_processor import BatchProcessor
                self.BatchProcessor = BatchProcessor
                logger.info("BatchProcessor导入成功")
            except ImportError as e:
                logger.warning(f"BatchProcessor导入失败: {e}")
                self.BatchProcessor = None
            
            logger.info("组件初始化完�?)
            return True
            
        except Exception as e:
            logger.error(f"组件初始化失�? {e}")
            traceback.print_exc()
            return False
    
    def process_projects(self, project_urls: str, clean_db: bool = False) -> str:
        """处理项目链接"""
        try:
            if not project_urls.strip():
                return "请输入项目链�?
            
            if self.BatchProcessor is None:
                return "BatchProcessor模块未正确导入，无法处理项目"
            
            # 解析项目链接
            urls = [url.strip() for url in project_urls.split('\n') if url.strip()]
            if not urls:
                return "没有有效的项目链�?
            
            # 创建临时文件
            temp_file = f"temp_repos_{int(time.time())}.txt"
            with open(temp_file, 'w', encoding='utf-8') as f:
                for url in urls:
                    f.write(url + '\n')
            
            # 初始化批量处理器
            self.batch_processor = self.BatchProcessor(
                txt_file=temp_file,
                neo4j_uri=self.neo4j_config['uri'],
                neo4j_user=self.neo4j_config['user'],
                neo4j_password=self.neo4j_config['password'],
                neo4j_database=self.neo4j_config['database'],
                clean_db=clean_db,
                use_cache=True
            )
            
            # 开始处�?            result = []
            repo_urls = self.batch_processor.read_repo_urls()
            
            for i, repo_url in enumerate(repo_urls):
                try:
                    result.append(f"正在处理 {i+1}/{len(repo_urls)}: {repo_url}")
                    yield "\n".join(result)
                    
                    # 处理单个仓库
                    self.batch_processor.process_single_repo(repo_url)
                    result.append(f"�?完成: {repo_url}")
                    
                except Exception as e:
                    result.append(f"�?失败: {repo_url} - {str(e)}")
                
                yield "\n".join(result)
            
            # 清理临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            result.append(f"\n🎉 批量处理完成！共处理 {len(repo_urls)} 个项�?)
            return "\n".join(result)
            
        except Exception as e:
            logger.error(f"处理项目失败: {e}")
            traceback.print_exc()
            return f"处理失败: {str(e)}"
    
    def get_project_status(self) -> str:
        """获取项目处理状�?""
        try:
            status_info = []
            status_info.append("📊 项目处理状态报�?)
            status_info.append("=" * 50)
            
            # 检查输出目�?            output_dirs = ['output/extract_output', 'output/description_output', 'output/vector_embedding_output', 'output/ingest_output']
            for dir_name in output_dirs:
                if os.path.exists(dir_name):
                    files = list(Path(dir_name).glob('*'))
                    status_info.append(f"📁 {dir_name}: {len(files)} 个文�?)
                else:
                    status_info.append(f"📁 {dir_name}: 目录不存�?)
            
            # 检查Neo4j连接
            try:
                from neo4j import GraphDatabase
                driver = GraphDatabase.driver(
                    self.neo4j_config['uri'],
                    auth=(self.neo4j_config['user'], self.neo4j_config['password'])
                )
                with driver.session() as session:
                    result = session.run("MATCH (n) RETURN count(n) as count")
                    count = result.single()["count"]
                    status_info.append(f"🔗 Neo4j数据�? 连接正常，包�?{count} 个节�?)
                driver.close()
            except Exception as e:
                status_info.append(f"🔗 Neo4j数据�? 连接失败 - {str(e)}")
            
            # 检查最近的处理日志
            log_files = list(Path('.').glob('*.log'))
            if log_files:
                latest_log = max(log_files, key=os.path.getctime)
                status_info.append(f"📝 最新日�? {latest_log.name}")
            
            return "\n".join(status_info)
            
        except Exception as e:
            logger.error(f"获取状态失�? {e}")
            return f"获取状态失�? {str(e)}"
    
    def semantic_search(self, query: str, limit: int = 10) -> str:
        """语义搜索"""
        try:
            if not query.strip():
                return "请输入搜索查�?
            
            # 尝试导入语义搜索模块
            try:
                # 尝试多种导入路径
                try:
                    from semantic_search.single_repo import SingleRepoSearch, create_single_repo_search
                except ImportError:
                    # 如果上面的导入失败，尝试从github目录导入
                    from github.semantic_search.single_repo import SingleRepoSearch, create_single_repo_search
                
                # 创建搜索�?                searcher = create_single_repo_search(
                    neo4j_uri=self.neo4j_config['uri'],
                    neo4j_user=self.neo4j_config['user'],
                    neo4j_password=self.neo4j_config['password'],
                    neo4j_database=self.neo4j_config['database']
                )
                
                # 执行搜索
                results = searcher.search_by_natural_language(
                    query=query,
                    limit=limit,
                    search_type="semantic",
                    similarity_threshold=0.1  # 降低阈值以找到更多结果
                )
                
                if not results:
                    return f"未找到与 '{query}' 相关的函�?
                
                # 格式化结�?                formatted_results = []
                formatted_results.append(f"🔍 语义搜索结果 (查询: '{query}')")
                formatted_results.append("=" * 60)
                
                for i, result in enumerate(results, 1):
                    func_name = result.get('name', 'Unknown')
                    file_path = result.get('file_path', 'Unknown')
                    content = result.get('content', 'No content')
                    similarity = result.get('similarity_score', 0.0)
                    explanation = result.get('explanation', 'No explanation')
                    
                    formatted_results.append(f"\n{i}. {func_name}")
                    formatted_results.append(f"   文件: {file_path}")
                    formatted_results.append(f"   相似�? {similarity:.4f}")
                    formatted_results.append(f"   内容: {content[:100]}...")
                
                return "\n".join(formatted_results)
                
            except ImportError:
                return "语义搜索模块未正确导入，请检查依�?
            except Exception as e:
                return f"搜索失败: {str(e)}"
                
        except Exception as e:
            logger.error(f"语义搜索失败: {e}")
            return f"搜索失败: {str(e)}"
    
    def generate_visualization(self, viz_type: str = "interactive"):
        """生成可视化图�?""
        try:
            if viz_type == "interactive":
                return self._generate_draggable_plotly_graph()
            elif viz_type == "neo4j":
                return self._generate_draggable_plotly_graph()
            elif viz_type == "statistics":
                return self._generate_statistics_plotly()
            elif viz_type == "graph":
                # 兼容旧的"graph"选项，映射到"interactive"
                return self._generate_draggable_plotly_graph()
            else:
                return self._generate_draggable_plotly_graph()
                
        except Exception as e:
            logger.error(f"生成可视化失�? {e}")
            return self._generate_draggable_plotly_graph()
    
    def _generate_draggable_plotly_graph(self):
        """生成可拖拽的Plotly图表"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from neo4j import GraphDatabase
            import numpy as np
            
            # 连接Neo4j数据�?            driver = GraphDatabase.driver(
                self.neo4j_config['uri'],
                auth=(self.neo4j_config['user'], self.neo4j_config['password'])
            )
            
            with driver.session(database=self.neo4j_config['database']) as session:
                # 获取节点数据
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
                
                # 获取边数�?                edge_query = """
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
                
                # 如果没有数据，使用测试数�?                if not nodes:
                    nodes = [
                        {"id": "node1", "name": "项目A", "type": "Project", "size": 5, "cluster": 0},
                        {"id": "node2", "name": "项目B", "type": "Project", "size": 3, "cluster": 1},
                        {"id": "node3", "name": "项目C", "type": "Project", "size": 4, "cluster": 0},
                        {"id": "node4", "name": "项目D", "type": "Project", "size": 2, "cluster": 2},
                        {"id": "node5", "name": "项目E", "type": "Project", "size": 6, "cluster": 1},
                        {"id": "node6", "name": "项目F", "type": "Project", "size": 3, "cluster": 2}
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
            
            # 创建节点位置（使用力导向布局�?            node_positions = self._calculate_node_positions(nodes, edges)
            
            # 准备边数�?            edge_x = []
            edge_y = []
            edge_info = []
            
            for edge in edges:
                source_pos = node_positions.get(edge['source'])
                target_pos = node_positions.get(edge['target'])
                if source_pos and target_pos:
                    edge_x.extend([source_pos[0], target_pos[0], None])
                    edge_y.extend([source_pos[1], target_pos[1], None])
                    edge_info.append(f"{edge['source']} �?{edge['target']}<br>类型: {edge['type']}<br>权重: {edge['weight']:.2f}")
            
            # 创建边轨�?            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='rgba(125,125,125,0.5)'),
                hoverinfo='none',
                mode='lines',
                name='连接'
            )
            
            # 准备节点数据
            node_x = [pos[0] for pos in node_positions.values()]
            node_y = [pos[1] for pos in node_positions.values()]
            node_text = [node['name'] for node in nodes]
            node_sizes = [max(10, node['size'] * 3) for node in nodes]
            node_colors = [hash(node['type']) % 10 for node in nodes]
            
            # 创建节点轨迹
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="middle center",
                hovertext=[f"名称: {node['name']}<br>类型: {node['type']}<br>大小: {node['size']}" for node in nodes],
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    colorscale='Viridis',
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                name='节点'
            )
            
            # 创建图形
            fig = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              title=dict(
                                  text='Neo4j交互式拖拽图',
                                  font=dict(size=20, color='#2c3e50')
                              ),
                              showlegend=True,
                              hovermode='closest',
                              margin=dict(b=20,l=5,r=5,t=40),
                              annotations=[ dict(
                                  text="拖拽节点重新布局，悬停查看详�?,
                                  showarrow=False,
                                  xref="paper", yref="paper",
                                  x=0.005, y=-0.002,
                                  xanchor='left', yanchor='bottom',
                                  font=dict(color='#666', size=12)
                              )],
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              plot_bgcolor='rgba(0,0,0,0)',
                              paper_bgcolor='rgba(0,0,0,0)',
                              width=800,
                              height=600
                          ))
            
            # 添加拖拽功能
            fig.update_layout(
                dragmode='pan',  # 允许拖拽
                xaxis=dict(scaleanchor="y", scaleratio=1),  # 保持比例
                yaxis=dict(scaleanchor="x", scaleratio=1)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"生成可拖拽图失败: {str(e)}")
            # 返回一个简单的测试?            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[1, 2, 3, 4],
                y=[1, 2, 3, 4],
                mode='markers+text',
                text=['节点1', '节点2', '节点3', '节点4'],
                marker=dict(size=20, color='blue')
            ))
            fig.update_layout(
                title="测试�?- 可拖拽节�?,
                xaxis=dict(scaleanchor="y", scaleratio=1),
                yaxis=dict(scaleanchor="x", scaleratio=1),
                dragmode='pan'
            )
            return fig
    
    def _calculate_node_positions(self, nodes, edges):
        """计算节点位置（简单的力导向布局�?""
        import numpy as np
        import random
        
        # 初始化随机位�?        positions = {}
        for i, node in enumerate(nodes):
            angle = 2 * np.pi * i / len(nodes)
            radius = 50 + random.uniform(-20, 20)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions[node['id']] = (x, y)
        
        # 简单的力导向迭�?        for _ in range(50):
            forces = {node_id: [0, 0] for node_id in positions.keys()}
            
            # 计算斥力（节点间�?            for i, node1 in enumerate(nodes):
                for j, node2 in enumerate(nodes):
                    if i != j:
                        pos1 = np.array(positions[node1['id']])
                        pos2 = np.array(positions[node2['id']])
                        diff = pos1 - pos2
                        dist = np.linalg.norm(diff)
                        if dist > 0:
                            force = 100 / (dist ** 2)  # 斥力
                            forces[node1['id']] += force * diff / dist
            
            # 计算引力（边连接�?            for edge in edges:
                if edge['source'] in positions and edge['target'] in positions:
                    pos1 = np.array(positions[edge['source']])
                    pos2 = np.array(positions[edge['target']])
                    diff = pos2 - pos1
                    dist = np.linalg.norm(diff)
                    if dist > 0:
                        force = dist * 0.1  # 引力
                        forces[edge['source']] += force * diff / dist
                        forces[edge['target']] -= force * diff / dist
            
            # 更新位置
            for node_id in positions:
                pos = np.array(positions[node_id])
                force = np.array(forces[node_id])
                pos += force * 0.1  # 步长
                positions[node_id] = tuple(pos)
        
        return positions
    
    def _generate_interactive_graph(self) -> str:
        """生成交互式节点图HTML内容"""
        try:
            # 直接从Neo4j获取数据并生成HTML
            from neo4j import GraphDatabase
            import json
            
            # 连接Neo4j数据�?            driver = GraphDatabase.driver(
                self.neo4j_config['uri'],
                auth=(self.neo4j_config['user'], self.neo4j_config['password'])
            )
            
            with driver.session(database=self.neo4j_config['database']) as session:
                # 获取节点数据 - 使用正确的属性名
                node_query = """
                MATCH (n)
                RETURN n.id as id, n.name as name, labels(n) as labels,
                       n.lines_of_code as size, n.importance_score as importance
                LIMIT 50
                """
                nodes_result = session.run(node_query)
                nodes = []
                for record in nodes_result:
                    # 获取节点类型（从labels中取第一个）
                    node_type = record["labels"][0] if record["labels"] else "Node"
                    # 使用importance_score作为size，如果没有则使用lines_of_code
                    size = record["importance"] or record["size"] or 1
                    nodes.append({
                        "id": record["id"] or f"node_{len(nodes)}",
                        "name": record["name"] or "Unknown",
                        "type": node_type,
                        "size": max(5, int(size)),  # 确保size至少�?，使节点更可�?                        "cluster": 0  # 默认聚类
                    })
                
                logger.info(f"从Neo4j获取�?{len(nodes)} 个节�?)
                
                # 获取边数�?- 过滤掉source或target为null的边
                edge_query = """
                MATCH (a)-[r]->(b)
                WHERE a.id IS NOT NULL AND b.id IS NOT NULL
                RETURN a.id as source, b.id as target, type(r) as type, 
                       COALESCE(r.weight, 1.0) as weight
                LIMIT 100
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
                
                logger.info(f"从Neo4j获取�?{len(edges)} 条边")
                
                # 如果没有数据，添加测试数�?                if not nodes:
                    logger.info("Neo4j中没有数据，使用测试数据")
                    nodes = [
                        {"id": "node1", "name": "项目A", "type": "Project", "size": 5, "cluster": 0},
                        {"id": "node2", "name": "项目B", "type": "Project", "size": 3, "cluster": 1},
                        {"id": "node3", "name": "项目C", "type": "Project", "size": 4, "cluster": 0},
                        {"id": "node4", "name": "项目D", "type": "Project", "size": 2, "cluster": 2},
                        {"id": "node5", "name": "项目E", "type": "Project", "size": 6, "cluster": 1},
                        {"id": "node6", "name": "项目F", "type": "Project", "size": 3, "cluster": 2}
                    ]
                    edges = [
                        {"source": "node1", "target": "node2", "type": "RELATED", "weight": 0.8},
                        {"source": "node1", "target": "node3", "type": "RELATED", "weight": 0.6},
                        {"source": "node2", "target": "node4", "type": "RELATED", "weight": 0.4},
                        {"source": "node3", "target": "node5", "type": "RELATED", "weight": 0.7},
                        {"source": "node4", "target": "node6", "type": "RELATED", "weight": 0.5},
                        {"source": "node5", "target": "node6", "type": "RELATED", "weight": 0.3}
                    ]
                
                # 确保至少有测试数�?                if len(nodes) == 0:
                    logger.info("强制使用测试数据")
                    nodes = [
                        {"id": "node1", "name": "项目A", "type": "Project", "size": 5, "cluster": 0},
                        {"id": "node2", "name": "项目B", "type": "Project", "size": 3, "cluster": 1},
                        {"id": "node3", "name": "项目C", "type": "Project", "size": 4, "cluster": 0},
                        {"id": "node4", "name": "项目D", "type": "Project", "size": 2, "cluster": 2},
                        {"id": "node5", "name": "项目E", "type": "Project", "size": 6, "cluster": 1},
                        {"id": "node6", "name": "项目F", "type": "Project", "size": 3, "cluster": 2}
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
            
            # 生成HTML内容
            html_content = self._create_interactive_html(nodes, edges)
            return html_content
                
        except Exception as e:
            logger.error(f"生成交互式图失败: {str(e)}")
            return f"�?生成交互式图失败: {str(e)}"
    
    def _create_interactive_html(self, nodes, edges):
        """创建支持拖拽的交互式HTML内容"""
        import json
        
        # 将数据转换为JSON格式
        nodes_json = json.dumps(nodes, ensure_ascii=False)
        edges_json = json.dumps(edges, ensure_ascii=False)
        
        # 完整的HTML模板，包含Cytoscape.js和拖拽功�?        html_template = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neo4j交互式拖拽图</title>
    <script src="https://unpkg.com/cytoscape@3.26.0/dist/cytoscape.min.js"></script>
    <script src="https://unpkg.com/cytoscape-cose-bilkent@5.0.0/cytoscape-cose-bilkent.js"></script>
    <script src="https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
    <script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        
        .controls {{
            background: #f8f9fa;
            padding: 20px;
            border-bottom: 1px solid #dee2e6;
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            align-items: center;
        }}
        
        .control-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .control-group label {{
            font-weight: 600;
            color: #495057;
            min-width: 80px;
        }}
        
        .control-group select, .control-group input {{
            padding: 8px 12px;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            background: white;
            transition: all 0.3s ease;
        }}
        
        .control-group select:focus, .control-group input:focus {{
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }}
        
        .btn {{
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .btn-primary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        
        .btn-primary:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }}
        
        .btn-secondary {{
            background: #6c757d;
            color: white;
        }}
        
        .btn-secondary:hover {{
            background: #5a6268;
            transform: translateY(-2px);
        }}
        
        .graph-container {{
            position: relative;
            height: 600px;
            background: #f8f9fa;
        }}
        
        #cy {{
            width: 100%;
            height: 100%;
            border: none;
        }}
        
        .info-panel {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            min-width: 250px;
            max-width: 350px;
            z-index: 1000;
        }}
        
        .info-panel h3 {{
            margin: 0 0 10px 0;
            color: #495057;
            font-size: 1.2em;
        }}
        
        .info-panel p {{
            margin: 0;
            color: #6c757d;
            line-height: 1.5;
        }}
        
        .legend {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            z-index: 1000;
        }}
        
        .legend h4 {{
            margin: 0 0 10px 0;
            color: #495057;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 5px;
        }}
        
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
        }}
        
        .stats {{
            background: #f8f9fa;
            padding: 20px;
            display: flex;
            justify-content: center;
            gap: 40px;
        }}
        
        .stat-item {{
            text-align: center;
        }}
        
        .stat-number {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }}
        
        .stat-label {{
            color: #6c757d;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .node-tooltip {{
            position: absolute;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1001;
        }}
        
        .selected-node {{
            border: 3px solid #ff6b6b !important;
            box-shadow: 0 0 20px rgba(255, 107, 107, 0.5) !important;
        }}
        
        .highlighted-edge {{
            line-color: #ff6b6b !important;
            target-arrow-color: #ff6b6b !important;
            width: 4px !important;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Neo4j交互式拖拽图</h1>
            <p>支持拖拽、缩放、选择等交互操�?/p>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label>布局算法:</label>
                <select id="layout-select">
                    <option value="cose">力导向布局 (COSE)</option>
                    <option value="dagre">层次布局 (Dagre)</option>
                    <option value="circle">圆形布局</option>
                    <option value="grid">网格布局</option>
                    <option value="breadthfirst">广度优先</option>
                </select>
            </div>
            
            <div class="control-group">
                <label>节点大小:</label>
                <input type="range" id="node-size" min="10" max="80" value="30">
                <span id="node-size-value">30</span>
            </div>
            
            <div class="control-group">
                <label>边宽�?</label>
                <input type="range" id="edge-width" min="1" max="8" value="2">
                <span id="edge-width-value">2</span>
            </div>
            
            <div class="control-group">
                <label>节点间距:</label>
                <input type="range" id="node-spacing" min="50" max="300" value="100">
                <span id="node-spacing-value">100</span>
        </div>
        
            <button class="btn btn-primary" onclick="applyLayout()">🔄 应用布局</button>
            <button class="btn btn-primary" onclick="cy.fit()">📐 适应画布</button>
            <button class="btn btn-primary" onclick="cy.center()">🎯 居中显示</button>
            <button class="btn btn-secondary" onclick="clearSelection()">�?清除选择</button>
        </div>
        
        <div class="graph-container">
            <div id="cy"></div>
            
            <div class="info-panel" id="info-panel">
                <h3 id="info-title">节点信息</h3>
                <p id="info-content">点击节点查看详细信息，拖拽节点重新布局</p>
            </div>
            
            <div class="legend">
                <h4>图例说明</h4>
                <div class="legend-item">
                    <div class="legend-color" style="background: #3498db;"></div>
                    <span>项目节点</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #e74c3c;"></div>
                    <span>关系�?/span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ff6b6b;"></div>
                    <span>选中节点</span>
                </div>
            </div>
        </div>
        
        <div class="stats">
            <div class="stat-item">
                <div class="stat-number" id="node-count">{len(nodes)}</div>
                <div class="stat-label">节点数量</div>
            </div>
            <div class="stat-item">
                <div class="stat-number" id="edge-count">{len(edges)}</div>
                <div class="stat-label">关系数量</div>
            </div>
            <div class="stat-item">
                <div class="stat-number" id="selected-count">0</div>
                <div class="stat-label">已选择</div>
            </div>
        </div>
    </div>

    <script>
        // 全局变量
        let cy;
        let selectedNodes = new Set();
        
        // 数据
        const graphData = {{
            nodes: {nodes_json},
            edges: {edges_json}
        }};
        
        // 初始化Cytoscape
        function initCytoscape() {{
            cy = cytoscape({{
                container: document.getElementById('cy'),
                
                elements: [
                    // 节点
                    ...graphData.nodes.map(node => ({{
            data: {{
                id: node.id,
                            name: node.name,
                type: node.type,
                            size: node.size || 30,
                            cluster: node.cluster || 0
                        }}
                    }})),
                    // �?                    ...graphData.edges.map(edge => ({{
            data: {{
                id: edge.source + '-' + edge.target,
                source: edge.source,
                target: edge.target,
                            type: edge.type || 'RELATION',
                            weight: edge.weight || 1
                        }}
                    }}))
                ],
                
            style: [
                {{
                    selector: 'node',
                    style: {{
                            'background-color': '#3498db',
                            'label': 'data(name)',
                        'text-valign': 'center',
                        'text-halign': 'center',
                            'color': 'white',
                        'font-size': '12px',
                        'font-weight': 'bold',
                        'text-outline-width': 2,
                            'text-outline-color': '#2c3e50',
                            'width': 'data(size)',
                            'height': 'data(size)',
                        'border-width': 2,
                            'border-color': '#2c3e50',
                        'cursor': 'pointer'
                    }}
                }},
                {{
                        selector: 'node:selected',
                    style: {{
                            'background-color': '#ff6b6b',
                            'border-color': '#e74c3c',
                            'border-width': 4,
                            'box-shadow': '0 0 20px rgba(255, 107, 107, 0.5)'
                        }}
                    }},
                    {{
                        selector: 'edge',
                        style: {{
                            'width': 2,
                        'line-color': '#95a5a6',
                        'target-arrow-color': '#95a5a6',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                            'opacity': 0.8
                    }}
                }},
                {{
                    selector: 'edge:selected',
                    style: {{
                            'line-color': '#ff6b6b',
                            'target-arrow-color': '#ff6b6b',
                        'width': 4
                    }}
                }}
            ],
                
            layout: {{
                    name: 'cose',
                    animate: true,
                    animationDuration: 1000,
                    nodeSpacing: 100,
                    idealEdgeLength: 100,
                    edgeElasticity: 0.45,
                    nestingFactor: 0.1,
                    gravity: 0.25,
                    numIter: 1000,
                    tile: true,
                    animate: true,
                    animationDuration: 1000,
                    randomize: false,
                fit: true,
                    padding: 30
            }}
        }});
        
            // 添加交互事件
            setupInteractions();
            
            // 应用初始布局
            applyLayout();
        }}
        
        // 设置交互功能
        function setupInteractions() {{
            // 节点点击事件
            cy.on('tap', 'node', function(evt) {{
                const node = evt.target;
                const nodeData = node.data();
                
                // 切换选择状�?                if (node.selected()) {{
                    node.unselect();
                    selectedNodes.delete(node.id());
                }} else {{
                    node.select();
                    selectedNodes.add(node.id());
                }}
                
                // 更新信息面板
                updateInfoPanel(nodeData);
                updateStats();
            }});
            
            // 边点击事�?            cy.on('tap', 'edge', function(evt) {{
                const edge = evt.target;
                const edgeData = edge.data();
                
                edge.select();
                updateInfoPanel(edgeData, 'edge');
            }});
            
            // 背景点击事件
        cy.on('tap', function(evt) {{
            if (evt.target === cy) {{
                    clearSelection();
            }}
        }});
        
            // 节点拖拽事件
            cy.on('drag', 'node', function(evt) {{
                const node = evt.target;
                // 拖拽时高亮相关边
                highlightConnectedEdges(node);
            }});
            
            // 节点拖拽结束事件
            cy.on('dragfree', 'node', function(evt) {{
                const node = evt.target;
                // 恢复边的样式
                cy.edges().style('opacity', 0.8);
            }});
            
            // 鼠标悬停事件
            cy.on('mouseover', 'node', function(evt) {{
                const node = evt.target;
                node.style('background-color', '#e74c3c');
                highlightConnectedEdges(node);
            }});
            
            cy.on('mouseout', 'node', function(evt) {{
                const node = evt.target;
                if (!node.selected()) {{
                    node.style('background-color', '#3498db');
                }}
                cy.edges().style('opacity', 0.8);
            }});
        }}
        
        // 高亮相关�?        function highlightConnectedEdges(node) {{
            const connectedEdges = node.connectedEdges();
            cy.edges().style('opacity', 0.3);
            connectedEdges.style('opacity', 1);
        }}
        
        // 更新信息面板
        function updateInfoPanel(data, type = 'node') {{
            const title = document.getElementById('info-title');
            const content = document.getElementById('info-content');
            
            if (type === 'edge') {{
                title.textContent = '关系信息';
                content.innerHTML = `
                    <strong>关系类型:</strong> ${{data.type}}<br>
                    <strong>权重:</strong> ${{data.weight}}<br>
                    <strong>连接:</strong> ${{data.source}} �?${{data.target}}
                `;
            }} else {{
                title.textContent = '节点信息';
                content.innerHTML = `
                    <strong>名称:</strong> ${{data.name}}<br>
                    <strong>类型:</strong> ${{data.type}}<br>
                    <strong>大小:</strong> ${{data.size}}<br>
                    <strong>聚类:</strong> ${{data.cluster}}
                `;
            }}
        }}
        
        // 更新统计信息
        function updateStats() {{
            document.getElementById('selected-count').textContent = selectedNodes.size;
        }}
        
        // 应用布局
        function applyLayout() {{
            const layoutType = document.getElementById('layout-select').value;
            const nodeSpacing = parseInt(document.getElementById('node-spacing').value);
            
            const layout = cy.layout({{
                name: layoutType,
                animate: true,
                animationDuration: 1000,
                nodeSpacing: nodeSpacing,
                    fit: true,
                    padding: 30
            }});
            
            layout.run();
        }}
        
        // 清除选择
        function clearSelection() {{
            cy.elements().unselect();
            selectedNodes.clear();
            updateStats();
            document.getElementById('info-title').textContent = '节点信息';
            document.getElementById('info-content').textContent = '点击节点查看详细信息，拖拽节点重新布局';
        }}
        
        // 控制面板事件
        document.getElementById('node-size').addEventListener('input', function() {{
            const size = this.value;
            document.getElementById('node-size-value').textContent = size;
            cy.style().selector('node').style('width', size).style('height', size).update();
        }});
        
        document.getElementById('edge-width').addEventListener('input', function() {{
            const width = this.value;
            document.getElementById('edge-width-value').textContent = width;
            cy.style().selector('edge').style('width', width).update();
        }});
        
        document.getElementById('node-spacing').addEventListener('input', function() {{
            const spacing = this.value;
            document.getElementById('node-spacing-value').textContent = spacing;
        }});
        
        // 页面加载完成后初始化
        document.addEventListener('DOMContentLoaded', function() {{
            initCytoscape();
        }});
    </script>
</body>
</html>
        """
        
        return html_template
    
    def _generate_neo4j_graph(self) -> str:
        """生成Neo4j图可视化"""
        try:
            # 尝试连接Neo4j获取图数�?            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                self.neo4j_config['uri'],
                auth=(self.neo4j_config['user'], self.neo4j_config['password'])
            )
            
            with driver.session() as session:
                # 获取节点统计
                result = session.run("MATCH (n) RETURN labels(n) as label, count(n) as count")
                node_stats = []
                for record in result:
                    label = record["label"][0] if record["label"] else "Unknown"
                    count = record["count"]
                    node_stats.append(f"  {label}: {count} 个节�?)
                
                # 获取关系统计
                result = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(r) as count")
                rel_stats = []
                for record in result:
                    rel_type = record["type"]
                    count = record["count"]
                    rel_stats.append(f"  {rel_type}: {count} 个关�?)
            
            driver.close()
            
            viz_info = []
            viz_info.append("📊 Neo4j图数据统�?)
            viz_info.append("=" * 40)
            viz_info.append("\n🔵 节点统计:")
            viz_info.extend(node_stats)
            viz_info.append("\n🔗 关系统计:")
            viz_info.extend(rel_stats)
            viz_info.append("\n💡 提示: 使用'交互式图'选项获得更好的可视化体验")
            
            return "\n".join(viz_info)
            
        except Exception as e:
            return f"获取Neo4j图数据失�? {str(e)}"
    
    def _create_plotly_graph(self, nodes, edges):
        """使用Plotly创建交互式图�?""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            import networkx as nx
            import numpy as np
            
            # 创建NetworkX�?            G = nx.Graph()
            
            # 添加节点
            for node in nodes:
                G.add_node(node['id'], **node)
            
            # 添加�?            for edge in edges:
                if edge['source'] in G.nodes and edge['target'] in G.nodes:
                    G.add_edge(edge['source'], edge['target'], **edge)
            
            # 使用spring布局计算位置
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # 准备节点数据
            node_x = []
            node_y = []
            node_text = []
            node_info = []
            
            for node_id in G.nodes():
                x, y = pos[node_id]
                node_x.append(x)
                node_y.append(y)
                node_data = G.nodes[node_id]
                node_text.append(node_data.get('name', node_id))
                node_info.append(f"ID: {node_id}<br>Type: {node_data.get('type', 'Unknown')}<br>Size: {node_data.get('size', 30)}")
            
            # 准备边数�?            edge_x = []
            edge_y = []
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            # 创建边轨�?            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            # 创建节点轨迹
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="middle center",
                hovertext=node_info,
                marker=dict(
                    showscale=True,
                    colorscale='Viridis',
                    reversescale=True,
                    color=[],
                    size=20,
                    colorbar=dict(
                        thickness=15,
                        title="节点类型",
                        xanchor="left",
                        title_side="right"
                    ),
                    line=dict(width=2, color='white')
                )
            )
            
            # 设置节点颜色
            node_trace.marker.color = [hash(node['type']) % 10 for node in nodes if node['id'] in G.nodes()]
            
            # 创建图形
            fig = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              title=dict(text='Neo4j交互式图', font=dict(size=16)),
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=20,l=5,r=5,t=40),
                              annotations=[ dict(
                                  text="点击节点查看详细信息",
                                  showarrow=False,
                                  xref="paper", yref="paper",
                                  x=0.005, y=-0.002,
                                  xanchor="left", yanchor="bottom",
                                  font=dict(color="gray", size=12)
                              )],
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              plot_bgcolor='white'
                          ))
            
            return fig
            
        except Exception as e:
            logger.error(f"创建Plotly图形失败: {str(e)}")
            return f"�?创建图形失败: {str(e)}"
    
    def _generate_neo4j_plotly_graph(self):
        """生成Neo4j的Plotly交互式图�?""
        try:
            from neo4j import GraphDatabase
            import plotly.graph_objects as go
            import networkx as nx
            
            # 连接Neo4j
            driver = GraphDatabase.driver(
                self.neo4j_config['uri'],
                auth=(self.neo4j_config['user'], self.neo4j_config['password'])
            )
            
            with driver.session() as session:
                # 获取节点数据
                node_result = session.run("MATCH (n) RETURN n LIMIT 50")
                nodes = []
                for record in node_result:
                    node = record["n"]
                    nodes.append({
                        "id": str(node.id),
                        "name": dict(node).get("name", "Unknown"),
                        "type": list(node.labels)[0] if node.labels else "Node",
                        "size": 30
                    })
                
                # 获取关系数据
                edge_result = session.run("MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 100")
                edges = []
                for record in edge_result:
                    a = record["a"]
                    r = record["r"]
                    b = record["b"]
                    edges.append({
                        "source": str(a.id),
                        "target": str(b.id),
                        "type": r.type
                    })
                
                if not nodes:
                    # 如果没有数据，创建一个示例图
                    return self._create_sample_plotly_graph()
                
                # 创建NetworkX�?                G = nx.Graph()
                
                # 添加节点
                for node in nodes:
                    G.add_node(node['id'], **node)
                
                # 添加�?                for edge in edges:
                    if edge['source'] in G.nodes and edge['target'] in G.nodes:
                        G.add_edge(edge['source'], edge['target'], **edge)
                
                # 使用spring布局计算位置
                pos = nx.spring_layout(G, k=1, iterations=50)
                
                # 准备节点数据
                node_x = []
                node_y = []
                node_text = []
                node_info = []
                
                for node_id in G.nodes():
                    x, y = pos[node_id]
                    node_x.append(x)
                    node_y.append(y)
                    node_data = G.nodes[node_id]
                    node_text.append(node_data.get('name', node_id))
                    node_info.append(f"ID: {node_id}<br>Type: {node_data.get('type', 'Unknown')}")
                
                # 准备边数�?                edge_x = []
                edge_y = []
                
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                
                # 创建边轨�?                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=2, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                )
                
                # 创建节点轨迹
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    hoverinfo='text',
                    text=node_text,
                    textposition="middle center",
                    hovertext=node_info,
                    marker=dict(
                        showscale=True,
                        colorscale='Viridis',
                        reversescale=True,
                        color=[],
                        size=20,
                        colorbar=dict(
                            thickness=15,
                            title="节点类型",
                            xanchor="left",
                            title_side="right"
                        ),
                        line=dict(width=2, color='white')
                    )
                )
                
                # 设置节点颜色
                node_trace.marker.color = [hash(node['type']) % 10 for node in nodes if node['id'] in G.nodes()]
                
                # 创建图形
                fig = go.Figure(data=[edge_trace, node_trace],
                              layout=go.Layout(
                                  title=dict(text='Neo4j交互式图', font=dict(size=16)),
                                  showlegend=False,
                                  hovermode='closest',
                                  margin=dict(b=20,l=5,r=5,t=40),
                                  width=None,  # 让图表自适应容器宽度
                                  height=600,  # 设置固定高度
                                  autosize=True,  # 启用自动调整大小
                                  annotations=[ dict(
                                      text="点击节点查看详细信息",
                                      showarrow=False,
                                      xref="paper", yref="paper",
                                      x=0.005, y=-0.002,
                                      xanchor="left", yanchor="bottom",
                                      font=dict(color="gray", size=12)
                                  )],
                                  xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                  yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                  plot_bgcolor='white',
                                  # 添加响应式配�?                                  responsive=True,
                                  # 设置容器样式
                                  paper_bgcolor='white',
                                  # 确保图表占满容器
                                  bargap=0,
                                  bargroupgap=0
                              ))
                
                return fig
                
        except Exception as e:
            logger.error(f"生成Neo4j Plotly图形失败: {str(e)}")
            return self._create_sample_plotly_graph()
    
    def _create_sample_plotly_graph(self):
        """创建示例Plotly图形"""
        try:
            import plotly.graph_objects as go
            import networkx as nx
            
            # 创建示例�?            G = nx.karate_club_graph()
            
            # 使用spring布局
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # 准备节点数据
            node_x = []
            node_y = []
            node_text = []
            node_info = []
            
            for node_id in G.nodes():
                x, y = pos[node_id]
                node_x.append(x)
                node_y.append(y)
                node_text.append(f"Node {node_id}")
                node_info.append(f"ID: {node_id}<br>Type: Example Node")
            
            # 准备边数�?            edge_x = []
            edge_y = []
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            # 创建边轨�?            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            # 创建节点轨迹
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="middle center",
                hovertext=node_info,
                marker=dict(
                    showscale=True,
                    colorscale='Viridis',
                    reversescale=True,
                    color=list(range(len(G.nodes()))),
                    size=20,
                    colorbar=dict(
                        thickness=15,
                        title="节点ID",
                        xanchor="left",
                        title_side="right"
                    ),
                    line=dict(width=2, color='white')
                )
            )
            
            # 创建图形
            fig = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              title=dict(text='示例交互式图 (Neo4j连接失败)', font=dict(size=16)),
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=20,l=5,r=5,t=40),
                              width=None,  # 让图表自适应容器宽度
                              height=600,  # 设置固定高度
                              autosize=True,  # 启用自动调整大小
                              annotations=[ dict(
                                  text="这是示例图，请检查Neo4j连接",
                                  showarrow=False,
                                  xref="paper", yref="paper",
                                  x=0.005, y=-0.002,
                                  xanchor="left", yanchor="bottom",
                                  font=dict(color="gray", size=12)
                              )],
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              plot_bgcolor='white',
                              # 添加响应式配�?                              responsive=True,
                              # 设置容器样式
                              paper_bgcolor='white',
                              # 确保图表占满容器
                              bargap=0,
                              bargroupgap=0
                          ))
            
            return fig
            
        except Exception as e:
            logger.error(f"创建示例图形失败: {str(e)}")
            return None

    def _generate_statistics_plotly(self):
        """生成统计可视�?- 使用Plotly"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            import os
            from pathlib import Path
            
            # 检查输出文件统�?            output_dirs = ['output/extract_output', 'output/description_output', 'output/vector_embedding_output']
            dir_names = []
            file_counts = []
            sizes = []
            
            for dir_name in output_dirs:
                if os.path.exists(dir_name):
                    files = list(Path(dir_name).glob('*'))
                    total_size = sum(f.stat().st_size for f in files if f.is_file())
                    dir_names.append(dir_name.split('/')[-1])
                    file_counts.append(len(files))
                    sizes.append(total_size/1024/1024)  # MB
            
            if not dir_names:
                # 如果没有数据，创建示例图
                dir_names = ['示例目录1', '示例目录2', '示例目录3']
                file_counts = [10, 25, 15]
                sizes = [5.2, 12.8, 7.5]
            
            # 创建子图
            from plotly.subplots import make_subplots
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('文件数量统计', '目录大小分布', '文件数量趋势', '大小分布'),
                specs=[[{"type": "bar"}, {"type": "pie"}],
                       [{"type": "scatter"}, {"type": "histogram"}]]
            )
            
            # 文件数量柱状�?            fig.add_trace(
                go.Bar(x=dir_names, y=file_counts, name="文件数量", marker_color='lightblue'),
                row=1, col=1
            )
            
            # 目录大小饼图
            fig.add_trace(
                go.Pie(labels=dir_names, values=sizes, name="目录大小"),
                row=1, col=2
            )
            
            # 文件数量散点�?            fig.add_trace(
                go.Scatter(x=dir_names, y=file_counts, mode='markers+lines', 
                          name="文件趋势", marker=dict(size=10, color='red')),
                row=2, col=1
            )
            
            # 大小分布直方�?            fig.add_trace(
                go.Histogram(x=sizes, name="大小分布", marker_color='green'),
                row=2, col=2
            )
            
            fig.update_layout(
                title_text="项目统计分析",
                title_x=0.5,
                height=600,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"生成统计可视化失�? {str(e)}")
            return self._create_sample_plotly_graph()
    
    
    def build_interface(self):
        """构建新布局的Gradio界面"""
        with gr.Blocks(
            title="OSSCompass 开源项目分析工�?- 新布局", 
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
            gr.Markdown("# 🚀 OSSCompass 开源项目分析工�?)
            gr.Markdown("输入开源项目链接，进行语义搜索、相似性对比、API推荐、可视化和聚类分析�?)
            
            # 上半部分：批量处�?            gr.Markdown("## 📁 批量处理")
            
            with gr.Row():
                with gr.Column(scale=4):
                    project_input = gr.Textbox(
                        label="项目链接",
                        placeholder="输入GitHub项目链接，每行一个\n例如:\nhttps://github.com/user/repo1\nhttps://github.com/user/repo2",
                        lines=10
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            clean_db_checkbox = gr.Checkbox(
                                label="清空数据�?,
                                value=False,
                                # info="处理前清空Neo4j数据�?
                            )
                        with gr.Column(scale=2):
                            process_btn = gr.Button("🚀 开始处�?, variant="primary")
                
                
            # 下半部分：处理结果和项目状�?            with gr.Row():
                # 左侧：处理结�?                with gr.Column(scale=1, elem_classes="result-container"):
                    gr.Markdown("## 📋 处理结果")
                    
                    process_output = gr.Textbox(
                        label="处理结果",
                        lines=11,
                        max_lines=25,
                        interactive=False,
                        show_copy_button=True,
                        elem_classes="result-textbox"
                    )
                
                # 右侧：项目状�?                with gr.Column(scale=1, elem_classes="status-container"):
                    gr.Markdown("## 📊 项目状�?)
                    
                    status_output = gr.Textbox(
                        label="系统状�?,
                        lines=8,
                        interactive=False,
                        show_copy_button=True,
                        elem_classes="status-textbox"
                    )
                    
                    with gr.Row():
                        status_btn = gr.Button("🔄 刷新状�?, variant="primary")
            
            # 绑定事件
            process_btn.click(
                fn=self.process_projects,
                inputs=[project_input, clean_db_checkbox],
                outputs=process_output
            )
            
            status_btn.click(
                fn=self.get_project_status,
                outputs=status_output
            )
            
            # 分隔�?            gr.Markdown("---")
            
            # 下半部分：书签切换的功能
            with gr.Tabs():
                # 语义搜索标签�?                with gr.Tab("🔍 语义搜索"):
                    gr.Markdown("## 智能语义搜索")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            search_query = gr.Textbox(
                                label="搜索查询",
                                placeholder="输入函数名、功能描述或关键�?,
                                lines=2
                            )
                            search_limit = gr.Slider(
                                minimum=1,
                                maximum=50,
                                value=10,
                                step=1,
                                label="结果数量"
                            )
                            search_btn = gr.Button("🔍 搜索", variant="primary")
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### 🔍 搜索说明")
                            gr.Markdown("""
                            **搜索功能:**
                            �?函数名搜�? 
                            �?功能描述搜索  
                            �?关键词搜�? 
                            �?相似化推�? 
                            
                            **智能特�?**
                            �?语义理解  
                            �?相似度排�? 
                            �?跨项目搜�? 
                            """)
                    
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
                
                # 可视化图标签�?                with gr.Tab("📊 可视化图"):
                    gr.Markdown("## 交互式节点图可视�?)
                    
                    # 控制面板
                    with gr.Row():
                        with gr.Column(scale=1):
                            viz_type = gr.Radio(
                                choices=["interactive", "neo4j", "statistics", "graph"],
                                value="interactive",
                                label="可视化类�?,
                                info="交互式图支持拖拽和选中功能"
                            )
                            viz_btn = gr.Button("📊 生成可视�?, variant="primary")
                        
                        with gr.Column(scale=2):
                            gr.Markdown("### 🎯 交互式图功能")
                            gr.Markdown("""
                            **交互式节点图特�?**
                            �?🖱�?拖拽节点重新布局  
                            �?👆 点击节点查看详情  
                            �?🔍 缩放和平移视�? 
                            �?🎨 多种布局算法  
                            �?⚙️ 可调整节点大小和边宽�? 
                            �?🎯 支持节点选中和高�? 
                            
                            **布局算法:**
                            �?力导向布局 (COSE)
                            �?层次布局 (Dagre)  
                            �?圆形布局 (Circle)
                            �?网格布局 (Grid)
                            """)
                    
                    # 图表区域 - 独立占满整行
                    viz_output = gr.Plot(
                        label="交互式可视化�?,
                        value=None,
                        container=True,
                        show_label=True,
                        scale=1,
                        elem_classes=["full-width-plot"],
                        # 添加更多布局参数
                        elem_id="main-plot"
                    )
                    
                    viz_btn.click(
                        fn=self.generate_visualization,
                        inputs=[viz_type],
                        outputs=viz_output
                    )
                
            
            # 页面加载时自动刷新状�?            demo.load(
                fn=self.get_project_status,
                outputs=status_output
            )
            
            # 添加自定义CSS样式确保图表占满宽度
            demo.css = """
            .full-width-plot {
                width: 100% !important;
                max-width: 100% !important;
                min-width: 100% !important;
            }
            
            .full-width-plot .plotly {
                width: 100% !important;
                max-width: 100% !important;
            }
            
            .full-width-plot .js-plotly-plot {
                width: 100% !important;
                max-width: 100% !important;
            }
            
            #main-plot {
                width: 100% !important;
                max-width: 100% !important;
            }
            
            #main-plot .plotly {
                width: 100% !important;
                max-width: 100% !important;
            }
            """
        
        return demo
    
    def launch(self, server_port=8002, share=False):
        """启动应用"""
        if not self.initialize_components():
            print("组件初始化失败，但可以继续启动应�?)
            print("某些功能可能无法正常使用")
        
        demo = self.build_interface()
        demo.launch(
            server_port=server_port,
            share=share,
            show_error=True,
            quiet=False
        )

def main():
    """主函�?""
    app = OSSCompassAppNew()
    app.launch()

if __name__ == "__main__":
    main()
