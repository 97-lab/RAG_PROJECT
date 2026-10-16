# RAG_PROJECT —— 自适应自评估 RAG 问答系统

基于 **LangChain + LangGraph** 构建的进阶 RAG（检索增强生成）问答系统。相比朴素 RAG，它在检索前后加入了多重 LLM 决策与自评估闭环：问题路由 → 混合检索 → 相关性过滤 → 查询重写 → 生成 → 幻觉/有用性评分，不合格自动纠错重试。

知识库领域：**半导体/芯片技术 + Milvus 向量数据库文档**。

## 项目亮点

- **自适应路由**：入口由 LLM 判断问题走向量知识库还是联网搜索
- **混合检索**：Milvus 中 BM25 稀疏向量 + BGE 稠密向量，RRF 融合排序
- **文档过滤**：检索结果逐条做相关性评分，过滤噪声
- **查询重写**：检索失败时 LLM 重写问题再检索（最多 2 次，超限转联网）
- **自评估闭环**：生成后做幻觉评分 + 有用性评分，不合格回环重试
- **可视化编排**：LangGraph 状态图，支持 draw_graph 导出流程图

## 项目结构

```
RAG_PROJECT/
├── graph2/                  # 主力工作流：自适应自评估 RAG（LangGraph）
│   ├── graph_2.py           # 图编排：节点 + 条件边 + 交互入口
│   ├── graph_state2.py      # 共享状态定义（question/documents/generation/transform_count）
│   ├── query_route_chain.py # 问题路由链（vectorstore / web_search）
│   ├── grader_chain.py      # 文档相关性评分器
│   ├── grade_hallucinations_chain.py  # 幻觉评分器
│   ├── grade_answer_chain.py          # 有用性评分器
│   ├── transform_query_node.py        # 查询重写节点
│   ├── generate_node2.py              # 生成节点（RAG 链）
│   ├── web_search_node.py             # 联网搜索节点
│   └── graph_gradio.py      # Gradio 聊天界面
├── graph/                   # 备用工作流：Agentic RAG（bind_tools + ToolNode + MemorySaver）
├── agent/                   # AgentExecutor 封装
├── tools/
│   └── retriever_tools.py   # Milvus 混合检索器（BM25 + 向量，RRF）→ LangChain Tool
├── documents/               # 数据层
│   ├── milvus_db.py         # 建表：稀疏(BM25/jieba) + 稠密(HNSW/IP, dim=512) 双字段
│   ├── write_milvus.py      # 双进程流水线写入（解析进程 + 写入进程）
│   └── markdown_parser.py   # Markdown 解析 + 标题合并 + 语义切分(>5000字)
├── llm_models/
│   ├── all_llm.py           # LLM 实例（deepseek-chat）+ Tavily 搜索工具
│   └── embeddings_model.py  # 本地 BGE 嵌入（BAAI/bge-small-zh-v1.5, 512维）
├── datas/md/                # 知识库原始文档（半导体技术报告 + Milvus 文档）
├── utils/                   # 环境变量、日志（loguru）、打印工具
└── requirements.txt         # 依赖锁定清单（Python 3.11）
```

## 工作流程

```
用户问题
   │
   ├─ 问题路由（LLM 结构化输出）──非半导体问题──→ 联网搜索 ──→ 生成
   │
   └─ 半导体问题 ──→ Milvus 混合检索（BM25 + BGE 向量，RRF 融合）
                        │
                        ▼
                  文档相关性过滤 ──全不相关──→ 查询重写（≤2次，超限转联网）
                        │ 有相关文档
                        ▼
                     生成答案 ──→ 幻觉评分 + 有用性评分
                        │             ├─ 有幻觉 → 重新生成
                        │             ├─ 答非所问 → 查询重写
                        │             └─ 通过 → END
```

## 技术栈

| 类别 | 技术 |
|---|---|
| 编排 | LangChain 0.3 / LangGraph 0.3 |
| LLM | DeepSeek（deepseek-chat，OpenAI 兼容接口） |
| Embedding | BAAI/bge-small-zh-v1.5（本地 HuggingFace，512 维） |
| 向量库 | Milvus 2.5 Standalone（Docker Compose 部署） |
| 检索 | BM25 稀疏 + HNSW 稠密 + RRF 融合 |
| 联网搜索 | Tavily |
| 文档解析 | unstructured + SemanticChunker |
| UI | Gradio |
| 其他 | Pydantic（结构化输出）、loguru、python-dotenv |

## 快速开始

### 1. 环境要求

- Python **3.11**（requirements 按 3.11 锁定，3.13 会因 onnxruntime 等包安装失败）
- Docker Desktop（需 WSL2）

### 2. 创建虚拟环境并安装依赖

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 配置环境变量

在项目根目录创建 `.env`：

```ini
# 必填：DeepSeek API Key（https://platform.deepseek.com）
DEEPSEEK_API_KEY=sk-你的key

# 联网搜索用（不联网可不填，并注释 all_llm.py 中的 web_search_tool）
TAVILY_API_KEY=tvly-你的key
```

### 4. 启动 Milvus

```powershell
cd RAG_PROJECT
curl.exe -L -o docker-compose.yml https://github.com/milvus-io/milvus/releases/download/v2.5.6/milvus-standalone-docker-compose.yml
docker compose up -d          # 首次拉镜像需几分钟
docker compose ps             # 确认三个容器均为 healthy
```

> 注意文件名后缀是 `.yml` 不是 `.yaml`。版本需与 pymilvus 2.5.6 匹配，勿用 3.x。

确认 [utils/env_utils.py](utils/env_utils.py) 中 `MILVUS_URI` 指向 `http://localhost:19530`。

### 5. 导入知识库

```powershell
python documents/write_milvus.py   # 解析 datas/md 下的文档并写入 Milvus
```

### 6. 运行

```powershell
# 命令行交互（graph2 主工作流）
python graph2/graph_2.py

# 或 Gradio 网页界面
python graph2/graph_gradio.py
```

## 评测（Ragas）

支持上下文召回率 / 上下文精确率 / 忠实度三指标评测：

```powershell
pip install ragas datasets
```

准备评测集（question + 人工标注 ground_truth），运行流程抓取 answer/contexts 后交给 Ragas 计算：

```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_recall, context_precision, faithfulness

dataset = Dataset.from_dict({
    "question": [...],        # 评测问题
    "answer": [...],          # graph 生成的答案
    "contexts": [[...]],      # 检索到的文档
    "ground_truth": [...],    # 人工标注标准答案
})
result = evaluate(dataset, metrics=[context_recall, context_precision, faithfulness])
```

## 常见问题

| 问题 | 解决 |
|---|---|
| `Did not find tavily_api_key` | `.env` 填 TAVILY_API_KEY，或注释 all_llm.py 中 `web_search_tool` |
| 连不上 Milvus（illegal connection params） | 确认 `docker compose ps` 三容器 healthy，URI 为 localhost |
| PyCharm 报找不到 env_utils | 右键项目根目录 → Mark Directory as → Sources Root |
| BGE 模型下载慢 | 设置环境变量 `HF_ENDPOINT=https://hf-mirror.com` |
| requirements 安装失败 | 使用 Python 3.11，不要用 3.13 |

## License

仅供学习交流使用。
