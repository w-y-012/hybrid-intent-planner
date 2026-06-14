# Controllable RAG Agent

> 基于 LangGraph 的企业级 RAG 知识图谱问答系统，支持动态意图分析、多跳推理、自适应路由与可视化追踪。

## 🏗️ 系统架构
用户 Query
│
▼
┌──────────────────────────────────────────────────────┐
│ LangGraph 工作流 │
│ │
│ intent_analyzer → planner → validate → execute → synthesize
│ │
│ 5 个节点：意图分析 → 计划生成 → 计划验证 → 执行检索 → 答案合成 │
└──────────────────────────────────────────────────────┘
│ │ │
▼ ▼ ▼
┌──────────┐ ┌──────────────┐ ┌──────────────┐
│ Stanza │ │ Neo4j (KG) │ │ Qdrant (向量) │
│ 依存分析 │ │ 知识图谱查询 │ │ 混合检索 │
└──────────┘ └──────────────┘ └──────────────┘

## ✨ 核心特性

### 1. 动态意图分析 (`intent_analyzer.py`)
- **零硬编码**：基于 Stanza Universal Dependencies 的纯特征级疑问判定
- **递归依存树追踪**：自动识别间接疑问词（如 "教什么课" 中的 "课"）
- **粘连疑问谓词处理**：处理 "教哪"、"去哪" 等被误分词的情况
- **被动语态归一化**：自动交换施受关系，匹配 KG 边方向
- **论元自愈**：Stanza 漏掉的主语/宾语自动从 KG 实体列表或主事件补全

### 2. 自适应路由系统
| 路由类型 | 触发条件 | 示例 |
|---------|---------|------|
| `SINGLE_HOP` | 主谓宾完整，单槽位 | "Hermione保护了谁？" |
| `MULTI_HOP_CASCADE` | 含定语从句 + 概念消解 | "帮助反派的那个教授教什么课？" |
| `MULTI_HOP` | 多实体链式查询 | "Harry的对手的老师是谁？" |
| `CONCEPT_MATCH` | 依存失败但有实体 | "Voldemort" |
| `SUBGRAPH_CO_OCCUR` | 多实体子图共现 | "Harry和Voldemort之间有什么关系？" |
| `VECTOR` | 完全降级 | "哈利波特里最勇敢的人是谁？" |

### 3. 知识图谱检索 (`retriever_kg.py`)
- **实体链接**：O(1) 内存级 HashMap 缓存 + 主动实体名消解
- **谓词对齐**：双语同义词表 + 向量余弦相似度匹配
- **K-近邻自愈**：原始谓词未命中时自动泛化到近邻关系
- **闭环事实修正**：LLM 从文本中推断正确实体名后重新查询

### 4. Schema 类型解析器 (`SchemaTypeResolver`)
- 在线查询 KG Schema 获取候选标签
- Embedding 向量余弦相似度动态选择最佳类型
- 自动适配不同 KG 的标签体系

### 5. 英文特征空间对齐 (HyDE)
- 中文 Query 自动翻译为英文
- LLM 生成假设性英文段落增强向量检索
- 检索效果提升 17%-32%

### 6. 可视化追踪平台 (`simulate_agent.py`)
- Streamlit 实时执行流追踪
- 拓扑路由高亮（PyVis 动态图）
- 终端日志实时捕获
- State 更新逐行展示

## 📦 技术栈

| 组件 | 技术 |
|------|------|
| 工作流编排 | LangGraph + LangChain |
| 依存句法分析 | Stanza (UD 中文模型) |
| 知识图谱 | Neo4j (736 实体, 325 关系类型) |
| 向量检索 | Qdrant + BGE-small-en-v1.5 |
| 可视化 | Streamlit + PyVis |
| LLM | OpenAI API / 兼容接口 |

## 🚀 快速开始

### 环境要求
- Python 3.12+
- Neo4j (bolt://localhost:7687)
- Qdrant (localhost:6333)

### 安装
```bash
# 克隆仓库
git clone https://github.com/your-username/Controllable-RAG-Agent.git
cd Controllable-RAG-Agent

# 安装依赖
pip install -r requirements.txt

# 下载 Stanza 中文模型（首次运行自动下载）
python -c "import stanza; stanza.download('zh')"
# 1. 启动 Neo4j
neo4j start

# 2. 启动 Qdrant
docker run -p 6333:6333 qdrant/qdrant

# 3. 运行离线评估
python executor_agent.py
浏览器访问 http://localhost:8501

📁 项目结构
text
├── intent_analyzer.py      # 动态意图分析编译器
├── retriever_kg.py          # 知识图谱 + 向量检索
├── executor_agent.py        # LangGraph 工作流编排
├── enterprise_evaluation.py # 离线评估 + 在线哨兵
├── enterprise_governance.py # MCP 沙盒 + 事务治理
├── functions_for_pipeline.py # 全局配置与工具函数
├── simulate_agent.py        # Streamlit 可视化追踪
├── run_optuna_tuning.py     # 贝叶斯超参调优
└── predicate_config.json    # 谓词双语对照表
📊 测试用例
Case	Query	预期路由	预期答案
1	帮助反派的那个教授教什么课？	MULTI_HOP_CASCADE	Defence Against the Dark Arts
2	Hermione保护了谁？	SINGLE_HOP	Harry Potter
3	Severus Snape教哪一门特定的科目？	SINGLE_HOP	Potions
📝 超参调优
bash
python run_optuna_tuning.py
自动寻找最优的谓词对齐阈值、实体链接阈值等 7 个参数，结果保存至 rag_hyperparameters.json。

<img width="1809" height="1298" alt="🤖 RAG Agent 实时追踪平台" src="https://github.com/user-attachments/assets/2d4277e1-9b9d-4dcb-84b7-da3d83c96bea" />



