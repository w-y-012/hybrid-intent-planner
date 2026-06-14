# -*- coding: utf-8 -*-
import quiet_logging  # 必须是第一行
import json
import os
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer


# =========================================================================
# 1) 全局槽位与 DSL 基础类型定义 (原 shared_types)
# =========================================================================
SLOT_TAG = "__SLOT__"
GENERIC_SLOT_TAG = "__GENERIC_SLOT__"


@dataclass
class DSLConstraint:
    field: str
    operator: str  # "EQUALS" | "CONTAINS" | "IN"
    value: Any


@dataclass
class KGQueryDSL:
    subject: str
    predicate: str
    object: str
    is_negated: bool = False
    constraints: List[DSLConstraint] = field(default_factory=list)
    output_mode: str = "single"  # "single" | "list"


# =========================================================================
# 2) 数据库与模型全局配置
# =========================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("BasePipeline")


# =========================================================================
# 2) 数据库与模型全局配置 (补充谓词集合名称)
# =========================================================================
class DBConfig:
    ENTITIES_COL = "graph_entities"
    COMMUNITIES_COL = "graph_communities"
    CHUNKS_COL = "chunks"
    PREDICATES_COL = "graph_predicates" # 【新增】专门用于 OIE 谓词路由的高速向量网关集合



@dataclass
class UnifiedConfig:
    qwen_api_key: str = field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", ""))
    qwen_model: str = "qwen-turbo"
    qwen_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    embed_model_name: str = "BAAI/bge-small-en-v1.5"
    node2vec_dimensions: int = 64

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "test1234"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333


# =========================================================================
# 3) 适配器设计：解决 SentenceTransformer 无 embed_query 报错问题
# =========================================================================
class SentenceTransformerWrapper:
    """包装原生 SentenceTransformer，使其对齐支持 LangChain 类 embed_query 标准接口"""

    def __init__(self, model: SentenceTransformer):
        self.model = model

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def __getattr__(self, name):
        # 兜底将其他属性调用委托给原生的 SentenceTransformer
        return getattr(self.model, name)


class ModelHub:
    _embedder = None

    @classmethod
    def get_embedder(cls, model_name: str = "BAAI/bge-small-en-v1.5",
                     device: str = "cpu") -> SentenceTransformerWrapper:
        if cls._embedder is None:
            local_path = os.path.expanduser(
                f"~/.cache/huggingface/hub/models--{model_name.replace('/', '--')}/snapshots"
            )
            if os.path.exists(local_path):
                dirs = os.listdir(local_path)
                if dirs:
                    model_path = os.path.join(local_path, dirs[0])
                    logger.info(f"[ModelHub] 检测并从本地缓存加载 Embedding: {model_path}")
                    raw_model = SentenceTransformer(model_path, device=device)
                    cls._embedder = SentenceTransformerWrapper(raw_model)
                    return cls._embedder

            logger.info(f"[ModelHub] 在线载入 Embedding 模型: {model_name}")
            raw_model = SentenceTransformer(model_name, device=device)
            cls._embedder = SentenceTransformerWrapper(raw_model)
        return cls._embedder


# 初始化通用全局 LLM
cfg = UnifiedConfig()
llm = ChatOpenAI(
    model=cfg.qwen_model,
    temperature=0.1,
    openai_api_key=cfg.qwen_api_key,
    openai_api_base=cfg.qwen_api_base,
    timeout=15
)

# =========================================================================
# 中央超参动态载入与持久化模块
# =========================================================================
@dataclass
class RAGHyperparams:
    # 相似度决策阈值
    align_predicate_th: float = 0.72
    entity_link_th: float = 0.75
    bridge_th: float = 0.82
    min_recall_gate: float = 0.55

    # 句法树加减分权重
    w_main_predicate_root: int = 20
    w_modifier_predicate: int = -15
    w_arg_bonus: int = 2

    # 自适应置信度标记
    conf_cascade_plan: float = 0.95
    conf_single_hop: float = 0.88
    conf_fallback_vector: float = 0.45

    def save(self, filepath: str = "rag_hyperparameters.json"):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str = "rag_hyperparameters.json") -> "RAGHyperparams":
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls(**data)
            except Exception as e:
                pass
        # 默认参数
        return cls()


# 全局超参单例
hyperparams = RAGHyperparams.load()
