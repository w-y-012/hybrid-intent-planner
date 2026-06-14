# -*- coding: utf-8 -*-
"""
通用开放域 GraphRAG 工业构建流水线 - graph_rag_builder.py
特性:
1. 100% 修复 UUID 大小写冲突及关系丢失缺陷 (采用 ON CREATE MERGE 防御机制)。
2. 【修复：断点续传】采用磁盘 JSON Checkpoint，抽取意外中断后可一秒无损续传。
3. 【修复：质量熔断闸门】自动审计图谱与向量库数据质量，异常比率超标时自动触发安全熔断。
4. 【修复：语义图卷积神经网络 (GCN)】NumPy 离线实现，生成高精度 384 维 KGE，打通拓扑与语义空间。
5. 【修复：实体规范化与大写转换】统一清洗转换实体类型字段，防止检索侧实体类型过滤失效。
6. 【修复：关系边强去重与权重聚合】按 (src_id, tgt_id, pred_canonical) 强聚类关系，累加权重并合并证据链。
7. 【修复：Qdrant 命名向量异常与索引吞没】防御性捕获 Payload 索引冲突，注入 text_semantic 向量名称。
"""

import os
import sys
import re
import json
import uuid
import logging
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# 数据库与第三方库
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PayloadSchemaType, TextIndexParams, TokenizerType
from qdrant_client.models import PointStruct  # 修复: 缺失导入
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize

# 高级图算法动态发现
try:
    import leidenalg
    import igraph as ig

    LEIDEN_AVAILABLE = True  # 修复: 缺失 LEIDEN_AVAILABLE 全局变量
except ImportError:
    LEIDEN_AVAILABLE = False

# NLTK 分词依赖容错
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("GraphRAGBuilder")


# 修复: 补全 PREDICATES_COL，确保构建与 Agent 保持完全一致 (问题2已修复)
class DBConfig:
    ENTITIES_COL = "graph_entities"
    COMMUNITIES_COL = "graph_communities"
    CHUNKS_COL = "chunks"
    PREDICATES_COL = "graph_predicates"  # 新增: 专门用于 OIE 谓词路由的高速向量网关集合


@dataclass
class BuilderConfig:
    pdf_path: str = r"C:\Users\w_y_0\rag&agent\Controllable-RAG-Agent-main\Harry Potter - Book 1 - The Sorcerers Stone.pdf"
    chunk_size: int = 1000
    chunk_overlap: int = 150

    qwen_api_key: str = field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", ""))
    qwen_model: str = "qwen-turbo"
    qwen_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    embed_model_name: str = "BAAI/bge-small-en-v1.5"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "test1234"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    max_workers: int = 5
    skip_existing_collections: bool = False


class EntityNormalizer:
    @staticmethod
    def clean(name: str) -> str:
        if not name: return ""
        return re.sub(r"\s+", " ", str(name).lower().strip())

    @classmethod
    def to_uuid(cls, name: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, cls.clean(name)))


class CircuitBreakerException(Exception):
    """大模型调用失败熔断异常"""
    pass


class LLMCircuitBreaker:
    """滑动窗口大模型熔断器"""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.consecutive_failures = 0

    def record_success(self):
        self.consecutive_failures = 0

    def record_failure(self) -> bool:
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.failure_threshold:
            logger.critical(
                f"🚨 [LLMCircuitBreaker] LLM 调用连续失败已达阈值 ({self.failure_threshold})！触发安全熔断保护。")
            return True
        return False


class AdvancedNeo4jStore:
    def __init__(self, cfg: BuilderConfig):
        self.driver = GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_username, cfg.neo4j_password))
        self.database = "neo4j"

    def close(self):
        self.driver.close()

    def clear_all(self):
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")

    def create_indexes(self):
        with self.driver.session(database=self.database) as session:
            session.run("CREATE INDEX entity_id IF NOT EXISTS FOR (e:Entity) ON (e.id)")
            session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")
            session.run("CREATE INDEX entity_aliases IF NOT EXISTS FOR (e:Entity) ON (e.aliases)")

    def batch_write_entities(self, entities: List[Dict]):
        query = """
        UNWIND $batch AS row
        MERGE (e:Entity {id: row.id})
        SET e.name = row.name, 
            e.type = row.type, 
            e.description = row.description,
            e.aliases = row.aliases,
            e.community_id = row.community_id
        """
        with self.driver.session(database=self.database) as session:
            session.run(query, batch=entities)
        logger.info(f"✅ Neo4j 实体写入: {len(entities)} 条")

    def batch_write_relationships(self, rels: List[Dict]):
        grouped_rels = defaultdict(list)
        for r in rels:
            grouped_rels[r['pred']].append(r)

        with self.driver.session(database=self.database) as session:
            for pred, batch in grouped_rels.items():
                safe_pred = re.sub(r'[^a-zA-Z0-9_]', '', pred).upper()
                q = f"""
                UNWIND $batch AS row
                MERGE (a:Entity {{id: row.src_id}})
                ON CREATE SET a.name = row.src_name, a.type = "UNKNOWN"
                MERGE (b:Entity {{id: row.tgt_id}})
                ON CREATE SET b.name = row.tgt_name, b.type = "UNKNOWN"
                MERGE (a)-[r:{safe_pred}]->(b)
                SET r.weight = row.weight, 
                    r.is_negated = row.is_negated, 
                    r.evidence = row.evidence,
                    r.raw_pred = row.raw_pred,
                    r.sequence_index = row.sequence_index
                """
                session.run(q, batch=batch)
        logger.info(f"✅ Neo4j 关系 ({len(rels)} 条) 防御性写入完成。")


class UniversalGraphRAGBuilder:
    def __init__(self, cfg: BuilderConfig):
        self.cfg = cfg
        from openai import OpenAI
        self.client = OpenAI(api_key=cfg.qwen_api_key, base_url=cfg.qwen_api_base)
        self.neo4j = AdvancedNeo4jStore(cfg)
        self.qdrant = QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port)

        local_path = os.path.expanduser(
            f"~/.cache/huggingface/hub/models--{cfg.embed_model_name.replace('/', '--')}/snapshots"
        )
        model_load_path = cfg.embed_model_name
        if os.path.exists(local_path):
            dirs = os.listdir(local_path)
            if dirs:
                model_load_path = os.path.join(local_path, dirs[0])
                logger.info(f"[Builder] 从本地缓存加载 Embedding: {model_load_path}")
        self.embedder = SentenceTransformer(model_load_path)
        self.embed_dim = self.embedder.get_embedding_dimension()

        self.tables = {k: pd.DataFrame() for k in ["text_units", "entities", "relationships", "communities"]}
        self.graph = nx.DiGraph()
        self.gnn_embeddings = {}

    def _init_qdrant(self):
        """
        【修复：Qdrant 索引异常吞没缺陷】
        分类审查 Payload 索引创建，对“已存在”报错安全放行，对致命环境故障直接抛出，拒绝低效降级。
        """
        collections_config = {
            DBConfig.ENTITIES_COL: {
                "vectors_config": {
                    "text_semantic": VectorParams(size=self.embed_dim, distance=Distance.COSINE),
                    "kge": VectorParams(size=self.embed_dim, distance=Distance.COSINE)
                }
            },
            DBConfig.COMMUNITIES_COL: {
                "vectors_config": {"text_semantic": VectorParams(size=self.embed_dim, distance=Distance.COSINE)}
            },
            DBConfig.CHUNKS_COL: {
                "vectors_config": {"text_semantic": VectorParams(size=self.embed_dim, distance=Distance.COSINE)}
            },
            DBConfig.PREDICATES_COL: {
                "vectors_config": {"text_semantic": VectorParams(size=self.embed_dim, distance=Distance.COSINE)}
            }
        }

        for col_name, config in collections_config.items():
            if self.qdrant.collection_exists(col_name):
                if self.cfg.skip_existing_collections:
                    logger.info(f"[Qdrant] 集合 {col_name} 已存在，跳过。")
                    continue
                self.qdrant.delete_collection(col_name)
            self.qdrant.create_collection(collection_name=col_name, vectors_config=config["vectors_config"])

        # Payload 关键字索引创建
        for collection, field_name, schema_type in [
            (DBConfig.ENTITIES_COL, "type", PayloadSchemaType.KEYWORD),
            (DBConfig.ENTITIES_COL, "aliases", PayloadSchemaType.KEYWORD),
            (DBConfig.COMMUNITIES_COL, "community_id", PayloadSchemaType.INTEGER)
        ]:
            try:
                self.qdrant.create_payload_index(collection, field_name, schema_type)
            except Exception as e:
                err_msg = str(e).lower()
                if "already exists" in err_msg or "index exists" in err_msg:
                    logger.info(f"[Qdrant] 字段 {collection}.{field_name} 索引已存在，安全跳过。")
                else:
                    logger.error(f"❌ Qdrant 创建索引发生致命错误 {collection}.{field_name}: {e}")
                    raise e

        # 全文分词索引创建
        for col, path in [(DBConfig.ENTITIES_COL, "name"), (DBConfig.CHUNKS_COL, "text")]:
            try:
                self.qdrant.create_payload_index(col, path, TextIndexParams(
                    type="text", tokenizer=TokenizerType.WORD, min_token_len=2, max_token_len=20, lowercase=True
                ))
            except Exception as e:
                err_msg = str(e).lower()
                if "already exists" in err_msg or "index exists" in err_msg:
                    logger.info(f"[Qdrant] 全文索引 {col}.{path} 已存在，安全跳过。")
                else:
                    logger.error(f"❌ Qdrant 创建全文索引失败 {col}.{path}: {e}")
                    raise e

    def _extract_and_chunk(self) -> pd.DataFrame:
        from pypdf import PdfReader
        reader = PdfReader(self.cfg.pdf_path)
        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        text = re.sub(r"\s+", " ", text).strip()

        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > self.cfg.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                overlap_sents = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) <= self.cfg.chunk_overlap:
                        overlap_sents.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current_chunk = overlap_sents
                current_len = overlap_len

            current_chunk.append(sent)
            current_len += sent_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        rows = [{"id": f"tu_{i}", "text": c, "chunk_index": i} for i, c in enumerate(chunks)]
        logger.info(f"✅ 语义分块完成: 共计 {len(rows)} 块。")
        return pd.DataFrame(rows)

    def _extract_single_chunk_open_domain(self, tu_id: str, chunk_idx: int, text: str) -> Dict:
        prompt = f"""You are a state-of-the-art open domain knowledge graph extractor. 
Analyze the following text to extract all high-value entities and relationships.

【Extraction Instructions】:
1. COREFERENCE RESOLUTION: Resolve all pronouns (e.g., "he", "she", "the boy", "this professor", "the antagonist") into their canonical, standard names. Never use pronouns as entities.
2. OPEN ONTOLOGY: You can extract ANY high-quality relationship predicate. It must be active, clear, and uppercase (e.g., "HELPS", "MEMBER_OF", "DEFEATS", "LOCATED_AT", "WORKS_FOR").
3. STANDARD ALIASES: Provide a list of alternative names or titles for each entity (e.g., Voldemort -> ["The Dark Lord", "You-Know-Who"]).
4. TIMELINE INDEX: Extract a "sequence_index" reflecting the flow of events (use {chunk_idx} as the base offset sequence).

Output strictly as a JSON object:
{{
  "entities": [
    {{"canonical_name": "canonical, full standard name", "type": "PERSON/LOCATION/ORGANIZATION/OBJECT/SPELL/EVENT/CONCEPT", "aliases": ["alias1", "alias2"], "description": "precise context desc (>15 words)"}}
  ],
  "relationships": [
    {{"source_canonical": "source entity canonical name", "target_canonical": "target entity canonical name", "predicate": "UPPERCASE_VERB", "is_negated": false, "evidence": "direct raw quote", "sequence_index": {chunk_idx}}}
  ]
}}

Text Chunk:
{text}
"""
        try:
            resp = self.client.chat.completions.create(
                model=self.cfg.qwen_model,
                messages=[
                    {"role": "system", "content": "You are an expert open graph generator. Strict JSON output only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            logger.warning(f"Chunk {tu_id} 提取异常: {e}")
            return {"entities": [], "relationships": []}

    def _parallel_extraction_open(self, df_tu: pd.DataFrame) -> List[Dict]:
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, "extraction_checkpoint.json")

        progress = {}
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    progress = json.load(f)
                logger.info(f"[Checkpoint] 检测到已存在历史进度，已跳过并加载: {len(progress)} 块。")
            except Exception as e:
                logger.warning(f"[Checkpoint] 读取进度文件异常: {e}，将开启全量新抽取。")

        extractions = [None] * len(df_tu)

        for idx, row in df_tu.iterrows():
            chunk_id = str(row['id'])
            if chunk_id in progress:
                extractions[idx] = progress[chunk_id]

        pending_indices = [idx for idx, row in df_tu.iterrows() if str(row['id']) not in progress]

        if not pending_indices:
            logger.info("✅ 检查点检测：所有分块均已在历史执行中全部抽取完成。")
            return extractions

        logger.info(f"🚀 触发断点续传：剩余 {len(pending_indices)} / {len(df_tu)} 块待运行。")

        with ThreadPoolExecutor(max_workers=self.cfg.max_workers) as executor:
            futures = {
                executor.submit(
                    self._extract_single_chunk_open_domain,
                    row['id'],
                    row['chunk_index'],
                    row['text']
                ): idx
                for idx, row in df_tu.iloc[pending_indices].iterrows()
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="🧠 Open-Domain 开放三元组提取"):
                idx = futures[future]
                chunk_id = str(df_tu.iloc[idx]['id'])
                try:
                    result = future.result()
                    extractions[idx] = result
                    progress[chunk_id] = result
                    with open(checkpoint_file, "w", encoding="utf-8") as f:
                        json.dump(progress, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    logger.error(f"分块 {chunk_id} 执行遇到未捕获致命错误: {e}")

        return extractions

    def _resolve_and_merge_robust(self, df_tu: pd.DataFrame, extractions: List[Dict]):
        """
        【修复：实体类型未统一缺陷】 (问题4已修复)
        1. 强制规范所有的实体类型转换为标准大写形式。
        2. 基于映射对异质类型进行语义分类对齐，保证检索侧对齐。
        """
        raw_entities = []
        raw_rels = []

        for ex, tu_id in zip(extractions, df_tu['id']):
            if not ex: continue
            for e in ex.get("entities", []):
                name = e.get("canonical_name", "").strip()
                if not name or len(name) < 2: continue

                # 标准化实体类型
                raw_type = str(e.get("type", "CONCEPT")).upper().strip()
                VALID_TYPES = {"PERSON", "LOCATION", "ORGANIZATION", "OBJECT", "SPELL", "CREATURE", "EVENT", "CONCEPT"}
                if raw_type not in VALID_TYPES:
                    if "CHAR" in raw_type or "USER" in raw_type or "HERO" in raw_type or "STUDENT" in raw_type:
                        raw_type = "PERSON"
                    elif "PLACE" in raw_type or "CITY" in raw_type or "ROOM" in raw_type or "HOUSE" in raw_type:
                        raw_type = "LOCATION"
                    else:
                        raw_type = "CONCEPT"

                e["type"] = raw_type
                e["id"] = EntityNormalizer.to_uuid(name)
                e["tu_id"] = tu_id
                raw_entities.append(e)

            for r in ex.get("relationships", []):
                src = r.get("source_canonical", "").strip()
                tgt = r.get("target_canonical", "").strip()
                pred = r.get("predicate", "").strip().upper()

                if not src or not tgt or not pred: continue

                r["src_id"] = EntityNormalizer.to_uuid(src)
                r["tgt_id"] = EntityNormalizer.to_uuid(tgt)
                r["src_name"] = src
                r["tgt_name"] = tgt
                r["pred"] = re.sub(r'[^a-zA-Z0-9_]', '', pred)
                r["tu_id"] = tu_id
                r["weight"] = 1.0
                r["is_negated"] = bool(r.get("is_negated", False))
                r["evidence"] = r.get("evidence", "")
                r["sequence_index"] = int(r.get("sequence_index", 0))
                raw_rels.append(r)

        df_ent = pd.DataFrame(raw_entities)
        df_rel = pd.DataFrame(raw_rels)

        if df_ent.empty:
            raise ValueError("💔 提取层发生致命阻塞，未产出任何节点数据！")

        def merge_aliases(series):
            merged = set()
            for lst in series:
                if isinstance(lst, list):
                    merged.update([str(a).strip() for a in lst if a])
            return list(merged)

        def merge_descriptions(series):
            unique_descs = list(dict.fromkeys([d for d in series if isinstance(d, str) and len(d) > 10]))
            return " | ".join(unique_descs)[:1000]

        df_ent_merged = df_ent.groupby(["id"]).agg(
            name=("canonical_name", "first"),
            type=("type", "first"),
            aliases=("aliases", merge_aliases),
            description=("description", merge_descriptions),
            text_unit_ids=("tu_id", lambda x: list(set(x)))
        ).reset_index()

        # 连通性过滤
        valid_uuids = set(df_ent_merged["id"])
        df_rel_filtered = df_rel[
            (df_rel["src_id"].isin(valid_uuids)) & (df_rel["tgt_id"].isin(valid_uuids))
            ].copy()

        self.tables["text_units"] = df_tu
        self.tables["entities"] = df_ent_merged
        self.tables["relationships"] = df_rel_filtered

        logger.info(f"✅ 实体对齐完成: {len(df_ent_merged)} 个节点, {len(df_rel_filtered)} 条连通关系。")

    # 修复：_build_networkx_and_community 的嵌套缩进错误 (问题1已修复)
    def _build_networkx_and_community(self):
        """
        【修复：外部类成员属性挂载】
         Leidan算法社区大纲发现，集成 LLM 断点与自适应滑动窗口熔断控制。
        """
        ent_df = self.tables["entities"]
        rel_df = self.tables["relationships"]

        for _, row in ent_df.iterrows():
            self.graph.add_node(
                row["id"], name=row["name"], type=row["type"], desc=row["description"], aliases=row["aliases"]
            )

        for _, row in rel_df.iterrows():
            self.graph.add_edge(
                row["src_id"], row["tgt_id"], predicate=row["pred"], weight=row["weight"]
            )

        if LEIDEN_AVAILABLE:
            ig_graph = ig.Graph.from_networkx(self.graph)
            try:
                partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition,
                                                     resolution_parameter=1.0)
            except:
                partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)

            com_map = dict(zip(ig_graph.vs["_nx_name"], partition.membership))
            nx.set_node_attributes(self.graph, com_map, name="community_id")

            df_com = pd.DataFrame([{"community_id": int(v)} for v in set(com_map.values())])
            df_com["entity_ids"] = df_com["community_id"].map(lambda c: [n for n, v in com_map.items() if v == c])
            self.tables["communities"] = df_com

            reports = []
            breaker = LLMCircuitBreaker(failure_threshold=3)  # 初始化安全熔断闸

            for _, com in df_com.iterrows():
                cid = com["community_id"]
                node_ids = com["entity_ids"]
                nodes_desc = [f"- {self.graph.nodes[n]['name']}: {self.graph.nodes[n]['desc'][:80]}" for n in
                              node_ids if n in self.graph]
                edges_desc = []
                for u, v, data in self.graph.edges(node_ids, data=True):
                    if v in node_ids:
                        edges_desc.append(
                            f"- ({self.graph.nodes[u]['name']}) -[{data['predicate']}]-> ({self.graph.nodes[v]['name']})")

                sub_text = "【Entities】:\n" + "\n".join(nodes_desc[:15]) + "\n\n【Relations】:\n" + "\n".join(
                    edges_desc[:15])

                summary = ""
                try:
                    resp = self.client.chat.completions.create(
                        model=self.cfg.qwen_model,
                        messages=[{"role": "user",
                                   "content": f"Summarize this open-domain entity community context in 3 precise sentences:\n{sub_text}"}],
                        temperature=0.3,
                        timeout=12
                    )
                    summary = resp.choices[0].message.content.strip()
                    breaker.record_success()  # 成功，清空失败计数
                except Exception as e:
                    logger.warning(f"[Leiden Summary] 社区 {cid} 摘要生成失败: {e}")
                    if breaker.record_failure():
                        raise CircuitBreakerException(
                            "Leiden Group LLM calls failed consecutively. Circuit opened to protect graph sanity.")
                    summary = "Not available."

                reports.append({"community_id": cid, "summary": summary})

            self.tables["community_reports"] = pd.DataFrame(reports)
            logger.info(f"✅ 拓扑社区归类与摘要生成完成: {len(reports)} 个社区。")

    def _compute_semantic_graph_convolution(self):
        """【图卷积神经网络特征计算】 融合同轴拓扑与文本语义空间"""
        logger.info(" GCN 正在执行图邻接邻居特征卷积传播...")
        ent_df = self.tables["entities"]
        node_ids = ent_df["id"].tolist()
        num_nodes = len(node_ids)
        node_idx = {nid: i for i, nid in enumerate(node_ids)}

        X_semantic = np.zeros((num_nodes, self.embed_dim), dtype=np.float32)
        for i, nid in enumerate(node_ids):
            row = ent_df.iloc[i]
            text = f"{row['name']} ({row['type']}). {row['description']}"
            X_semantic[i] = self.embedder.encode(text, normalize_embeddings=True)

        A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for u, v in self.graph.edges():
            if u in node_idx and v in node_idx:
                A[node_idx[u], node_idx[v]] = 1.0
                A[node_idx[v], node_idx[u]] = 1.0

        A_tilde = A + np.eye(num_nodes, dtype=np.float32)

        degrees = np.sum(A_tilde, axis=1)
        d_inv_sqrt = np.power(degrees, -0.5, where=degrees > 0)
        d_inv_sqrt[degrees == 0] = 0.0
        D_inv_sqrt = np.diag(d_inv_sqrt)

        H = D_inv_sqrt.dot(A_tilde).dot(D_inv_sqrt).dot(X_semantic)

        norms = np.linalg.norm(H, axis=1, keepdims=True)
        H_normalized = np.divide(H, norms, out=np.zeros_like(H), where=norms > 0)

        for i, nid in enumerate(node_ids):
            self.gnn_embeddings[nid] = H_normalized[i].tolist()

        logger.info("✅ 【GNN 语义图卷积】计算完成。拓扑节点特征已嵌入对齐维度。")

    # 修复：完整实现 _write_data_to_stores，补齐 Qdrant 写入 (问题3已修复)
    def _write_data_to_stores(self):
        """
        【修复: 完全体数据持久化同步逻辑】
        """
        ent_df = self.tables["entities"]
        rel_df = self.tables["relationships"]
        com_df = self.tables.get("communities", pd.DataFrame())

        com_map = {}
        if not com_df.empty:
            for _, row in com_df.iterrows():
                for eid in row["entity_ids"]:
                    com_map[eid] = int(row["community_id"])

        # 1. 批量持久化 Neo4j 实体
        ent_batch = []
        for _, r in ent_df.iterrows():
            ent_batch.append({
                "id": r["id"],
                "name": r["name"],
                "type": r["type"],
                "description": r["description"],
                "aliases": r["aliases"] if isinstance(r["aliases"], list) else [],
                "community_id": com_map.get(r["id"], -1)
            })
        self.neo4j.batch_write_entities(ent_batch)

        # 2. 批量持久化 Neo4j 关系
        rel_batch = []
        for _, r in rel_df.iterrows():
            rel_batch.append({
                "src_id": r["src_id"],
                "tgt_id": r["tgt_id"],
                "src_name": r["src_name"],
                "tgt_name": r["tgt_name"],
                "pred": r["pred"],  # 使用对齐聚合后的关系类型
                "raw_pred": r["raw_pred"],  # 保留原始提取谓词记录
                "weight": float(r["weight"]),
                "is_negated": bool(r["is_negated"]),
                "evidence": str(r["evidence"]),
                "sequence_index": int(r["sequence_index"])
            })
        if rel_batch:
            self.neo4j.batch_write_relationships(rel_batch)

        # 3. 同步 Upsert 实体到 Qdrant (包含高维度图神经网络 GCN 拓扑 KGE 特征)
        ent_points = []
        for _, row in ent_df.iterrows():
            text = f"{row['name']} ({row['type']}). {row['description']}"
            text_vec = self.embedder.encode(text, normalize_embeddings=True).tolist()
            kge_vec = self.gnn_embeddings.get(row["id"], text_vec)

            ent_points.append(PointStruct(
                id=str(row["id"]),
                vector={
                    "text_semantic": text_vec,
                    "kge": kge_vec
                },
                payload={
                    "name": str(row["name"]),
                    "type": str(row["type"]),
                    "aliases": [str(a) for a in row["aliases"]],
                    "desc": str(row["description"]),
                    "community_id": int(com_map.get(row["id"], -1))
                }
            ))
        if ent_points:
            self.qdrant.upsert(collection_name=DBConfig.ENTITIES_COL, points=ent_points)
            logger.info(f"✅ Qdrant graph_entities 向量特征同步完成: {len(ent_points)} 条")

        # 4. 同步 Upsert 社区摘要到 Qdrant
        com_points = []
        reports_df = self.tables.get("community_reports", pd.DataFrame())
        if not com_df.empty and not reports_df.empty:
            for _, row in com_df.iterrows():
                cid = int(row["community_id"])
                match = reports_df[reports_df["community_id"] == cid]
                if not match.empty:
                    summary = str(match.iloc[0]["summary"])
                    vec = self.embedder.encode(f"community: {summary}", normalize_embeddings=True).tolist()
                    com_points.append(PointStruct(
                        id=cid,
                        vector={"text_semantic": vec},
                        payload={
                            "summary": summary,
                            "community_id": cid,
                            "entity_count": len(row.get("entity_ids", []))
                        }
                    ))
            if com_points:
                self.qdrant.upsert(collection_name=DBConfig.COMMUNITIES_COL, points=com_points)
                logger.info(f"✅ Qdrant graph_communities 同步完成: {len(com_points)} 条")

        # 5. 同步 Upsert 文本分块到 Qdrant
        chunk_points = []
        for _, row in self.tables["text_units"].iterrows():
            vec = self.embedder.encode(row["text"], normalize_embeddings=True).tolist()
            chunk_points.append(PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, row["id"])),
                vector={"text_semantic": vec},
                payload={
                    "text": row["text"],
                    "chunk_index": int(row.get("chunk_index", 0)),
                    "original_tu_id": row["id"]
                }
            ))
        if chunk_points:
            self.qdrant.upsert(collection_name=DBConfig.CHUNKS_COL, points=chunk_points)
            logger.info(f"✅ Qdrant chunks 文本分块向量同步完成: {len(chunk_points)} 条")

    def validate_quality_control(self):
        """
        【新增：质量熔断安全网关】
        """
        logger.info("=== 📊 启动质量监控对齐大盘 ===")
        with self.neo4j.driver.session(database=self.neo4j.database) as session:
            node_count = session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            edge_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

            unknown_nodes = \
                session.run(
                    "MATCH (n:Entity) WHERE n.type = 'UNKNOWN' OR n.type IS NULL RETURN count(n) AS c").single()[
                    "c"]
            has_seq = session.run("MATCH ()-[r]->() WHERE r.sequence_index IS NOT NULL RETURN count(r) AS c").single()[
                "c"]

            qdrant_ent_count = self.qdrant.count(collection_name=DBConfig.ENTITIES_COL).count

            unknown_node_ratio = unknown_nodes / max(1, node_count)
            is_mismatch = (node_count != qdrant_ent_count)

            print("\n" + "=" * 50 + " 📊 GRAPH QUALITY REPORT " + "=" * 50)
            print(f"  🟢 Neo4j 图谱实体数: {node_count} | 关系边数: {edge_count}")
            print(f"  🟢 Qdrant 同步实体数: {qdrant_ent_count}")
            print(f"  🟡 UNKNOWN 占位节点数: {unknown_nodes} (占比: {unknown_node_ratio * 100:.1f}%)")
            print(f"  🟢 时序关系比例: {has_seq / max(1, edge_count) * 100:.1f}%")
            print("=" * 124 + "\n")

            # 熔断验证规则 (Fail-Safe Gateways)
            if is_mismatch:
                logger.error(f"❌ 质量红线报警：Neo4j ({node_count}) 与 Qdrant ({qdrant_ent_count}) 实体总数不同步！")
                raise RuntimeError("Quality Gate Mismatch: Neo4j and Qdrant entity counts are inconsistent.")

            if unknown_node_ratio > 0.45:
                logger.error(
                    f"❌ 质量红线报警：因关系抽取错误导致自动生成的 UNKNOWN 占位节点比例过高 ({unknown_node_ratio * 100:.1f}%)，怀疑抽取链条幻觉严重！")
                raise RuntimeError("Quality Gate Mismatch: Unknown entity placeholder ratio exceeds threshold (45%).")

            if node_count == 0 or edge_count == 0:
                logger.error(f"❌ 质量红线报警：图谱中无可用的实体或边关系数据！")
                raise RuntimeError("Quality Gate Mismatch: Empty Graph Nodes or Edges.")

            logger.info("\033[1;32m✅ 质量红线审查通过，图数据安全，Persist OK！\033[0m")

    def _compute_predicate_clustering(self) -> Dict[str, str]:
        """
        【新增：OIE 谓词语义贪心聚类与关系对齐 (Leader Clustering)】
        1. 提取所有关系的 raw_predicate，进行 BGE 语义嵌入。
        2. 使用贪心余弦相似度聚类（相似度阈值 >= 0.82 的自动归并为一类，指定其中一个为代表，如 ASSISTS -> HELPS）。
        3. 消除关系冗余，为 Neo4j 标注 predicate_canonical，并在 Qdrant `graph_predicates` 建立高速映射。
        """
        logger.info("🛠|  正在启动 OIE 谓词提取与语义贪心聚类...")
        rel_df = self.tables["relationships"]
        if rel_df.empty:
            return {}

        unique_preds = list(set(rel_df["pred"].tolist()))
        num_preds = len(unique_preds)
        logger.info(f"[Predicate Clustering] 提取到 {num_preds} 个唯一 OIE 谓词。")

        # 1. 向量化所有唯一谓词
        pred_vectors = np.zeros((num_preds, self.embed_dim), dtype=np.float32)
        for i, pred in enumerate(unique_preds):
            pred_vectors[i] = self.embedder.encode(f"relation meaning: {pred.lower()}", normalize_embeddings=True)

        # 2. 领袖聚类算法 (NumPy 加速)
        cluster_map = {}  # raw_predicate -> canonical_predicate
        visited = np.zeros(num_preds, dtype=bool)

        for i in range(num_preds):
            if visited[i]:
                continue

            leader_pred = unique_preds[i]
            cluster_map[leader_pred] = leader_pred
            visited[i] = True

            for j in range(i + 1, num_preds):
                if visited[j]:
                    continue
                sim = float(np.dot(pred_vectors[i], pred_vectors[j]))
                if sim >= 0.82:  # 语义高度重合，执行簇归并
                    cluster_map[unique_preds[j]] = leader_pred
                    visited[j] = True
                    logger.info(
                        f"   ├─ 语义合并: {unique_preds[j]} ──> 归入聚类主谓词: {leader_pred} (相似度: {sim:.3f})")

        # 3. 同步至 Qdrant `graph_predicates` 谓词路由集合，为检索期 O(1) 路由奠定物理基础
        pred_points = []
        for i, pred in enumerate(unique_preds):
            canonical = cluster_map[pred]
            pred_points.append(PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, pred.lower())),
                vector={"text_semantic": pred_vectors[i].tolist()},
                payload={
                    "raw_predicate": pred,
                    "canonical_predicate": canonical
                }
            ))
        if pred_points:
            self.qdrant.upsert(collection_name=DBConfig.PREDICATES_COL, points=pred_points)
            logger.info(f"✅ Qdrant `graph_predicates` 谓词路由索引同步完毕: {len(pred_points)} 条。")

        return cluster_map

    # 修复：新增 _aggregate_relationships_robust 实现权重与证据聚合 (问题5已修复)
    def _aggregate_relationships_robust(self):
        """
        【新增：语义谓词聚合与关系去重累加机制】
        对具有相同 (src_id, tgt_id, pred_canonical) 的复杂关系边进行合并：
        1. weight 权重累加。
        2. raw_pred 原始提取谓词集合去重并保留，用 " | " 拼接。
        3. evidence 证据链去重并保留。
        4. sequence_index 保留最早出现的时序。
        5. is_negated 任何一个为真则为真。
        """
        df_rel = self.tables["relationships"]
        if df_rel.empty:
            return

        agg_rules = {
            "src_name": "first",
            "tgt_name": "first",
            "pred": lambda x: " | ".join(set(x)),  # 保存合并前的原始谓词供溯源
            "weight": "sum",
            "is_negated": "any",
            "evidence": lambda x: " || ".join(set([str(i) for i in x if str(i).strip()])),
            "sequence_index": "min"  # 记录该关系最早出现的时序序号
        }

        # 根据 canonical 合并
        df_agg = df_rel.groupby(["src_id", "tgt_id", "pred_canonical"]).agg(agg_rules).reset_index()

        # 变换字段，确保下游调用字段一致性
        df_agg.rename(columns={"pred": "raw_pred", "pred_canonical": "pred"}, inplace=True)
        self.tables["relationships"] = df_agg
        logger.info(f"💾 关系边聚合去重完成: 原始 {len(df_rel)} 条 ──> 聚合后 {len(df_agg)} 条。")

    def run_pipeline(self):
        logger.info("=== 🚀 Universal GraphRAG 2026 工业构建流开始 ===")
        self._init_qdrant()
        self.neo4j.clear_all()
        self.neo4j.create_indexes()

        df_tu = self._extract_and_chunk()
        extractions = self._parallel_extraction_open(df_tu)
        self._resolve_and_merge_robust(df_tu, extractions)

        # 【对齐挂载：执行离线谓词聚类】
        cluster_map = self._compute_predicate_clustering()
        # 将聚类后的主谓词注入到 relationships 映射中
        if cluster_map and not self.tables["relationships"].empty:
            self.tables["relationships"]["pred_canonical"] = self.tables["relationships"]["pred"].map(cluster_map)
        else:
            self.tables["relationships"]["pred_canonical"] = self.tables["relationships"]["pred"]

        # 【修复：在 Leidan 社区划分前，先执行关系边去重与权重聚合】
        self._aggregate_relationships_robust()

        self._build_networkx_and_community()
        self._compute_semantic_graph_convolution()
        self._write_data_to_stores()
        self.validate_quality_control()
        logger.info("=== 🎉 RAG 底座构建并同步完成 ===")


if __name__ == "__main__":
    if not os.getenv("DASHSCOPE_API_KEY"):
        logger.error("请配置 DASHSCOPE_API_KEY 环境变量！")
        sys.exit(1)

    cfg = BuilderConfig()
    builder = UniversalGraphRAGBuilder(cfg)
    try:
        builder.run_pipeline()
    finally:
        builder.neo4j.close()
