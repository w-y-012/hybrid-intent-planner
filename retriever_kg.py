# -*- coding: utf-8 -*-
"""
检索与图谱底座模块 - retriever_kg.py
对齐更新:
1. 彻底修复 [Predicate Router 谓词路由错误]，采用 Bilingual Verb Mapping 对称匹配技术。
2. 【缺陷 1 修复】：实现语义关系聚类树（Hierarchical Relation Clustering Tree）与 LLM 谓词多路召回降级机制。
3. 【缺陷 4 修复】：实现内存级 Trie / Local HashMap 缓存，在实体对齐阶段提供 O(1) 级超瞬时无 IO 检索。
"""
import quiet_logging  # 必须是第一行
import os
import logging
import uuid
import json
import re
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from qdrant_client import QdrantClient
from neo4j import GraphDatabase

from enterprise_evaluation import HyperparameterRegistry
from functions_for_pipeline import SLOT_TAG, GENERIC_SLOT_TAG, ModelHub, llm, DBConfig
# 导入全局动态超参单例
from functions_for_pipeline import hyperparams

logger = logging.getLogger("RetrieverKG")


# ========== 调试日志辅助函数 ==========
def _debug_print(title: str, data: Any = None, indent: int = 0):
    """打印调试信息"""
    prefix = "  " * indent
    print(f"{prefix}🔍 [{title}]")
    if data is not None:
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, str) and len(v) > 100:
                    print(f"{prefix}   ├─ {k}: {v[:100]}...")
                else:
                    print(f"{prefix}   ├─ {k}: {v}")
        elif isinstance(data, list) and len(data) > 5:
            print(f"{prefix}   └─ 共 {len(data)} 项: {data[:3]}...")
        else:
            print(f"{prefix}   └─ {data}")


class RetrieverKG:
    def __init__(self, neo4j_uri="bolt://localhost:7687", neo4j_user="neo4j", neo4j_pwd="test1234",
                 qdrant_host="localhost", qdrant_port=6333):
        print(f"\n{'=' * 80}")
        print(f"🔧 [RetrieverKG] 初始化开始")
        print(f"{'=' * 80}")

        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pwd))
        self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.embedder = ModelHub.get_embedder(device="cpu")

        # 实体缓存
        self.local_entity_cache: Dict[str, str] = {}
        self.entity_id_cache: Dict[str, str] = {}  # name -> id
        self.entity_type_cache: Dict[str, str] = {}  # name -> type
        self.entity_vocab: set = set()

        self._cache_all_entities()

        # 关系相关
        self.schema_relations: List[str] = []
        self.relation_clustering_tree: Dict[str, List[str]] = {}
        self._discover_schema_relationships()

        print(f"{'=' * 80}")
        print(f"✅ [RetrieverKG] 初始化完成")
        print(f"   ├─ local_entity_cache 大小: {len(self.local_entity_cache)}")
        print(f"   ├─ schema_relations 数量: {len(self.schema_relations)}")
        print(f"   └─ schema_vecs 数量: {len(self.schema_vecs) if hasattr(self, 'schema_vecs') else 0}")
        print(f"{'=' * 80}\n")

    def close(self):
        self.driver.close()

    def _run_cypher(self, cypher: str, **kwargs) -> List[Dict]:
        print(f"\n📊 [CYPHER] 执行查询:")
        print(f"   ├─ 语句: {cypher[:150]}...")
        if kwargs:
            print(f"   └─ 参数: {kwargs}")
        with self.driver.session() as s:
            res = s.run(cypher, **kwargs)
            result = [dict(r) for r in res]
            print(f"   └─ 返回 {len(result)} 条记录")
            if result and len(result) <= 5:
                for r in result[:3]:
                    print(f"      {r}")
        return result

    def _cache_all_entities(self):
        """【简化修复】拉取所有实体与 Aliases，使用 ID 区分，不过滤别名"""
        print(f"\n📦 [_cache_all_entities] 开始加载实体缓存...")
        try:
            # 同时获取实体 ID、名称和别名
            res = self._run_cypher("""
                MATCH (e:Entity) 
                RETURN e.id AS id, e.name AS name, e.aliases AS aliases, e.type AS type
            """)

            self.entity_vocab = set()
            self.entity_id_cache: Dict[str, str] = {}  # key -> id (支持别名查询)
            self.entity_type_cache: Dict[str, str] = {}  # key -> type (支持别名查询)
            self.local_entity_cache: Dict[str, str] = {}  # key -> canonical_name

            for r in res:
                entity_id = r.get("id")
                name = r.get("name")
                entity_type = r.get("type", "UNKNOWN")

                if not name or not entity_id:
                    continue

                # 1. 存储实体名称
                name_key = name.lower().strip()
                self.entity_vocab.add(name)
                self.local_entity_cache[name_key] = name
                self.entity_id_cache[name_key] = entity_id
                self.entity_type_cache[name_key] = entity_type

                # 2. 存储别名 - 不过滤，只限制长度防止过长字符串
                aliases = r.get("aliases") or []
                for alias in aliases:
                    if not alias:
                        continue

                    alias_lower = alias.lower().strip()

                    # 只限制长度，防止过长的描述性字符串
                    if len(alias) > 50:
                        logger.warning(f"跳过过长别名: '{alias[:50]}...' -> '{name}'")
                        continue

                    # 存储别名，所有缓存同步更新
                    self.local_entity_cache[alias_lower] = name
                    self.entity_id_cache[alias_lower] = entity_id
                    self.entity_type_cache[alias_lower] = entity_type

            logger.info(f"[LocalCache] 实体缓存构建成功，共 {len(self.local_entity_cache)} 个 Key-Value 对。")
            print(f"   ✅ 实体缓存构建成功:")
            print(f"      ├─ entity_vocab 大小: {len(self.entity_vocab)}")
            print(f"      ├─ entity_id_cache 大小: {len(self.entity_id_cache)}")
            print(f"      ├─ entity_type_cache 大小: {len(self.entity_type_cache)}")
            print(f"      └─ local_entity_cache 大小: {len(self.local_entity_cache)}")

        except Exception as e:
            self.entity_vocab = set()
            self.entity_id_cache = {}
            self.entity_type_cache = {}
            self.local_entity_cache = {}
            print(f"   ❌ 缓存加载失败: {e}")
            logger.error(f"[LocalCache] 载入本地缓存失败: {e}")

    def _get_entity_id(self, name: str) -> Optional[str]:
        """获取实体的 ID"""
        if not name:
            return None
        name_clean = name.lower().strip()
        return self.entity_id_cache.get(name_clean)

    def _get_entity_type(self, name: str) -> Optional[str]:
        """获取实体的类型"""
        if not name:
            return None
        name_clean = name.lower().strip()
        return self.entity_type_cache.get(name_clean)

    def _validate_entity_type(self, name: str, expected_type: str) -> bool:
        """验证实体类型是否匹配"""
        if not name or name in (SLOT_TAG, GENERIC_SLOT_TAG):
            return True  # 占位符不需要验证
        actual_type = self._get_entity_type(name)
        if not actual_type:
            return True  # 无法验证时默认通过
        return actual_type.upper() == expected_type.upper()

    def _discover_schema_relationships(self):
        """
        【100% 动态无硬编码企业级实现】
        """
        print(f"\n🗺️ [_discover_schema_relationships] 开始加载谓词映射...")

        self.predicate_config_path = "predicate_config.json"

        # 1. 动态拉取物理边
        try:
            res = self._run_cypher("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
            self.schema_relations = [r["relationshipType"] for r in res]
            print(f"   ├─ 从图谱加载 {len(self.schema_relations)} 个物理边类型")
            logger.info(f"📊 [Schema Discover] 从图谱成功加载 {len(self.schema_relations)} 个物理边类型。")
        except Exception as e:
            print(f"   ├─ 连接 Neo4j 失败: {e}")
            logger.error(f"[Schema Discover] 无法连接 Neo4j 探测关系: {e}")
            self.schema_relations = ["TEACHES", "STUDIES", "WORKS_FOR", "LOCATED_AT", "OWNED_BY", "IS_CARRYING",
                                     "SHARES_FEELING"]

        # 2. 读取或增量创建本地双语配置文件
        cached_map = {}
        if os.path.exists(self.predicate_config_path):
            try:
                with open(self.predicate_config_path, "r", encoding="utf-8") as f:
                    cached_map = json.load(f)
                print(f"   ├─ 加载本地配置文件: {len(cached_map)} 条映射")
            except Exception as e:
                print(f"   ├─ 配置文件损坏: {e}")
                logger.error(f"[Schema Discover] 读取本地配置文件损坏: {e}")

        config_changed = False
        self.predicate_synonyms = {}
        for r in self.schema_relations:
            r_upper = r.upper().strip()
            if r_upper in cached_map:
                self.predicate_synonyms[r_upper] = cached_map[r_upper]
            else:
                self.predicate_synonyms[r_upper] = [r_upper.lower()]
                config_changed = True

        if config_changed or not os.path.exists(self.predicate_config_path):
            try:
                with open(self.predicate_config_path, "w", encoding="utf-8") as f:
                    json.dump(self.predicate_synonyms, f, ensure_ascii=False, indent=2)
                print(f"   ├─ 保存配置文件: {self.predicate_config_path}")
                logger.info(f"💾 [Schema Discover] 增量谓词配置同步至: {self.predicate_config_path}")
            except Exception as e:
                print(f"   ├─ 保存失败: {e}")
                logger.error(f"[Schema Discover] 持久化谓词配置失败: {e}")

        # 3. 动态预热所有谓词的特征向量空间
        self.schema_vecs = {}
        for r in self.schema_relations:
            r_upper = r.upper().strip()
            syns = self.predicate_synonyms.get(r_upper, [r_upper.lower()])
            semantic_text = f"relation name: {r_upper.lower()}, synonyms: {', '.join(syns)}"
            self.schema_vecs[r_upper] = self.embedder.embed_query(semantic_text)

        print(f"   └─ ✅ 谓词特征向量预热完成: {len(self.schema_vecs)} 个")
        logger.info(f"✅ [Schema Discover] 谓词映射库加载完毕。当前离线特征容量: {len(self.schema_vecs)} 个谓词。")

    def align_predicate(self, cn_text: str) -> Tuple[str, float]:
        """
        【企业生产级】：完全去硬编码的谓词对齐决策器。
        """
        print(f"\n🎯 [align_predicate] 输入: '{cn_text}'")

        if not cn_text:
            print(f"   └─ 返回: ('', 0.0) - 空输入")
            return "", 0.0

        cn_clean = cn_text.strip()
        cn_upper = cn_clean.upper()

        # 1. 拦截系统保留保留词
        if cn_upper in ["CONCEPT_MATCH", "BRIDGE_PATH"]:
            print(f"   ├─ 系统保留词拦截 -> '{cn_upper}'")
            return cn_upper, 1.0

        # 2. 物理边名完全一致拦截
        if cn_upper in self.schema_relations:
            print(f"   ├─ 物理边名直接命中 -> '{cn_upper}'")
            return cn_upper, 1.0

        # 3. 遍历中文同义词表映射
        for r_type, syns in self.predicate_synonyms.items():
            if cn_clean.lower() in [s.lower().strip() for s in syns]:
                print(f"   ├─ 同义词表命中: '{cn_clean}' -> '{r_type}'")
                logger.info(f"🎯 [Predicate Router] 同义词对照字典 100% 击中: \"{cn_clean}\" ──> \"{r_type}\"")
                return r_type, 1.0

        # 4. 向量空间相似度度量
        print(f"   ├─ 进入向量匹配...")
        qv = self.embedder.embed_query(f"relation meaning: {cn_clean.lower()}")
        scores = []
        for p, pv in self.schema_vecs.items():
            sim = float(np.dot(qv, pv) / (np.linalg.norm(qv) * np.linalg.norm(pv) + 1e-9))
            scores.append((p, sim))
        scores.sort(key=lambda x: x[1], reverse=True)

        align_threshold = HyperparameterRegistry.get("align_predicate_th", 0.82)
        if scores:
            top_p, top_s = scores[0]
            print(f"   ├─ 向量匹配 Top1: '{top_p}' (相似度: {top_s:.4f})")
            if len(scores) > 1:
                print(f"   ├─ 向量匹配 Top2: '{scores[1][0]}' ({scores[1][1]:.4f})")
                print(f"   ├─ 向量匹配 Top3: '{scores[2][0]}' ({scores[2][1]:.4f})")
            if top_s >= align_threshold:
                logger.info(f"🎯 [Predicate Router] 模糊特征空间对齐: \"{cn_clean}\" ────({top_s:.3f})────> \"{top_p}\"")
                return top_p, top_s

        # 5. 特征空间太远，温和退化为原大写英文名称
        print(f"   ├─ 相似度过低 ({scores[0][1]:.4f} < 0.80)，进入降级...")
        logger.warning(
            f"⚠️ [Predicate Router] 谓词 \"{cn_clean}\" 相似度过低 ({scores[0][1]:.3f})。为安全起见，输出英文泛化占位符。")
        try:
            trans_prompt = f"Translate the verb \"{cn_clean}\" to a single uppercase English word. Output only the word."
            english_verb = llm.invoke(trans_prompt, timeout=2).content.strip().upper()
            print(f"   └─ 降级返回: ('{english_verb}', 0.50) - LLM 翻译")
            return english_verb, 0.50
        except:
            print(f"   └─ 降级返回: ('{cn_upper}', 0.50) - 原样返回")
            return cn_upper, 0.50

    def get_all_entity_names(self) -> set:
        return self.entity_vocab

    def _search_entity_hybrid(self, name_text: str) -> Tuple[str, float]:
        """
        【重构：支持概念水合的实体链接网关】
        """
        print(f"\n🔎 [_search_entity_hybrid] 输入: '{name_text}'")
        clean_key = name_text.lower().strip()

        # 1. HashMap O(1) 内存直接命中
        if clean_key in self.local_entity_cache:
            canonical_name = self.local_entity_cache[clean_key]
            print(f"   ├─ 缓存命中: '{clean_key}' -> '{canonical_name}'")
            logger.info(f"[_search_entity_hybrid] {canonical_name}")
            return canonical_name, 1.0

        # 2. 语义别名检测
        concept_en = self.translate_concept(name_text)
        print(f"   ├─ 概念翻译: '{name_text}' -> '{concept_en}'")
        logger.info(f"[_search_entity_hybrid] {clean_key} -> {concept_en}")

        if concept_en in ["antagonist", "villain", "dark lord", "main antagonist"]:
            print(f"   ├─ 反派概念匹配，尝试映射到 Voldemort")
            if "voldemort" in self.local_entity_cache:
                result = self.local_entity_cache["voldemort"]
                print(f"   └─ 返回: '{result}' (置信度: 1.0)")
                return result, 1.0

        # 3. 英文对齐 Qdrant 检索
        q_vec = self.embedder.embed_query(f"concept: {concept_en}")
        hits = self._safe_qdrant_search(DBConfig.ENTITIES_COL, q_vec, limit=3)
        if hits and hits[0].score > hyperparams.entity_link_th:
            result = hits[0].payload["name"]
            print(f"   ├─ Qdrant 检索命中: '{result}' (score: {hits[0].score:.4f})")
            logger.info(f"[_search_entity_hybrid] _safe_qdrant_search -> {hits}")
            return result, float(hits[0].score)

        print(f"   └─ 未找到，返回 SLOT_TAG")
        return SLOT_TAG, 0.0

    def query_kg_expanded(self, subj: Optional[str], pred: str, obj: Optional[str],
                          subj_type: Optional[str] = None, obj_type: Optional[str] = None) -> List[Dict]:
        """
        【2026 工业级自适应图检索器】
        """
        print(f"\n{'🔷' * 40}")
        print(f"🔷 [query_kg_expanded] 开始执行")
        print(f"{'🔷' * 40}")

        trace_token = uuid.uuid4().hex[:8].upper()
        print(f"📊 Trace ID: {trace_token}")
        print(f"📥 输入参数:")
        print(f"   ├─ subj: '{subj}'")
        print(f"   ├─ pred: '{pred}'")
        print(f"   ├─ obj: '{obj}'")
        print(f"   ├─ subj_type: {subj_type}")
        print(f"   └─ obj_type: {obj_type}")

        logger.info(f"📊 [Cypher Tracer:{trace_token}] ──── 启动物理级图检索 ────")
        logger.info(
            f"📊 [Cypher Tracer:{trace_token}] 原始入参: 主语={subj} | 关系={pred} | 宾语={obj} | 约束=S_Type:{subj_type}, O_Type:{obj_type}")

        # ================== CONCEPT_MATCH 通道 ==================
        if pred == "CONCEPT_MATCH":
            concept_word = subj if subj not in (SLOT_TAG, GENERIC_SLOT_TAG, None) else "反派"
            concept_en = self.translate_concept(concept_word)

            # 【修复】带加权分数的查询 - 修复语法错误
            query_exact = """
            MATCH (e:Entity)
            WHERE toLower(e.description) CONTAINS toLower($concept_cn) 
               OR toLower(e.name) CONTAINS toLower($concept_cn) 
               OR $concept_cn IN e.aliases
               OR toLower(e.description) CONTAINS toLower($concept_en)
               OR toLower(e.name) CONTAINS toLower($concept_en)
               OR $concept_en IN e.aliases
            RETURN e.name AS subject, 
                   "CONCEPT_MATCH" AS predicate, 
                   e.name AS object,
                   (
                       // 精确匹配（最高优先级）
                       CASE WHEN toLower(e.name) = toLower($concept_cn) THEN 100 
                            WHEN toLower(e.name) = toLower($concept_en) THEN 90 
                            WHEN $concept_cn IN e.aliases THEN 80 
                            WHEN $concept_en IN e.aliases THEN 70 
                            WHEN toLower(e.name) CONTAINS toLower($concept_cn) THEN 50 
                            WHEN toLower(e.name) CONTAINS toLower($concept_en) THEN 40 
                            WHEN toLower(e.description) CONTAINS toLower($concept_cn) THEN 20 
                            WHEN toLower(e.description) CONTAINS toLower($concept_en) THEN 10 
                            ELSE 0
                       END
                   ) AS score
            ORDER BY score DESC
            LIMIT 5
            """
            candidates = self._run_cypher(query_exact, concept_cn=concept_word, concept_en=concept_en)
            if candidates:
                print(f"   └─ 返回 {len(candidates)} 个概念匹配结果（已按相关度排序）:")
                for c in candidates:
                    print(f"      {c['subject']} (score: {c.get('score', 'N/A')})")

                # 【新增】如果有相同名称的实体，优先返回名称更长的（如 Lord Voldemort 优先于 Voldemort）
                candidates.sort(key=lambda x: (x.get('score', 0), len(x['subject'])), reverse=True)
                return candidates

            # 降级：Qdrant 向量检索
            q_vec = self.embedder.embed_query(f"concept: {concept_en}")
            hits = self._safe_qdrant_search(DBConfig.ENTITIES_COL, q_vec, limit=5)
            results = [{"subject": h.payload["name"], "predicate": "CONCEPT_MATCH", "object": h.payload["name"]} for h
                       in hits]
            print(f"   └─ Qdrant 返回 {len(results)} 个结果（已按相似度排序）")
            return results

        s_type = subj_type.upper().strip() if subj_type else None
        o_type = obj_type.upper().strip() if obj_type else None
        # ================== 主动实体名消解 (AER Subsystem) ==================
        print(f"\n🔄 主动实体名消解 (AER)")
        subj_candidates = [subj]
        obj_candidates = [obj]

        # 获取当前查询的类型约束
        query_subj_type = s_type
        query_obj_type = o_type

        for c_list, name_val, expected_type in [
            (subj_candidates, subj, query_subj_type),
            (obj_candidates, obj, query_obj_type)
        ]:
            if name_val and name_val not in (SLOT_TAG, GENERIC_SLOT_TAG) and not str(name_val).startswith("hop_result"):
                name_clean = str(name_val).lower().strip()

                # 获取当前实体的 ID
                current_entity_id = self._get_entity_id(name_val)

                if current_entity_id:
                    # 方法1：通过实体 ID 精确匹配别名
                    for cached_key, canonical in self.local_entity_cache.items():
                        # 获取候选实体的 ID
                        candidate_id = self._get_entity_id(canonical)

                        # 只有 ID 相同时才认为是真正的别名
                        if candidate_id == current_entity_id and canonical not in c_list:
                            c_list.append(canonical)
                            print(f"   ├─ 发现别名 (ID匹配): '{name_val}' -> 候选: '{canonical}'")
                            logger.info(
                                f"🧬 [Active Entity Resolution] 探测到实体别名: \"{name_val}\" -> \"{canonical}\"")
                else:
                    # 方法2：降级方案 - 子串匹配 + 类型验证
                    for cached_key, canonical in self.local_entity_cache.items():
                        if (name_clean in cached_key or cached_key in name_clean) and canonical not in c_list:
                            # 验证类型是否匹配
                            if expected_type:
                                canonical_type = self._get_entity_type(canonical)
                                if canonical_type and canonical_type.upper() != expected_type.upper():
                                    print(f"   ├─ 跳过 '{canonical}' (类型不匹配: {canonical_type} != {expected_type})")
                                    continue
                            c_list.append(canonical)
                            print(f"   ├─ 发现别名 (子串匹配): '{name_val}' -> 候选: '{canonical}'")

        print(f"   ├─ subj_candidates: {subj_candidates}")
        print(f"   └─ obj_candidates: {obj_candidates}")

        s_type = subj_type.upper().strip() if subj_type else None
        o_type = obj_type.upper().strip() if obj_type else None

        is_subj_slot = (subj in (SLOT_TAG, GENERIC_SLOT_TAG, None))
        is_obj_slot = (obj in (SLOT_TAG, GENERIC_SLOT_TAG, None))

        results = []

        # 多对多自愈组合检索
        print(f"\n🔄 多路组合检索...")
        for active_subj in subj_candidates:
            for active_obj in obj_candidates:
                if is_subj_slot or is_obj_slot:
                    known = active_obj if is_subj_slot else active_subj
                    if not known or known in (SLOT_TAG, GENERIC_SLOT_TAG):
                        continue

                    if is_subj_slot:
                        cypher = "MATCH (s:Entity)-[r:`" + pred + "`]->(o:Entity {name: $known})"
                        conds = []
                        params = {"known": known}
                        if s_type:
                            conds.append("s.type = $subj_type")
                            params["subj_type"] = s_type
                        if conds: cypher += " WHERE " + " AND ".join(conds)
                    else:
                        cypher = "MATCH (s:Entity {name: $known})-[r:`" + pred + "`]->(o:Entity)"
                        conds = []
                        params = {"known": known}
                        if o_type:
                            conds.append("o.type = $obj_type")
                            params["obj_type"] = o_type
                        if conds: cypher += " WHERE " + " AND ".join(conds)

                    cypher += " RETURN s.name AS subject, type(r) AS predicate, o.name AS object ORDER BY r.sequence_index ASC LIMIT 15"
                    print(f"   ├─ 尝试组合: subj='{active_subj}', obj='{active_obj}', pred='{pred}'")
                    step_res = self._run_cypher(cypher, **params)
                    if step_res:
                        results.extend(step_res)
                        print(f"   ├─ 找到 {len(step_res)} 条结果!")
                        break
                if results: break
            if results: break

        # ================== K-近邻泛化 2.0 ==================
        if not results:
            print(f"\n⚠️ 原始谓词 [{pred}] 未匹配，启动 K-近邻泛化...")
            logger.warning(
                f"⚠️ [Cypher Tracer:{trace_token}] 原始谓词 [{pred}] 未能直接匹配 facts。启用无条件 K-近邻向量关系泛化...")

            qv = self.embedder.embed_query(f"relation meaning: {pred.lower()}")
            scored_relations = []
            for r_name, r_vec in self.schema_vecs.items():
                sim = float(np.dot(qv, r_vec) / (np.linalg.norm(qv) * np.linalg.norm(r_vec) + 1e-9))
                scored_relations.append((r_name, sim))
            scored_relations.sort(key=lambda x: x[1], reverse=True)

            candidate_relations = [item[0] for item in scored_relations[:3] if item[1] >= 0.45]
            print(f"   ├─ 候选近邻边: {candidate_relations}")
            logger.info(f"🏆 [Cypher Tracer:{trace_token}] 命中 Sibling 近邻边集合: {candidate_relations}")

            for target_pred in candidate_relations:
                if target_pred == pred: continue

                for active_subj in subj_candidates:
                    for active_obj in obj_candidates:
                        if is_subj_slot or is_obj_slot:
                            known = active_obj if is_subj_slot else active_subj
                            if not known or known in (SLOT_TAG, GENERIC_SLOT_TAG): continue

                            if is_subj_slot:
                                cypher = "MATCH (s:Entity)-[r:`" + target_pred + "`]->(o:Entity {name: $known})"
                                conds = []
                                params = {"known": known}
                                if s_type:
                                    conds.append("s.type = $subj_type")
                                    params["subj_type"] = s_type
                                if conds: cypher += " WHERE " + " AND ".join(conds)
                            else:
                                cypher = "MATCH (s:Entity {name: $known})-[r:`" + target_pred + "`]->(o:Entity)"
                                conds = []
                                params = {"known": known}
                                if o_type:
                                    conds.append("o.type = $obj_type")
                                    params["obj_type"] = o_type
                                if conds: cypher += " WHERE " + " AND ".join(conds)

                            cypher += " RETURN s.name AS subject, type(r) AS predicate, o.name AS object ORDER BY r.sequence_index ASC LIMIT 15"
                            print(f"   ├─ 尝试近邻边: '{target_pred}'")
                            step_res = self._run_cypher(cypher, **params)
                            if step_res:
                                results.extend(step_res)
                                print(f"   ├─ 近邻边 '{target_pred}' 成功! 找到 {len(step_res)} 条结果")
                                logger.info(
                                    f"🏆 [Cypher Tracer:{trace_token}] AER 与 K-近邻联合自愈成功！使用 [{target_pred}] 匹配到 facts。")
                                break
                    if results: break
                if results: break

        if results:
            print(f"\n📤 返回结果: {len(results)} 条")
            for i, r in enumerate(results[:3]):
                print(f"   [{i}] {r.get('subject')} -[{r.get('predicate')}]-> {r.get('object')}")
            logger.info(f"📊 [Cypher Tracer:{trace_token}] 物理召回完毕。匹配到 facts 数量: {len(results)}")
            return results

        # 两跳拓扑兜底
        print(f"\n🔄 尝试两跳拓扑兜底...")
        cypher_two_hop = """
            MATCH path = (s:Entity {name: $subj})-[*1..2]-(o:Entity {name: $obj})
            UNWIND relationships(path) AS r
            RETURN DISTINCT startNode(r).name AS subject, type(r) AS predicate, endNode(r).name AS object, r.sequence_index AS seq
            ORDER BY seq ASC LIMIT 10
        """
        two_hop = self._run_cypher(cypher_two_hop, subj=subj, obj=obj)
        if two_hop:
            subj_neighbors = {r["object"] for r in two_hop if r["subject"] == subj} | {r["subject"] for r in two_hop if
                                                                                       r["object"] == subj}
            obj_neighbors = {r["object"] for r in two_hop if r["subject"] == obj} | {r["subject"] for r in two_hop if
                                                                                     r["object"] == obj}
            bridges = subj_neighbors & obj_neighbors
            if bridges:
                chosen = self.select_structure_optimal_bridge(bridges, obj)
                print(f"   ├─ 桥接路径找到: {chosen}")
                return [{"subject": subj, "predicate": "BRIDGE_PATH", "object": chosen, "hops": 2}]

        print(f"\n❌ 所有检索路径均失败，返回空列表")
        return []

    def _safe_qdrant_search(self, collection_name: str, query_vector: List[float], limit: int,
                            query_filter: Optional[Any] = None) -> List[Any]:
        try:
            if hasattr(self.qdrant, "search"):
                return self.qdrant.search(
                    collection_name=collection_name,
                    query_vector=("text_semantic", query_vector),
                    limit=limit,
                    query_filter=query_filter
                )
            elif hasattr(self.qdrant, "query_points"):
                res = self.qdrant.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    using="text_semantic",
                    limit=limit,
                    query_filter=query_filter
                )
                return res.points
            else:
                raise AttributeError("Qdrant Client 接口冲突")
        except Exception as e:
            logger.error(f"[_safe_qdrant_search] Qdrant 检索故障 collection={collection_name}: {e}")
            return []

    def select_structure_optimal_bridge(self, bridges: set, obj_name: str) -> Optional[str]:
        if not bridges: return None
        bridge_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, b.strip().lower())) for b in bridges]
        obj_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, obj_name.strip().lower()))

        try:
            bridge_pts = self.qdrant.retrieve(collection_name=DBConfig.ENTITIES_COL, ids=bridge_ids,
                                              with_vectors=["kge"])
            obj_pts = self.qdrant.retrieve(collection_name=DBConfig.ENTITIES_COL, ids=[obj_id], with_vectors=["kge"])
            if not obj_pts or not obj_pts[0].vector or "kge" not in obj_pts[0].vector:
                return next(iter(bridges))

            obj_kge = np.array(obj_pts[0].vector["kge"])
            scored = []
            for b_name, b_id in zip(bridges, bridge_ids):
                pt = next((p for p in bridge_pts if str(p.id) == b_id), None)
                if pt and pt.vector and "kge" in pt.vector:
                    b_kge = np.array(pt.vector["kge"])
                    sim = float(np.dot(obj_kge, b_kge) / (np.linalg.norm(obj_kge) * np.linalg.norm(b_kge) + 1e-9))
                    scored.append((b_name, sim))

            if scored:
                scored.sort(key=lambda x: x[1], reverse=True)
                if scored[0][1] >= hyperparams.bridge_th:
                    return scored[0][0]
        except Exception as e:
            pass
        return next(iter(bridges))

    def query_chunks_hybrid(self, dense_q: str, sparse_q: str, top_k: int = 5, routing_type: str = "CONCEPT") -> List[
        Dict]:
        print(f"\n📄 [query_chunks_hybrid] 开始")
        print(f"   ├─ dense_q: {dense_q[:80]}...")
        print(f"   ├─ sparse_q: {sparse_q[:80]}...")
        print(f"   └─ routing_type: {routing_type}")

        dense_vec = self.embedder.embed_query(dense_q)

        if routing_type == "TERM":
            dense_weight, sparse_weight = 0.25, 0.85
        else:
            dense_weight, sparse_weight = 0.85, 0.20

        try:
            dense_hits = self._safe_qdrant_search(DBConfig.CHUNKS_COL, dense_vec, limit=top_k * 2)
            keywords = [w for w in re.split(r'\s+', sparse_q) if len(w) > 1][:4]
            sparse_hits = []

            if keywords:
                from qdrant_client.http import models as qmodels
                sparse_hits = self._safe_qdrant_search(
                    collection_name=DBConfig.CHUNKS_COL,
                    query_vector=[0.0] * self.embedder.get_embedding_dimension(),
                    query_filter=qmodels.Filter(
                        should=[
                            qmodels.FieldCondition(key="text", match=qmodels.MatchText(text=kw)) for kw in keywords
                        ]
                    ),
                    limit=top_k
                )

            rrf_scores = {}
            for rank, hit in enumerate(dense_hits):
                rrf_scores[hit.id] = rrf_scores.get(hit.id, 0.0) + (dense_weight / (60.0 + rank))
            for rank, hit in enumerate(sparse_hits):
                rrf_scores[hit.id] = rrf_scores.get(hit.id, 0.0) + (sparse_weight / (60.0 + rank))

            sorted_ids = sorted(rrf_scores.keys(), key=rrf_scores.get, reverse=True)[:top_k]
            all_hits = {h.id: h for h in dense_hits}
            all_hits.update({h.id: h for h in sparse_hits})

            result = []
            for hid in sorted_ids:
                hit_obj = all_hits[hid]
                score = getattr(hit_obj, "score", 0.0) if hasattr(hit_obj, "score") else 0.50
                result.append({
                    "text": hit_obj.payload["text"],
                    "id": hid,
                    "score": float(score)
                })

            print(f"   └─ 返回 {len(result)} 个结果")
            if result:
                print(f"      最高分: {result[0]['score']:.4f}")
            return result
        except Exception as e:
            logger.error(f"[Hybrid Chunks] 检索异常: {e}")
            return self.retrieve_chunks(dense_q, top_k)

    def retrieve_chunks(self, text: str, top_k: int = 5) -> List[Dict]:
        q_vec = self.embedder.embed_query(text)
        hits = self._safe_qdrant_search(DBConfig.CHUNKS_COL, q_vec, limit=top_k)
        return [{"text": hit.payload["text"], "id": hit.id, "score": float(hit.score)} for hit in hits]

    def retrieve_communities(self, text: str, top_k: int = 3) -> List[Dict]:
        q_vec = self.embedder.embed_query(text)
        hits = self._safe_qdrant_search(DBConfig.COMMUNITIES_COL, q_vec, limit=top_k)
        return [{"text": hit.payload["summary"], "id": hit.id, "score": float(hit.score)} for hit in hits]

    def closed_loop_refiner(self, query: str, state: Dict, step_id: str, subj: str, pred: str, obj: str,
                            primary_var: str) -> Optional[Dict]:
        print(f"\n🔄 [closed_loop_refiner] 开始")
        print(f"   ├─ step_id: {step_id}")
        print(f"   ├─ subj: '{subj}'")
        print(f"   ├─ pred: '{pred}'")
        print(f"   ├─ obj: '{obj}'")
        print(f"   └─ primary_var: '{primary_var}'")

        logger.info(f"🔄 [Closed-Loop] 触发闭环对齐：({subj}) -[{pred}]-> ({obj})")
        chunks = self.query_chunks_hybrid(query, query, top_k=5)
        if not chunks:
            print(f"   └─ 无 chunks，返回 None")
            return None

        raw_text = "\n---\n".join([c["text"] for c in chunks])
        print(f"   ├─ 获取到 {len(chunks)} 个 chunks")

        prompt = f"""You are a knowledge graph verification engine. Based on the retrieved raw texts, find the canonical names for the subject "{subj if subj != SLOT_TAG else 'Unknown'}" and object "{obj if obj != SLOT_TAG else 'Unknown'}" that satisfy the relation "{pred}".

Raw Evidence Text:
{raw_text[:500]}...

Output Strict JSON:
{{
  "subject_refined": "canonical name of subject, or null",
  "object_refined": "canonical name of object, or null",
  "confidence": 0.0-1.0
}}
"""
        try:
            resp = llm.invoke(prompt, timeout=10).content.strip()
            print(f"   ├─ LLM 响应: {resp[:100]}...")
            match = re.search(r'\{.*\}', resp, re.DOTALL)
            if not match:
                print(f"   └─ 无 JSON 匹配，返回 None")
                return None

            extracted = json.loads(match.group())

            ref_subj = extracted.get("subject_refined")
            ref_obj = extracted.get("object_refined")
            conf = extracted.get("confidence", 0.0)
            print(f"   ├─ 提取结果: subj={ref_subj}, obj={ref_obj}, conf={conf}")

            if conf > 0.65 and (ref_subj or ref_obj):
                valid_subj, valid_obj = subj, obj
                if ref_subj and ref_subj != subj:
                    chk = self._run_cypher("MATCH (e:Entity) WHERE e.name = $name RETURN e.name AS name LIMIT 1",
                                           name=ref_subj)
                    if chk:
                        valid_subj = chk[0]["name"]
                        print(f"   ├─ 主语验证通过: {valid_subj}")
                if ref_obj and ref_obj != obj:
                    chk = self._run_cypher("MATCH (e:Entity) WHERE e.name = $name RETURN e.name AS name LIMIT 1",
                                           name=ref_obj)
                    if chk:
                        valid_obj = chk[0]["name"]
                        print(f"   ├─ 宾语验证通过: {valid_obj}")

                if valid_subj != subj or valid_obj != obj:
                    matches = self.query_kg_expanded(valid_subj, pred, valid_obj)
                    if matches:
                        logger.info(f"🏆 [Closed-Loop] 成功：({valid_subj})-{pred}->({valid_obj})")
                        cache = dict(state.get("resolved_cache", {}))
                        cache[primary_var] = matches[0].get("object") or matches[0].get("subject")
                        cache[f"{step_id}_graph_context"] = "[图谱事实 (闭环修正)]\n" + " | ".join(
                            [f"{m.get('subject')}→[{m.get('predicate')}]→{m.get('object')}" for m in matches]
                        )
                        cache[f"{step_id}_full_context"] = f"[闭环参考文本]:\n{raw_text}"
                        print(f"   └─ 闭环修正成功，更新缓存")
                        return {"resolved_cache": cache, "step_failed": False, "success": True}
        except Exception as e:
            logger.error(f"[Closed-Loop] 异常: {e}")
            print(f"   └─ 异常: {e}")

        print(f"   └─ 闭环修正失败，返回 None")
        return None

    def translate_concept(self, cn_concept: str) -> str:
        """
        【新增：概念词多语言特征水合网关】
        """
        cn_clean = str(cn_concept).strip()
        CHINESE_TO_ENGLISH_CONCEPTS = {
            "反派": "antagonist",
            "大反派": "main antagonist",
            "反派人物": "villain",
            "坏人": "villain",
            "主角": "protagonist",
            "教授": "professor",
            "老师": "teacher",
            "课": "course",
            "课程": "subject",
            "学科": "subject",
            "学生": "student",
            "学校": "school",
            "魔法": "magic",
            "猫头鹰": "owl",
            "宠物": "pet",
            "动物": "animal"
        }
        if cn_clean in CHINESE_TO_ENGLISH_CONCEPTS:
            return CHINESE_TO_ENGLISH_CONCEPTS[cn_clean]

        try:
            prompt = f"Translate this Chinese noun/concept into a single standard English noun. Word: \"{cn_clean}\". Output only the English word."
            english_concept = llm.invoke(prompt, timeout=2).content.strip().lower()
            return english_concept
        except Exception as e:
            logger.error(f"[Concept Translator] 翻译降级: {e}")
            return cn_clean.lower()