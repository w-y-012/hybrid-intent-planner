# -*- coding: utf-8 -*-
"""
intent_analyzer.py - 修复版：_is_interrogative_tree 容错 + 修饰事件 arg0 补全
"""
import stanza
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set

logger = logging.getLogger("IntentAnalyzer")

SLOT_TAG = "__SLOT__"


# =========================================================================
# 🧬 唯一实体提取引擎
# =========================================================================
def extract_entities_from_text(text: str, vocab: Set[str]) -> List[Dict[str, Any]]:
    if not text or not vocab:
        return []
    n = len(text)
    candidates = []
    for i in range(n):
        for j in range(i + 2, min(n + 1, i + 25)):
            sub = text[i:j]
            if sub in vocab:
                candidates.append({"text": sub, "start": i, "end": j, "pos": "PROPN"})
    candidates.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
    non_overlapping = []
    for cand in candidates:
        overlap = any(
            not (cand["end"] <= s["start"] or cand["start"] >= s["end"])
            for s in non_overlapping
        )
        if not overlap:
            non_overlapping.append(cand)
    return non_overlapping


# =========================================================================
# 🛡️ 零硬编码疑问判定 — 修复版
# =========================================================================
def _is_interrogative_tree(tok: Dict, tokens: List[Dict], visited: Optional[Set[int]] = None) -> bool:
    """
    递归检查 Token 及其子树是否具有疑问特征。

    【修复】：
    1. 基础 PRON 容错 — 即使 feats 缺失，pos=PRON 也判定为疑问
    2. 递归追踪子节点 — 解决"教哪门课"中"课"→"哪"的追踪
    """
    if visited is None:
        visited = set()
    if tok["id"] in visited:
        return False
    visited.add(tok["id"])

    feats = tok.get("feats", "")
    pos = tok.get("pos", "")
    deprel = tok.get("deprel", "")
    text = tok.get("text", "")

    # 1. 标准 UD 特征：PronType=Int
    if "PronType=Int" in feats or "prontype=int" in feats.lower():
        return True

    # 2. 【修复】基础容错：pos=PRON（代词）→ 极大概率为疑问词
    if pos == "PRON":
        return True

    # 3. 【修复】det/nummod 且 feats 包含 Int
    if pos in ("DET", "NUM") and "Int" in feats:
        return True

    # 4. 递归检查子树
    children = [t for t in tokens if t["head"] == tok["id"]]
    for child in children:
        if _is_interrogative_tree(child, tokens, visited):
            return True

    return False


# =========================================================================
# Schema 类型解析器
# =========================================================================
class SchemaTypeResolver:
    """在线向量空间对齐的 Schema 类型解析器"""

    def __init__(self, ret_if):
        self.ret_if = ret_if
        # 保持默认值，不要被 'Entity' 覆盖
        self.node_labels = ["CONCEPT", "PERSON", "LOCATION", "OBJECT", "ORGANIZATION"]
        self._label_vec_cache = {}
        # 尝试获取 KG 真实标签作为补充
        try:
            labels_data = self.ret_if._run_cypher("CALL db.labels()")
            if labels_data:
                extra = [str(item["label"]).upper() for item in labels_data]
                # 只添加非 Entity 的有意义标签
                for label in extra:
                    if label not in self.node_labels and label != "ENTITY":
                        self.node_labels.append(label)
        except Exception:
            pass

    def resolve_expected_type(self, predicate: str, slot_role: str, slot_word: str, tokens: List[Dict]) -> str:
        """根据 KG Schema 和槽位上下文动态消解类型"""
        aligned_pred, _ = self.ret_if.align_predicate(predicate)
        if not aligned_pred:
            aligned_pred = "RELATED_TO"

        # 从 KG Schema 获取候选标签
        candidate_labels = set()
        try:
            schema_query = f"""
            MATCH (s)-[r:`{aligned_pred}`]->(o)
            RETURN DISTINCT labels(o) AS o_labels LIMIT 10
            """
            records = self.ret_if._run_cypher(schema_query)
            for rec in records:
                for label in rec.get("o_labels", []):
                    label_str = str(label).upper()
                    # 【修复】过滤掉无意义的 ENTITY 标签
                    if label_str and label_str != "ENTITY":
                        candidate_labels.add(label_str)
        except Exception:
            pass

        if not candidate_labels:
            candidate_labels = set(label.upper() for label in self.node_labels if label.upper() != "ENTITY")

        # 向量对齐
        best_label = "CONCEPT"
        max_sim = -1.0
        try:
            head_vec = self.ret_if.embedder.embed_query(slot_word)
            for label in candidate_labels:
                if label not in self._label_vec_cache:
                    self._label_vec_cache[label] = self.ret_if.embedder.embed_query(label.lower())
                label_vec = self._label_vec_cache[label]
                sim = float(np.dot(head_vec, label_vec) / (np.linalg.norm(head_vec) * np.linalg.norm(label_vec) + 1e-9))
                if sim > max_sim:
                    max_sim = sim
                    best_label = label
        except Exception:
            best_label = list(candidate_labels)[0] if candidate_labels else "CONCEPT"

        logger.info(
            f"🧬 [Schema Type Resolver] slot_word='{slot_word}' → best_label='{best_label}' (sim={max_sim:.3f}), candidates={candidate_labels}")
        return best_label

# =========================================================================
# 1) 依存句法提取器
# =========================================================================
class StructureExtractorZH:
    """通用依存句法抽取器"""

    def __init__(self, vocab_set: Optional[Set[str]] = None):
        self.nlp = stanza.Pipeline(
            lang="zh",
            processors="tokenize,lemma,pos,depparse",
            tokenize_no_ssplit=True,
            download_method=None,
            model_dir=r"C:\Users\w_y_0\AppData\Local\StanfordNLP\stanza\Cache\1.12.0\resources"
        )
        self.kg_vocab = vocab_set or set()

    def extract(self, text: str) -> Dict[str, Any]:
        doc = self.nlp(text)
        toks = []
        curr_offset = 0
        for sent in doc.sentences:
            for w in sent.words:
                start_idx = text.find(w.text, curr_offset)
                if start_idx == -1:
                    start_idx = curr_offset
                end_idx = start_idx + len(w.text)
                curr_offset = end_idx
                head_id = w.head if isinstance(w.head, int) else (w.head.id if w.head else 0)
                toks.append({
                    "id": int(w.id),
                    "text": w.text,
                    "pos": str(w.pos),
                    "deprel": str(w.deprel).lower(),
                    "head": int(head_id) if head_id else 0,
                    "start": start_idx,
                    "end": end_idx,
                    "feats": str(w.feats) if w.feats else ""
                })

        toks = self._repair_tokenization(text, toks)

        pred_toks = [t for t in toks if str(t["pos"]).startswith("V") or t["deprel"] == "root"]
        seen = set()
        pred_toks_unique = [p for p in pred_toks if not (p["id"] in seen or seen.add(p["id"]))]
        root_pred = next((p for p in pred_toks_unique if p["deprel"] == "root"), None) or (
            pred_toks_unique[0] if pred_toks_unique else None)

        events = []
        for p in pred_toks_unique:
            pid = p["id"]
            children = [t for t in toks if t["head"] == pid]
            is_passive = any("pass" in c["deprel"] or c["deprel"] == "aux:pass" for c in children)

            subj, obj, agent = None, None, None
            for c in children:
                dep = c["deprel"]
                if dep in ("nsubj", "nsubjpass", "nsubj:pass"):
                    subj = c
                elif dep in ("obj", "dobj", "iobj", "pobj"):
                    obj = c
                elif dep in ("obl", "agent") or (is_passive and dep == "obl:agent"):
                    agent = c

            if is_passive:
                arg0 = agent["text"] if agent else None
                arg1 = subj["text"] if subj else None
            else:
                arg0 = subj["text"] if subj else None
                arg1 = obj["text"] if obj else None

            is_negated = any(c["deprel"] == "neg" for c in children)

            events.append({
                "predicate_text": p["text"],
                "predicate_token_id": pid,
                "arg0": arg0,
                "arg1": arg1,
                "is_passive": is_passive,
                "is_negated": is_negated,
                "is_main_predicate": (p["id"] == root_pred["id"]) if root_pred else False,
                "is_modifier_constraint": False
            })

        self._mark_modifier_constraints(toks, events)
        return {"events": events, "tokens": toks}

    def _repair_tokenization(self, text: str, toks: List[Dict]) -> List[Dict]:
        entities = extract_entities_from_text(text, self.kg_vocab)
        if not entities:
            return toks
        new_toks = []
        idx = 0
        n_toks = len(toks)
        while idx < n_toks:
            t = toks[idx]
            matched_entity = None
            for ent in entities:
                if t["start"] >= ent["start"] and t["end"] <= ent["end"]:
                    matched_entity = ent
                    break
            if matched_entity:
                while idx < n_toks and toks[idx]["start"] >= matched_entity["start"] and toks[idx]["end"] <= \
                        matched_entity["end"]:
                    idx += 1
                new_toks.append({
                    "id": len(new_toks) + 1, "text": matched_entity["text"],
                    "pos": "PROPN", "deprel": "nn", "head": 0,
                    "start": matched_entity["start"], "end": matched_entity["end"], "feats": ""
                })
            else:
                new_toks.append(t)
                idx += 1
        for i, nt in enumerate(new_toks):
            nt["id"] = i + 1
        return new_toks

    def _mark_modifier_constraints(self, toks: List[Dict], events: List[Dict]):
        for ev in events:
            pid = ev["predicate_token_id"]
            pred_tok = next((t for t in toks if t["id"] == pid), None)
            if pred_tok and pred_tok.get("deprel") in ("acl", "acl:relcl"):
                head_id = pred_tok.get("head")
                head_tok = next((t for t in toks if t["id"] == head_id), None)
                if head_tok and head_tok.get("pos") in ("NOUN", "PROPN"):
                    ev["modifies_entity"] = head_tok["text"]
                    ev["is_modifier_constraint"] = True


# =========================================================================
# 2) 论元自治愈引擎 — 修复版
# =========================================================================
class ArgumentCompositor:
    """零冗余论元自愈引擎"""

    def __init__(self, vocab: Set[str], schema_resolver: SchemaTypeResolver):
        self.vocab = vocab
        self.schema_resolver = schema_resolver

    def repair_event(self, event: Dict[str, Any], tokens: List[Dict], raw_text: str,
                     other_event: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        零冗余论元自愈引擎。

        处理流程：
        1. 被动语态施受交换
        2. 论元补全（实体列表 → other_event 借用 → 右侧名词）
        3. 槽位识别与类型消解
        4. 兜底：普通名词保留原值，不误设为 SLOT_TAG
        """
        repaired = event.copy()
        arg0 = repaired.get("arg0")
        arg1 = repaired.get("arg1")
        is_passive = repaired.get("is_passive", False)

        # ================================================================
        # 1. 被动语态交换
        # ================================================================
        if is_passive and arg0 and arg1:
            repaired["arg0"], repaired["arg1"] = arg1, arg0
            repaired["is_passive"] = False
            arg0, arg1 = repaired["arg0"], repaired["arg1"]

        # ================================================================
        # 2. 提取实体
        # ================================================================
        entities = extract_entities_from_text(raw_text, self.vocab)
        entity_names = set(e["text"] for e in entities)

        def get_tok(txt):
            if not txt: return None
            return next((t for t in tokens if t["text"] == str(txt)), None)

        def check_is_entity(val) -> bool:
            """判定 val 是否是 KG 中存在的命名实体"""
            if not val or str(val).strip() in ("", "None", SLOT_TAG):
                return False
            val_str = str(val).strip()
            tok = get_tok(val_str)
            if tok and _is_interrogative_tree(tok, tokens):
                return False
            return (val_str in self.vocab) or (val_str in entity_names)

        def check_is_slot(val) -> bool:
            """判定 val 是否是疑问槽位"""
            if not val or str(val).strip() in ("", "None", SLOT_TAG):
                return False
            if check_is_entity(val):
                return False
            tok = get_tok(val)
            if tok is None:
                return False
            return _is_interrogative_tree(tok, tokens)

        # ================================================================
        # 3. 论元补全（优先级链）
        # ================================================================
        arg0_val = repaired.get("arg0")

        if not arg0_val or str(arg0_val).strip() in ("", "None"):
            found = False
            # 优先级 1：从实体列表
            for ent in entity_names:
                if ent != repaired.get("arg1"):
                    repaired["arg0"] = ent
                    found = True
                    logger.info(f"🧬 [Arg Repair] arg0 从实体列表补全: '{ent}'")
                    break
            # 优先级 2：从 other_event 的 arg0 借用（仅当不是 SLOT_TAG）
            if not found and other_event:
                other_arg0 = other_event.get("arg0")
                if other_arg0 and other_arg0 != SLOT_TAG and str(other_arg0).strip() not in ("", "None"):
                    repaired["arg0"] = other_arg0
                    entity_names.add(str(other_arg0))
                    found = True
                    logger.info(f"🧬 [Arg Repair] arg0 从 other_event 借用: '{other_arg0}'")

        # arg1 补全
        if not repaired.get("arg1") or str(repaired.get("arg1")).strip() in ("", "None"):
            pid = repaired.get("predicate_token_id", 0)
            right_nouns = [t for t in tokens if t["id"] > pid and t["pos"] in ("NOUN", "PROPN")]
            for t in right_nouns:
                txt = t["text"]
                if txt != repaired.get("arg0") and not check_is_slot(txt):
                    repaired["arg1"] = txt
                    logger.info(f"🧬 [Arg Repair] arg1 从右侧名词补全: '{txt}'")
                    break

        # ================================================================
        # 4. 【修复】粘连疑问谓词处理（如 "教哪" 中包含 "哪"）
        # ================================================================
        pred_text = repaired.get("predicate_text", "")
        if pred_text and any(q in pred_text for q in ["哪", "谁", "什么", "几", "何"]):
            # 谓词本身含疑问字 → arg1 大概率是槽位
            if repaired.get("arg1") and not check_is_entity(repaired.get("arg1")):
                logger.info(f"🧬 [Arg Repair] 粘连疑问谓词 '{pred_text}' → arg1 强制设为 SLOT")
                repaired["arg1"] = SLOT_TAG

        # ================================================================
        # 5. 重新评估并执行槽位替换
        # ================================================================
        arg0_final = repaired.get("arg0")
        arg1_final = repaired.get("arg1")

        arg0_is_entity = check_is_entity(arg0_final)
        arg1_is_entity = check_is_entity(arg1_final)
        arg0_is_slot = check_is_slot(arg0_final)
        arg1_is_slot = check_is_slot(arg1_final)

        slot_role = None
        slot_type = "CONCEPT"

        if arg0_is_entity and arg1_is_slot:
            raw_slot_word = repaired["arg1"]
            repaired["arg1"] = SLOT_TAG
            slot_role = "object"
            slot_type = self.schema_resolver.resolve_expected_type(
                predicate=repaired.get("predicate_text", ""),
                slot_role=slot_role,
                slot_word=str(raw_slot_word),
                tokens=tokens
            )
        elif arg0_is_slot and arg1_is_entity:
            raw_slot_word = repaired["arg0"]
            repaired["arg0"] = SLOT_TAG
            slot_role = "subject"
            slot_type = self.schema_resolver.resolve_expected_type(
                predicate=repaired.get("predicate_text", ""),
                slot_role=slot_role,
                slot_word=str(raw_slot_word),
                tokens=tokens
            )
        elif arg0_is_entity and arg1_is_entity:
            # 两个都是实体 → 验证性查询
            slot_role = None
        else:
            # ============================================================
            # 【核心修复】兜底：普通名词保留原值，不强制设为 SLOT_TAG
            # ============================================================
            if repaired.get("arg0") and not arg0_is_entity:
                if arg0_is_slot:
                    repaired["arg0"] = SLOT_TAG
                    slot_role = "subject"
                # else: 保留原值（如 "教授"），供 compile 中 CONCEPT_MATCH 使用

            if repaired.get("arg1") and not arg1_is_entity:
                if arg1_is_slot:
                    repaired["arg1"] = SLOT_TAG
                    if slot_role is None:
                        slot_role = "object"
                # else: 保留原值（如 "反派"），供 compile 中 CONCEPT_MATCH 使用

        repaired["slot_role"] = slot_role
        repaired["slot_type"] = slot_type

        logger.info(f"🧬 [Arg Repair] 修复完成: arg0='{repaired.get('arg0')}', arg1='{repaired.get('arg1')}', "
                    f"slot_role={slot_role}, slot_type={slot_type}")
        return repaired

# =========================================================================
# 3) 认知编译器
# =========================================================================
class IntentInterpreterCompiler:
    """企业级自适应认知编译器"""

    def __init__(self, retriever_interface):
        self.ret_if = retriever_interface
        self.vocab = set(self.ret_if.get_all_entity_names())
        self.struct = StructureExtractorZH(self.vocab)
        self.schema_resolver = SchemaTypeResolver(self.ret_if)
        self.arg_repair = ArgumentCompositor(self.vocab, self.schema_resolver)

    def _pick_event(self, q_or_evs: Any, evs_or_q: Any = None) -> Optional[Dict[str, Any]]:
        events = q_or_evs if isinstance(q_or_evs, list) else evs_or_q
        if not events:
            return None
        scored = []
        for ev in events:
            score = 0
            if ev.get("is_main_predicate"): score += 20
            if ev.get("arg0") and ev["arg0"] != SLOT_TAG: score += 5
            if ev.get("arg1") and ev["arg1"] != SLOT_TAG: score += 5
            scored.append((ev, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0] if scored else None

    def _determine_routing(self, plan: List[Dict], repaired_main: Optional[Dict],
                           repaired_mod: Optional[Dict]) -> Tuple[str, str]:
        has_retrieve = any(s.get("type") == "retrieve" for s in plan)
        if has_retrieve:
            return "VECTOR", "依存句法树残缺或降级，走生成式 HyDE 向量检索"

        hop_steps = [s for s in plan if s.get("type") == "multi_hop_query"]
        hop_count = len(hop_steps)

        has_double_slots = any(
            s.get("subject") == SLOT_TAG and s.get("object") == SLOT_TAG
            for s in hop_steps
        )
        if has_double_slots:
            return "VECTOR", "检测到非连通的双 SLOT 错误，主动拦截熔断并走向量检索大盘"

        has_concept_match = any(s.get("predicate") == "CONCEPT_MATCH" for s in hop_steps)

        if hop_count == 1:
            return "SINGLE_HOP", "主谓宾完好契约通过，执行单跳物理图匹配"
        elif hop_count >= 2 and has_concept_match:
            return "MULTI_HOP_CASCADE", "含修饰定语从句，执行动态级联拓扑多步推理"
        elif hop_count >= 2:
            return "MULTI_HOP", "多跳实体链查询"
        return "TERM", "纯词项召回"

    # intent_analyzer.py 的 compile 方法 — 带完整调试日志版

    def compile(self, query: str, context_history: Optional[str] = None) -> Dict[str, Any]:
        logger.info(f"\n🔧 [Compiler 3.0] 启动 100% 零硬编码自适应编译管道...")
        execution_log = []

        try:
            parsed = self.struct.extract(query)
            events = parsed.get("events", [])
            tokens = parsed.get("tokens", [])
        except Exception as e:
            logger.error(f"❌ Stanza 提取事件异常: {e}")
            return self._build_fallback(query, "parser_crash", execution_log)

        # ============================================================
        # 🔍 调试块 1：打印 Stanza 提取的所有事件
        # ============================================================
        logger.info(f"🔍 [DEBUG] ========== Stanza 事件提取结果 ==========")
        logger.info(f"🔍 [DEBUG] Query: \"{query}\"")
        logger.info(f"🔍 [DEBUG] 事件数量: {len(events)}")
        for i, ev in enumerate(events):
            logger.info(f"🔍 [DEBUG]   事件[{i}]: predicate='{ev.get('predicate_text')}', "
                        f"arg0='{ev.get('arg0')}', arg1='{ev.get('arg1')}', "
                        f"is_main={ev.get('is_main_predicate')}, "
                        f"is_modifier={ev.get('is_modifier_constraint')}, "
                        f"is_passive={ev.get('is_passive')}")
        # 打印所有 token 的依存关系
        logger.info(f"🔍 [DEBUG] Token 依存树:")
        for t in tokens:
            logger.info(
                f"🔍 [DEBUG]   [{t['id']}] '{t['text']}' pos={t['pos']} deprel={t['deprel']} head={t['head']} feats='{t.get('feats', '')}'")
        logger.info(f"🔍 [DEBUG] =============================================")

        main_event = None
        modifier_event = None
        if len(events) >= 2:
            main_event = self._pick_event(events, query)
            modifiers = [e for e in events if e != main_event]
            modifier_event = modifiers[0] if modifiers else None
            logger.info(
                f"🔍 [DEBUG] _pick_event 结果: main='{main_event.get('predicate_text') if main_event else 'None'}', "
                f"modifier='{modifier_event.get('predicate_text') if modifier_event else 'None'}'")
        elif len(events) == 1:
            main_event = events[0]

        # ============================================================
        # 🔍 调试块 2：打印修复前的主事件和修饰事件
        # ============================================================
        if main_event:
            logger.info(f"🔍 [DEBUG] >>> 修复前 main_event: "
                        f"pred='{main_event.get('predicate_text')}', "
                        f"arg0='{main_event.get('arg0')}', arg1='{main_event.get('arg1')}'")
        if modifier_event:
            logger.info(f"🔍 [DEBUG] >>> 修复前 modifier_event: "
                        f"pred='{modifier_event.get('predicate_text')}', "
                        f"arg0='{modifier_event.get('arg0')}', arg1='{modifier_event.get('arg1')}'")

        # 修复主事件
        repaired_main = self.arg_repair.repair_event(main_event, tokens, query) if main_event else None

        if repaired_main:
            logger.info(f"🔍 [DEBUG] <<< 修复后 repaired_main: "
                        f"arg0='{repaired_main.get('arg0')}', arg1='{repaired_main.get('arg1')}', "
                        f"slot_role={repaired_main.get('slot_role')}, slot_type={repaired_main.get('slot_type')}")

        # 修复修饰事件（传入主事件）
        repaired_mod = self.arg_repair.repair_event(
            modifier_event, tokens, query, other_event=repaired_main
        ) if modifier_event else None

        if repaired_mod:
            logger.info(f"🔍 [DEBUG] <<< 修复后 repaired_mod: "
                        f"arg0='{repaired_mod.get('arg0')}', arg1='{repaired_mod.get('arg1')}', "
                        f"slot_role={repaired_mod.get('slot_role')}, slot_type={repaired_mod.get('slot_type')}")

        # ============================================================

        def is_valid_arg(val):
            return val is not None and str(val).strip() != "" and val != SLOT_TAG

        main_arg0_valid = is_valid_arg(repaired_main.get("arg0")) if repaired_main else False
        main_arg1_valid = is_valid_arg(repaired_main.get("arg1")) if repaired_main else False
        has_slot = (repaired_main.get("arg0") == SLOT_TAG or repaired_main.get(
            "arg1") == SLOT_TAG) if repaired_main else False

        double_slot_error = repaired_main and (
                repaired_main.get("arg0") == SLOT_TAG and repaired_main.get("arg1") == SLOT_TAG
        )
        kg_route_possible = repaired_main and (
                has_slot or (main_arg0_valid and main_arg1_valid)) and not double_slot_error

        # ============================================================
        # 🔍 调试块 3：打印路由决策依据
        # ============================================================
        logger.info(f"🔍 [DEBUG] 路由决策: main_arg0_valid={main_arg0_valid}, main_arg1_valid={main_arg1_valid}, "
                    f"has_slot={has_slot}, double_slot_error={double_slot_error}, kg_route_possible={kg_route_possible}")
        if repaired_mod:
            logger.info(f"🔍 [DEBUG] 修饰步骤检查: repaired_mod.arg0='{repaired_mod.get('arg0')}', "
                        f"repaired_mod.arg1='{repaired_mod.get('arg1')}', "
                        f"is_slot_arg0={repaired_mod.get('arg0') == SLOT_TAG}, "
                        f"is_slot_arg1={repaired_mod.get('arg1') == SLOT_TAG}")
        # ============================================================

        if kg_route_possible and repaired_main:
            try:
                execution_plan = []
                raw_pred = repaired_main.get("predicate_text") or repaired_main.get("predicate", "RELATED_TO")
                aligned_pred, _ = self.ret_if.align_predicate(raw_pred)
                if not aligned_pred:
                    aligned_pred = "RELATED_TO"

                slot_type = repaired_main.get("slot_type", "CONCEPT")

                main_step = {
                    "step_id": "g1" if repaired_mod else "g0",
                    "type": "multi_hop_query",
                    "subject": repaired_main["arg0"],
                    "predicate": aligned_pred,
                    "object": repaired_main["arg1"],
                    "outputs": ["hop_result_1" if repaired_mod else "hop_result_0"]
                }

                if repaired_main["arg0"] == SLOT_TAG:
                    main_step["subject_type_constraint"] = slot_type
                    main_step["output_binding"] = "subject"
                elif repaired_main["arg1"] == SLOT_TAG:
                    main_step["object_type_constraint"] = slot_type
                    main_step["output_binding"] = "object"
                else:
                    main_step["output_binding"] = "object"

                execution_plan.append(main_step)

                if repaired_mod:
                    mod_pred = repaired_mod.get("predicate_text") or repaired_mod.get("predicate", "RELATED_TO")
                    mod_aligned, _ = self.ret_if.align_predicate(mod_pred)
                    if not mod_aligned:
                        mod_aligned = "RELATED_TO"

                    mod_obj = repaired_mod.get("arg1")
                    if not mod_obj or str(mod_obj).strip() in ("", "None"):
                        mod_obj = repaired_mod.get("arg0")

                    logger.info(
                        f"🔍 [DEBUG] 修饰步骤 DAG 构建: mod_obj='{mod_obj}', in_vocab={mod_obj in self.vocab if mod_obj else False}")

                    if mod_obj and mod_obj not in self.vocab and mod_obj != SLOT_TAG:
                        logger.info(f"🔍 [DEBUG] → 走 CONCEPT_MATCH 路径")
                        concept_step = {
                            "step_id": "g0", "type": "multi_hop_query",
                            "subject": mod_obj, "predicate": "CONCEPT_MATCH",
                            "object": SLOT_TAG, "output_binding": "object",
                            "outputs": ["hop_result_0"]
                        }
                        execution_plan.insert(0, concept_step)

                        mod_step = {
                            "step_id": "g1", "type": "multi_hop_query",
                            "subject": SLOT_TAG, "predicate": mod_aligned,
                            "object": "hop_result_0", "output_binding": "subject",
                            "outputs": ["hop_result_1"]
                        }
                        execution_plan.insert(1, mod_step)
                        main_step["step_id"] = "g2"
                        main_step["subject"] = "hop_result_1"
                        main_step["outputs"] = ["hop_result_2"]
                    else:
                        logger.info(
                            f"🔍 [DEBUG] → 走普通修饰步骤路径, arg0='{repaired_mod.get('arg0')}', arg1='{repaired_mod.get('arg1')}'")
                        mod_step = {
                            "step_id": "g0", "type": "multi_hop_query",
                            "subject": repaired_mod["arg0"] if repaired_mod.get("arg0") and repaired_mod[
                                "arg0"] != SLOT_TAG else SLOT_TAG,
                            "predicate": mod_aligned,
                            "object": repaired_mod["arg1"] if repaired_mod.get("arg1") and repaired_mod[
                                "arg1"] != SLOT_TAG else SLOT_TAG,
                            "output_binding": "subject" if repaired_mod.get("arg0") == SLOT_TAG else "object",
                            "outputs": ["hop_result_0"]
                        }
                        execution_plan.insert(0, mod_step)
                        main_step["subject"] = "hop_result_0"

                execution_plan.append({"step_id": "synth", "type": "synthesize_answer"})

                # ============================================================
                # 🔍 调试块 4：打印最终执行计划
                # ============================================================
                logger.info(f"🔍 [DEBUG] ========== 最终执行计划 ==========")
                for s in execution_plan:
                    sid = s.get("step_id", "?")
                    stype = s.get("type", "?")
                    if stype == "multi_hop_query":
                        logger.info(
                            f"🔍 [DEBUG]   {sid}: ({s.get('subject')}) -[{s.get('predicate')}]-> ({s.get('object')}), binding={s.get('output_binding')}, subj_type={s.get('subject_type_constraint')}, obj_type={s.get('object_type_constraint')}")
                    else:
                        logger.info(f"🔍 [DEBUG]   {sid}: {stype}")
                logger.info(f"🔍 [DEBUG] =====================================")
                # ============================================================

                routing_type, route_reason = self._determine_routing(execution_plan, repaired_main, repaired_mod)
                return self._build_payload(0.95, execution_plan, query, execution_log, routing_type, route_reason)

            except Exception as e:
                logger.error(f"❌ [Compiler] DAG 解析崩溃: {e}")
                return self._build_fallback(query, f"dag_error: {e}", execution_log)
        else:
            return self._build_fallback(query, "argument_insufficient", execution_log)
    def _build_fallback(self, query: str, reason: str, log: List[str]) -> Dict[str, Any]:
        entities = extract_entities_from_text(query, self.vocab)
        execution_plan = []
        routing_type = "VECTOR"
        route_reason = f"句式不完整。原因: {reason}"

        if entities and len(entities) == 1:
            main_ent = entities[0]["text"]
            execution_plan = [
                {"step_id": "g0", "type": "multi_hop_query",
                 "subject": main_ent, "predicate": "CONCEPT_MATCH",
                 "object": SLOT_TAG, "output_binding": "object",
                 "outputs": ["fallback_hop"]},
                {"step_id": "synth", "type": "synthesize_answer"}
            ]
            routing_type = "CONCEPT_MATCH"
            route_reason = f"依存失效但提取到实体 \"{main_ent}\"，触发 L1 概念检索"
        elif entities and len(entities) >= 2:
            ent_list = [e["text"] for e in entities]
            execution_plan = [
                {"step_id": "g0", "type": "multi_hop_query",
                 "subject": ent_list[0], "predicate": "BRIDGE_PATH",
                 "object": ent_list[1], "output_binding": "object",
                 "outputs": ["fallback_hop"]},
                {"step_id": "synth", "type": "synthesize_answer"}
            ]
            routing_type = "SUBGRAPH_CO_OCCUR"
            route_reason = f"触发 L2 拓扑邻域共现。关联实体簇: {ent_list}"
        else:
            execution_plan = [
                {"step_id": "r0", "type": "retrieve",
                 "dense_query": query, "sparse_query": query,
                 "outputs": ["hop_result"]},
                {"step_id": "synth", "type": "synthesize_answer"}
            ]
            routing_type = "VECTOR"
            route_reason = "依存分析与实体识别全部落空，降级为纯 HyDE 向量检索"

        return self._build_payload(0.70, execution_plan, query, log, routing_type, route_reason)

    def _build_payload(self, conf: float, plan: List[Dict], query: str, log: List[str],
                       routing_type: str = "TERM", route_reason: str = "") -> Dict[str, Any]:
        return {
            "intent_meta": {
                "type": "adaptive_cognitive_compiled",
                "confidence": conf,
                "routing_type": routing_type,
                "route_reason": route_reason,
                "directive": {
                    "execution_plan": plan,
                    "dsl": {"rewritten": {"resolved": query, "dense": query, "sparse": query}}
                }
            },
            "resolved_cache": {},
            "derived_entities": {},
            "execution_log": log
        }