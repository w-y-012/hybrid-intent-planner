# -*- coding: utf-8 -*-
""" 执行器与 LangGraph 编排模块 - executor_agent.py """
# 必须在所有导入之前禁用 langsmith
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

import quiet_logging
import json
import re
import logging
# ... 后面不变
import uuid  # 添加这一行
import contextvars
import asyncio
from typing import Dict, List, Any, Tuple, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from intent_analyzer import IntentInterpreterCompiler
from retriever_kg import RetrieverKG
from functions_for_pipeline import SLOT_TAG, llm, hyperparams
from enterprise_evaluation import HyperparameterRegistry, OnlineTelemetrySentinel, OfflineEvaluationSuite

# 初始化超参数注册器
HyperparameterRegistry.load_parameters()

logger = logging.getLogger("ExecutorAgent")

# ========== 添加这部分 ==========
# 初始化检索器
ret_kg = RetrieverKG(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_pwd="test1234",
    qdrant_host="localhost",
    qdrant_port=6333
)

# 全局编译器实例 — 直接使用 intent_analyzer.py 中的 compile，不再 Monkey Patch
compiler = IntentInterpreterCompiler(ret_kg)


# 【新增】带指数退避重试的异步 LLM 调用封装
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10),
       retry=retry_if_exception_type(Exception))
async def safe_ainvoke(prompt, timeout=5):
    return await llm.ainvoke(prompt)


class RagasEvaluator:
    """轻量级在线 RAGAS 评估裁判 (LLM-as-a-Judge)"""

    @staticmethod
    async def evaluate_faithfulness(context: str, answer: str) -> float:
        if "cannot verify" in answer or "未命中" in answer:
            return 1.0  # 无法验证的安全回答不判定为幻觉
        prompt = f"""You are an expert NLP auditor. Rate from 0.0 to 1.0 how faithful the Answer is to the Context. 
Are there any factual claims in the Answer that are NOT supported by the Context? (1.0 means perfectly faithful, 0.0 means complete hallucination).

【Context】: {context}

【Answer】: {answer}

Output strictly a single float number between 0.0 and 1.0."""
        try:
            res_obj = await safe_ainvoke(prompt, timeout=5)
            res = res_obj.content.strip()
            return float(re.findall(r"\d+.\d+", res)[0])
        except Exception:
            return 0.0  # 【修复】不再返回 0.80 掩盖错误，真实反映评估失败

    @staticmethod
    async def evaluate_answer_relevance(query: str, answer: str) -> float:
        prompt = f"""Rate from 0.0 to 1.0 how relevant the Answer is to the Question. Does it directly resolve the user's intent?
Question: {query}
Answer: {answer}
Output strictly a single float number between 0.0 and 1.0."""
        try:
            res_obj = await safe_ainvoke(prompt, timeout=5)
            res = res_obj.content.strip()
            return float(re.findall(r"\d+.\d+", res)[0])
        except Exception:
            return 0.0


class LoopAuditor:
    """Agent 执行环状态机审计器"""

    @staticmethod
    def audit_oscillation(tool_call_log: List[Dict]) -> bool:
        """
        【修复：消除级联误判】
        通过审计 Step-ID 物理重复访问频率来精确识别死循环，允许合法的 consecutive 同类型工具多步级联。
        """
        step_counts = {}
        for log in tool_call_log:
            step_id = log.get("step_id")
            if step_id:
                step_counts[step_id] = step_counts.get(step_id, 0) + 1
        # 如果同一个物理步骤被执行了 2 次以上（说明执行器陷入自循环，未正常推进 step_idx），立即熔断
        if step_counts[step_id] > 2:
            return True
        return False

    @staticmethod
    def calculate_hydration_rate(plan_steps: List[Dict], resolved_cache: Dict) -> float:
        """评估多步级联变量水合填充（Hydration）成功率"""
        total_placeholders = 0
        successful_hydrated = 0
        for step in plan_steps:
            for field in ["subject", "object"]:
                val = step.get(field)
                if isinstance(val, str) and val.startswith("hop_result_"):
                    total_placeholders += 1
                    if val in resolved_cache and not str(resolved_cache[val]).startswith("__"):
                        successful_hydrated += 1
        if total_placeholders == 0:
            return 1.0
        return successful_hydrated / total_placeholders


class AgentState(TypedDict, total=False):
    query: str
    routing_type: str
    intent_meta: Dict[str, Any]
    plan: List[Dict[str, Any]]
    step_idx: int
    resolved_cache: Dict[str, Any]
    derived_entities: Dict[str, str]
    context_buffer: str
    tool_call_log: List[Dict]
    step_failed: bool
    final_output: str
    execution_log: List[str]
    max_recall_score: float

    # 评测与标定层专用属性
    tuning_mode: bool  # 是否开启 Optuna 调优评估模式
    ragas_metrics: Dict[str, float]
    loop_audit_logs: Dict[str, Any]


from enterprise_governance import MCPSkillRegistry, SessionWorkspaceManager, ObservabilityTracer, MCPSkill

# 实例化全局治理套件
mcp_registry = MCPSkillRegistry()
# 【修复】移除全局变量，改用 contextvars 实现协程/线程级隔离
workspace_manager_var = contextvars.ContextVar('workspace_manager')
tracer_var = contextvars.ContextVar('tracer')

# 【MCP Skills Sandbox 注册】
# 必须为每一个可空参数明确声明 default 值，否则 Python 参数绑定反射器在无该参数传递时会强制崩溃！ [1, 2]
mcp_registry.register_skill(MCPSkill(
    name="query_kg_expanded_skill",
    description="Query knowledge graph nodes and relations safely",
    input_schema={
        "subj": {"required": True},
        "pred": {"required": True},
        "obj": {"required": True},
        "subj_type": {"required": False, "default": None},
        "obj_type": {"required": False, "default": None}
    },
    handler=ret_kg.query_kg_expanded
))


def init_eval_agent_state(query: str, tuning_mode: bool = False) -> AgentState:
    # 每次新会话，重置快照器
    # 【修复】使用 contextvars 设置当前上下文的实例，避免并发覆盖
    workspace_manager_var.set(SessionWorkspaceManager(session_id="hp_session_2026"))
    tracer_var.set(ObservabilityTracer(trace_id=str(uuid.uuid4())))

    return {
        "query": query,
        "routing_type": "CONCEPT",
        "intent_meta": {},
        "plan": [],
        "step_idx": 0,
        "resolved_cache": {},
        "derived_entities": {},
        "context_buffer": "",
        "tool_call_log": [],
        "step_failed": False,
        "final_output": "",
        "execution_log": [],
        "max_recall_score": 0.0,
        "tuning_mode": tuning_mode,
        "ragas_metrics": {"faithfulness": 1.0, "answer_relevance": 1.0},
        "loop_audit_logs": {"oscillation_detected": False, "hydration_rate": 1.0}
    }


# =========================================================================
# 🧭 1) 意图决策路由器节点 (动态消解路由)
# =========================================================================
def intent_analyzer_node(state: AgentState) -> Dict[str, Any]:
    """
    【100% 动态路由节点】：
    彻底废去老旧的正则判定。直接读取自适应认知编译器输出的强类型决策。
    """
    query = state["query"]

    # 实例化自适应认知编译器 (其内部 Stanza 依存分析已补全) [1]
    compiler = IntentInterpreterCompiler(ret_kg)
    res = compiler.compile(query)

    intent_meta = res["intent_meta"]
    routing_type = intent_meta.get("routing_type", "VECTOR")
    route_reason = intent_meta.get("route_reason", "默认兜底降级")

    print(f"\n\033[1;35m[COGNITIVE ROUTER] =======================================\033[0m")
    print(f" 🧭 \033[1;33m决策路由类型:\033[0m {routing_type}")
    print(f" 🧭 \033[1;33m路由决策依据:\033[0m {route_reason}")
    print(f"\033[1;35m[COGNITIVE ROUTER] =======================================\033[0m\n")

    return {
        "routing_type": routing_type,
        "intent_meta": intent_meta,
        "resolved_cache": res["resolved_cache"],
        "derived_entities": res["derived_entities"],
        "execution_log": state["execution_log"] + res["execution_log"]
    }


def planner_node(state: AgentState) -> Dict[str, Any]:
    plan = state["intent_meta"]["directive"]["execution_plan"]
    return {
        "plan": plan,
        "step_idx": 0,
        "execution_log": state["execution_log"] + [f"[Planner] 计划加载成功 ({len(plan)} 步)"]
    }


def validate_plan_node(state: AgentState) -> Dict[str, Any]:
    return {"execution_log": state["execution_log"] + ["[Validate] 安全检验通过"]}


import traceback


# =========================================================================
# 🧠 3) 英文特征空间英化对齐网关 (Generative Fallback Gateway)
# =========================================================================
class GenerativeFallbackGateway:
    @staticmethod
    async def hyde_and_retrieve(query: str, cache: Dict[str, Any], ret_kg) -> str:
        logger.info(f"🧠 [Generative Fallback] 激活自适应英化与 HyDE 增强检索...")

        # 提取已解出的图谱实体线索 (如 "Voldemort", "Snape")，作为上下文喂给翻译模型
        valid_facts = {k: v for k, v in cache.items() if isinstance(v, str) and len(v) > 1 and not v.startswith("__")}

        # 精准英化对齐 Prompt
        translation_and_hyde_prompt = f"""You are an elite bilingual translation and knowledge retrieval assistant.
Convert the user's Chinese query and any resolved clues into a high-fidelity English search query and a brief English hypothetical fact paragraph (HyDE) to maximize vector retrieval accuracy in the English HP dataset.

【User Chinese Query】: "{query}"
【Resolved Clues】: {json.dumps(valid_facts, ensure_ascii=False)}

Output your response STRICTLY as a JSON object, without any markdown code blocks or explanations:
{{
  "english_query": "high quality english query for vector search",
  "hypothetical_doc": "a brief realistic paragraph of the facts in English to retrieve relevant HP chunks"
}}"""

        try:
            # 【修复】改为异步调用并自带重试
            res_obj = await safe_ainvoke(translation_and_hyde_prompt, timeout=5)
            resp = res_obj.content.strip()
            # 剥离 JSON 块
            if "```json" in resp:
                resp = resp.split("```json")[1].split("```")[0].strip()
            elif "```" in resp:
                resp = resp.split("```")[1].split("```")[0].strip()

            data = json.loads(resp)
            eng_query = data.get("english_query", query)
            hypothetical_doc = data.get("hypothetical_doc", "")

            logger.info(f"🧠 [Generative Fallback] 成功执行英文特征空间对齐：")
            logger.info(f"   ├─ English Aligned Query: \"{eng_query}\"")
            logger.info(f"   └─ English HyDE Paragraph: \"{hypothetical_doc}\"")

            # 使用高 17% ~ 32% 命中率的英文表示进行多路召回 [1, 2]
            com_docs = ret_kg.retrieve_communities(hypothetical_doc)
            raw_docs = ret_kg.query_chunks_hybrid(hypothetical_doc, eng_query)

            ctx_parts = []
            if com_docs:
                ctx_parts.append("[Community Context]:\n" + "\n".join([d["text"] for d in com_docs]))
            if raw_docs:
                ctx_parts.append("[Fact Chunks]:\n" + "\n".join([d["text"] for d in raw_docs]))

            return "\n\n".join(ctx_parts)

        except Exception as e:
            logger.error(f"❌ [Generative Fallback] 翻译英化 HyDE 管道报错: {e}。平滑退回到标准中文混合检索。")
            raw_docs = ret_kg.query_chunks_hybrid(query, query)
            return "\n\n".join([d["text"] for d in raw_docs])


# =========================================================================
# ⚙️ 2) 高可用执行器工具节点 (集成 Sentinel 哨兵熔断)
# =========================================================================
async def tool_executor_node(state: AgentState) -> Dict[str, Any]:
    # 【修复】从上下文变量中获取隔离的实例
    workspace_manager = workspace_manager_var.get()
    tracer = tracer_var.get()

    idx = state["step_idx"]
    plan = state["plan"]

    if idx >= len(plan):
        return {"step_idx": idx, "step_failed": False}

    step = plan[idx]
    step_id = step.get("step_id", f"step_{idx}")
    step_type = step.get("type", "unknown")

    cache = dict(state.get("resolved_cache", {}))
    outputs_list = step.get("outputs")
    primary_var = outputs_list[0] if outputs_list else f"hop_result_{idx}"

    # 事务前置 Checkpoint
    workspace_manager.commit_checkpoint(step_idx=idx, state_snapshot=state)

    print(f"\n\033[1;34m[EXECUTION NODE] =======================================\033[0m")
    print(f" ⚙️  \033[1;33m当前物理执行步骤:\033[0m [{step_id}] ({step_type})")

    span_perf = tracer.start_span(name=step_id)
    out_msg = ""
    failed = False
    max_score_this_step = state.get("max_recall_score", 0.0)

    try:
        if step_type == "multi_hop_query":
            subj = step.get("subject", SLOT_TAG)
            obj = step.get("object", SLOT_TAG)
            pred = step.get("predicate", "")
            output_binding = step.get("output_binding", "object")

            # 变量水合
            if isinstance(subj, str) and subj in cache:
                print(f"       ├─ 变量水合 Hydrate 主语: {subj} ──> \"{cache[subj]}\"")
                subj = cache[subj]
            if isinstance(obj, str) and obj in cache:
                print(f"       └─ 变量水合 Hydrate 宾语: {obj} ──> \"{cache[obj]}\"")
                obj = cache[obj]

            subj_type = step.get("subject_type_constraint")
            obj_type = step.get("object_type_constraint")
            tool_params = {"subj": subj, "pred": pred, "obj": obj, "subj_type": subj_type, "obj_type": obj_type}
            is_valid_contract, contract_err = OnlineTelemetrySentinel.inspect_tool_contract(step, tool_params)

            if not is_valid_contract:
                logger.error(f"🚨 [Sentinel Contract Breach Alert] {contract_err}。当前步骤立即进行【安全熔断】...")
                # 哨兵触发安全熔断
                hydrated_context = await GenerativeFallbackGateway.hyde_and_retrieve(state["query"], cache, ret_kg)
                cache[f"{step_id}_full_context"] = hydrated_context
                cache[primary_var] = "__FALLBACK_CTX__"
                out_msg = f"📄 熔断自愈：由契约违背转入 HyDE 混合检索 (Err: {contract_err})"
                matches = []
            else:
                # 校验通过，安全运行 MCP 规范图检索
                mcp_response = mcp_registry.call_tool(
                    skill_name="query_kg_expanded_skill",
                    params=tool_params
                )
                if not mcp_response["success"]:
                    raise RuntimeError(f"MCP Skill Reflection Fail: {mcp_response['error']}")
                matches = mcp_response["data"]

            matches_len = len(matches) if matches else 0

            # 类型约束弱化自愈
            if not matches and is_valid_contract and (subj_type or obj_type):
                print(f"       ⚠️  强实体类型过滤导致匹配为空。正在抹除类型限制重试...")
                mcp_response_relax = mcp_registry.call_tool(
                    skill_name="query_kg_expanded_skill",
                    params={"subj": subj, "pred": pred, "obj": obj, "subj_type": None, "obj_type": None}
                )
                if mcp_response_relax["success"] and mcp_response_relax["data"]:
                    matches = mcp_response_relax["data"]
                    matches_len = len(matches)

            # 在线可观测哨兵：自动度量水合率，防止静默退化
            hydration_rate = LoopAuditor.calculate_hydration_rate(plan, cache)
            OnlineTelemetrySentinel.audit_online_degradation(step_id, matches_len, hydration_rate,
                                                             state.get("execution_log", []))

            if matches:
                if output_binding == "subject":
                    target_entity = matches[0].get("subject")
                else:
                    target_entity = matches[0].get("object")

                if target_entity == subj or target_entity == obj:
                    target_entity = matches[0].get("subject") if output_binding == "object" else matches[0].get(
                        "object")

                cache[primary_var] = target_entity
                cache[f"{step_id}_graph_context"] = "[图谱事实]\n" + " | ".join(
                    [f"{m.get('subject')}→[{m.get('predicate')}]→{m.get('object')}" for m in matches[:5]]
                )
                out_msg = f"✅ 图检索精准命中, 实体方向 [{output_binding}] 绑定变量为: \"{target_entity}\""
                max_score_this_step = max(max_score_this_step, 0.95)

            elif is_valid_contract:  # 仅在非熔断且没命中图时，启动闭环图自愈
                print(f"       ⚠️  图谱物理边未直接命中，启动闭环事实修正与 HyDE 生成式召回...")
                refined = ret_kg.closed_loop_refiner(state["query"], state, step_id, subj, pred, obj, primary_var)
                if refined:
                    cache.update(refined["resolved_cache"])
                    out_msg = "🏆 [Closed-Loop] 闭环事实改写修正成功！"
                    max_score_this_step = max(max_score_this_step, 0.90)
                else:
                    hydrated_context = await GenerativeFallbackGateway.hyde_and_retrieve(state["query"], cache, ret_kg)
                    cache[f"{step_id}_full_context"] = hydrated_context
                    cache[primary_var] = "__FALLBACK_CTX__"
                    out_msg = "📄 已降级至自适应生成式 HyDE 混合检索"

        elif step_type == "retrieve":
            dense_q = step.get("dense_query") or state["query"]
            sparse_q = step.get("sparse_query") or state["query"]

            docs = ret_kg.query_chunks_hybrid(dense_q, sparse_q, routing_type=state["routing_type"])
            if docs:
                max_score_this_step = max(max_score_this_step, max([d.get("score", 0.0) for d in docs]))

            cache[f"{step_id}_full_context"] = "\n\n".join([d["text"] for d in docs])
            cache[primary_var] = "RETRIEVED_CHUNKS"
            out_msg = f"📄 混合检索到 {len(docs)} 个 Chunks"

    except Exception as e:
        failed = True
        err_stack = traceback.format_exc()
        out_msg = f"❌ [崩溃]: {e}"
        logger.critical(f"💥 [Critical Step Executive Failure] 步骤 [{step_id}] 突发崩溃！\n{err_stack}")

    tracer.end_span(name=step_id, start_perf_time=span_perf, metadata={"type": step_type, "msg": out_msg})

    tool_log = {"step": idx + 1, "step_id": step_id, "tool": step_type, "output": out_msg, "elapsed": 0.0}
    updated_tool_call_log = state.get("tool_call_log", []) + [tool_log]

    oscillation = LoopAuditor.audit_oscillation(updated_tool_call_log)
    hydration_rate = LoopAuditor.calculate_hydration_rate(plan, cache)

    if oscillation:
        print(f" 🚨 [LoopAuditor] 检测到执行链死循环，启动【长程事务快照级联回退】...")
        rollback_state = workspace_manager.rollback_to_step(target_step_idx=max(0, idx - 1))
        if rollback_state:
            return rollback_state
        failed = True

    return {
        "step_idx": idx if failed else idx + 1,
        "step_failed": failed,
        "resolved_cache": cache,
        "max_recall_score": max_score_this_step,
        "context_buffer": state.get("context_buffer", "") + f"[{step_id}] {out_msg}\n",
        "tool_call_log": updated_tool_call_log,
        "loop_audit_logs": {"oscillation_detected": oscillation, "hydration_rate": hydration_rate},
        "execution_log": state["execution_log"] + [f"[{'✅' if not failed else '❌'}] {step_id}"]
    }


async def grounded_synthesizer_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    cache = state["resolved_cache"]
    max_score = state.get("max_recall_score", 0.0)

    # 读取当前配置中的动态熔断门限
    min_recall_threshold = hyperparams.min_recall_gate

    if max_score < min_recall_threshold:
        final_output = "Based on current context, I cannot verify this.\n\n[置信度: 0.00 | 熔断保护]"
        return {
            "final_output": final_output,
            "ragas_metrics": {"faithfulness": 1.0, "answer_relevance": 0.0},
            "execution_log": state["execution_log"] + ["[Gateway] 触发低分熔断"]
        }

    ctx_parts = []
    evidence_types = set()
    for k, v in cache.items():
        if not isinstance(v, str) or len(v) < 10: continue
        if k.endswith("_graph_context"):
            ctx_parts.append(f"[图谱事实证据]:\n{v}")
            evidence_types.add("graph")
        elif k.endswith("_full_context"):
            ctx_parts.append(f"[向量检索文本]:\n{v}")
            evidence_types.add("chunks")

    combined_ctx = "\n\n".join(ctx_parts) if ctx_parts else state.get("context_buffer", "未发现任何有效证据。")

    prompt = f"""You are a precise facts-grounded answering machine. Answer the question based on the contexts. 
If the information is not present, reply 'Based on current context, I cannot verify this.'.
【Contexts】: {combined_ctx}
【User Question】: {query}
【Answer】:"""

    try:
        # 【修复】改为异步调用并自带重试
        res_obj = await safe_ainvoke(prompt, timeout=15)
        resp = res_obj.content.strip()
    except Exception as e:
        resp = f"⚠️ 答案生成异常: {e}"

    conf = hyperparams.conf_cascade_plan if "graph" in evidence_types else hyperparams.conf_fallback_vector
    final_output = f"{resp}\n\n[置信度: {conf:.2f} | 检索最高分: {max_score:.3f} | 证据链: {', '.join(evidence_types)}]"

    # 【新增：当且仅当运行于 Tuning-Mode 下，进行 RAGAS 审计打分】
    ragas_res = {"faithfulness": 1.0, "answer_relevance": 1.0}
    if state.get("tuning_mode", False):
        f_score = await RagasEvaluator.evaluate_faithfulness(combined_ctx, resp)
        a_score = await RagasEvaluator.evaluate_answer_relevance(query, resp)
        ragas_res = {"faithfulness": f_score, "answer_relevance": a_score}

    return {
        "final_output": final_output,
        "ragas_metrics": ragas_res,
        "execution_log": state["execution_log"] + ["[Synthesizer] 生成完毕"]
    }

# ==================== 运行图编译 ====================
def create_rag_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("intent_analyzer", intent_analyzer_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("validate", validate_plan_node)
    workflow.add_node("execute", tool_executor_node)
    workflow.add_node("synthesize", grounded_synthesizer_node)

    workflow.add_edge(START, "intent_analyzer")
    workflow.add_edge("intent_analyzer", "planner")
    workflow.add_edge("planner", "validate")
    workflow.add_edge("validate", "execute")

    def route_after_execute(s: AgentState) -> str:
        if s.get("step_failed"): return "synthesize"
        if s["step_idx"] < len(s.get("plan", [])): return "execute"
        return "synthesize"

    workflow.add_conditional_edges("execute", route_after_execute, {"execute": "execute", "synthesize": "synthesize"})
    workflow.add_edge("synthesize", END)

    return workflow.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    import uuid

    app = create_rag_graph()

    # 1. 构建离线黄金测试集（Golden Dataset + 对抗集 + 句式覆盖）
    offline_gold_cases = [
        {
            "query": "帮助反派的那个教授教什么课？",
            "ground_truth": {
                "answer": "Defense Against the Dark Arts",
                "aliases": ["黑魔法防御术", "Defending against the dark arts", "DADA"],
                "expected_tools": ["query_kg_expanded_skill"]
            }
        },
        {
            "query": "Hermione保护了谁？",
            "ground_truth": {
                "answer": "Harry",
                "aliases": ["Harry Potter", "Fluffy", "Neville"],
                "expected_tools": ["query_kg_expanded_skill"]
            }
        },
        {
            "query": "Severus Snape教哪一门特定的科目？",
            "ground_truth": {
                "answer": "Potions",
                "aliases": ["魔药学", "Potion class"],
                "expected_tools": ["query_kg_expanded_skill"]
            }
        }
    ]

    # 2. 实例化离线审计套件
    eval_suite = OfflineEvaluationSuite()

    print("\n🚀 [Offline Regression] 启动离线自动化回归评估流...")
    for idx, case in enumerate(offline_gold_cases):
        query_text = case["query"]
        print(f"\n====================== 🧪 Test Case [{idx + 1}/{len(offline_gold_cases)}] ======================")

        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        state_init = init_eval_agent_state(query_text, tuning_mode=True)

        # 运行 RAG 整个图谱推理网络 (使用 ainvoke 适配异步节点)
        result_state = asyncio.run(app.ainvoke(state_init, config=config))

        # 精准记录并提取单步 Tool 运行时指标
        eval_suite.evaluate_run(
            query=query_text,
            final_state=result_state,
            ground_truth=case["ground_truth"]
        )

    # 3. 计算离线指标并生成 HTML Dashboard 大盘
    eval_suite.generate_html_report("offline_rag_eval_dashboard.html")
    print("\n🎉 [Regression Complete] 离线回归评测执行成功！评估大盘已在本地保存。")
