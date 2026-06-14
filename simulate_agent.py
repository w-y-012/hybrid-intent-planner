# -*- coding: utf-8 -*-
"""
simulate_agent.py - Agent 执行流可视化与实时追踪平台
适配异步 executor_agent
"""
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
import warnings
import sys
import io
import re
import json
import uuid
import asyncio
from typing import Dict, List, Any, Tuple, Optional

warnings.filterwarnings("ignore")

from executor_agent import create_rag_graph, init_eval_agent_state

# 🎨 配色方案
BG_MAIN = "#0b0f19"
BG_CARD = "#111827"
TEXT_PRIMARY = "#f8fafc"
TEXT_SECONDARY = "#94a3b8"
ACCENT_BLUE = "#3b82f6"
ACCENT_GREEN = "#10b981"
ACCENT_RED = "#ef4444"

NODE_MAP = {
    "intent_analyzer": "🧠 意图分析",
    "planner": "📋 计划生成",
    "validate": "✅ 计划验证",
    "execute": "⚙️ 执行检索",
    "synthesize": "💡 答案合成",
    "__start__": "🚀 入口",
    "__end__": "🏁 结束",
}


def extract_dynamic_topology(compiled_app):
    """提取 LangGraph 拓扑"""
    try:
        graph = compiled_app.get_graph()
        nodes = [
            {"id": str(nid), "label": NODE_MAP.get(str(nid), getattr(node, 'name', str(nid)))}
            for nid, node in graph.nodes.items()
        ]
        edges = [
            (str(e.source) if hasattr(e, 'source') else str(e[0]),
             str(e.target) if hasattr(e, 'target') else str(e[1]))
            for e in graph.edges
        ]
        return nodes, edges
    except Exception:
        return [], []


def render_graph_html(curr, nodes, edges):
    """渲染 PyVis 拓扑图"""
    if not nodes:
        return ""
    net = Network(directed=True, height="360px", width="100%", bgcolor=BG_MAIN,
                  notebook=False, font_color=TEXT_PRIMARY)
    net.toggle_physics(False)
    net.set_options(json.dumps({
        "physics": {"enabled": False},
        "interaction": {"hover": True, "zoomView": False, "dragView": False}
    }))
    for n in nodes:
        active = n["id"] == curr
        net.add_node(
            n["id"], label=n["label"],
            color=ACCENT_GREEN if active else "#374151",
            font=dict(size=13, color=TEXT_PRIMARY if active else "#d1d5db", face="sans-serif"),
            borderWidth=2, borderColor=ACCENT_GREEN if active else "#4b5563",
            size=20 if active else 15, physics=False
        )
    for s, t in edges:
        net.add_edge(s, t, color="#4b5563", arrows="to", width=1.5, smooth=False)
    return net.generate_html(notebook=False)


class StreamCapture:
    """安全捕获 stdout"""
    def __init__(self):
        self.buffer = io.StringIO()
        self.old_stdout = None

    def __enter__(self):
        self.old_stdout = sys.stdout
        sys.stdout = self.buffer
        return self

    def __exit__(self, *args):
        sys.stdout = self.old_stdout

    def get_and_clear(self):
        content = self.buffer.getvalue()
        self.buffer.truncate(0)
        self.buffer.seek(0)
        return content

    def get_remaining(self):
        return self.buffer.getvalue()


async def _async_stream(app, state_init, config, limit, log_ph, sys_log_ph, g_ph,
                        s_nodes, s_edges, push_log, push_sys_log, capture):
    """异步执行 Agent 流"""
    prog = st.progress(0.0, text="初始化执行流...")
    step = 0
    full_state = {}
    last_curr = None
    final_resp = "未生成"

    try:
        async for output in app.astream(state_init, config=config):
            step += 1

            # 捕获终端日志
            captured = capture.get_and_clear()
            if captured:
                for line in captured.splitlines():
                    push_sys_log(line)

            for node, delta in output.items():
                if not isinstance(delta, dict):
                    continue
                full_state.update(delta)

                plan = full_state.get("plan", [])
                step_idx = full_state.get("step_idx", 0)
                routing_type = full_state.get("routing_type", "")
                max_recall = full_state.get("max_recall_score", 0.0)
                final_output = full_state.get("final_output", "")
                resolved_cache = full_state.get("resolved_cache", {})

                if final_output:
                    final_resp = final_output

                # 构建日志
                lines = []
                if routing_type:
                    lines.append(f"🧭 **路由**: `{routing_type}`")
                lines.append(f"📋 **进度**: `{step_idx}/{len(plan)}` | 召回分: `{max_recall:.3f}`")

                if plan and step_idx < len(plan):
                    cs = plan[step_idx]
                    sid = cs.get("step_id", "?")
                    stype = cs.get("type", "?")
                    lines.append(f"   ↳ 当前: `{sid}` ({stype})")
                    if stype == "multi_hop_query":
                        lines.append(f"   ↳ `({cs.get('subject', '?')}) -[{cs.get('predicate', '?')}]-> ({cs.get('object', '?')})`")

                if resolved_cache:
                    items = [f"`{k}`=`{v}`" for k, v in resolved_cache.items() if k.startswith("hop_result_")]
                    if items:
                        lines.append(f"💧 **水合**: {' | '.join(items[:5])}")

                push_log(NODE_MAP.get(node, node.upper()), "<br>".join(lines))

                # 更新拓扑图
                if node != last_curr:
                    html = render_graph_html(node, s_nodes, s_edges)
                    if html:
                        g_ph.empty()
                        with g_ph:
                            components.html(html, height=370, scrolling=False)
                    last_curr = node

            prog.progress(min(step / limit, 1.0),
                          text=f"Step {step}/{limit} | {NODE_MAP.get(node, node.upper())}")

            if step >= limit:
                st.warning("⚠️ 触及递归上限")
                break

        push_log("🏁 流程结束", f"✅ 最终答案 (长度: `{len(final_resp)}` 字)")
        return final_resp

    except Exception as e:
        st.error(f"💥 异常: {e}")
        import traceback
        with st.expander("🔍 错误详情"):
            st.code(traceback.format_exc())
        return f"Error: {e}"
    finally:
        remaining = capture.get_remaining()
        if remaining:
            for line in remaining.splitlines():
                push_sys_log(line)
        prog.empty()


def run_streamed(query, app, log_ph, sys_log_ph, g_ph, limit=40):
    """同步包装异步执行"""
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state_init = init_eval_agent_state(query, tuning_mode=True)
    s_nodes, s_edges = extract_dynamic_topology(app)

    # 初始渲染拓扑图
    with g_ph.container():
        components.html(render_graph_html("__start__", s_nodes, s_edges), height=370, scrolling=False)

    # 日志收集器
    timeline = []
    sys_logs = []

    def push_log(title, details):
        timeline.append(
            f"""<div style='margin:5px 0;padding:8px 10px;background:{BG_CARD};border-left:3px solid {ACCENT_BLUE};
            border-radius:6px;font-family:ui-monospace,monospace;font-size:12.5px;color:{TEXT_PRIMARY};
            line-height:1.6;'>
            <b style='color:{ACCENT_BLUE};'>📍 {title}</b><br>
            <span style='color:{TEXT_SECONDARY};'>{details}</span></div>""")
        log_ph.markdown(
            f"<div style='height:340px;overflow-y:auto;padding:4px;background:{BG_MAIN};border-radius:8px;'>"
            f"{''.join(timeline[-30:])}</div>",
            unsafe_allow_html=True)

    def push_sys_log(msg):
        clean = re.sub(r'\x1b\[[0-9;]*m', '', msg).strip()
        if clean:
            sys_logs.append(
                f'<span style="color:#6b7280;font-size:11.5px;font-family:monospace;">{clean}</span>')
        sys_log_ph.markdown(
            f"<div style='height:140px;overflow-y:auto;padding:4px;background:{BG_CARD};border-radius:6px;'>"
            f"{''.join(sys_logs[-80:])}</div>",
            unsafe_allow_html=True)

    # 异步执行
    with StreamCapture() as capture:
        return asyncio.run(
            _async_stream(app, state_init, config, limit, log_ph, sys_log_ph, g_ph,
                          s_nodes, s_edges, push_log, push_sys_log, capture)
        )


def main():
    st.set_page_config(page_title="RAG Agent 执行流可视化", page_icon="🤖", layout="wide")
    st.title("🤖 RAG Agent 实时追踪平台")
    st.markdown("---")

    if "app" not in st.session_state:
        with st.spinner("⚙️ 编译 LangGraph & 加载 KG + 模型..."):
            st.session_state.app = create_rag_graph()
            st.success("✅ Agent 就绪 | Neo4j + Qdrant + Stanza 已加载")

    PRESET_CASES = {
        "自定义": "",
        "Case 1: 多跳级联": "帮助反派的那个教授教什么课？",
        "Case 2: 单跳查询": "Hermione保护了谁？",
        "Case 3: 概念检索": "Severus Snape教哪一门特定的科目？",
        "对抗: 中英混杂": "Harry的best friend是谁？",
        "对抗: KG外实体": "Iron Man教什么课？",
    }

    c1, c2 = st.columns([1, 3])
    with c1:
        preset = st.selectbox("📋 预设用例", list(PRESET_CASES.keys()))
    with c2:
        query = st.text_input("🔍 输入问题:",
                              value=PRESET_CASES[preset] or "帮助反派的那个教授教什么课？")

    if st.button("▶️ 启动执行流", type="primary", use_container_width=True):
        col_l, col_r = st.columns([2, 1])
        with col_l:
            st.markdown("### 📜 执行日志 & State追踪")
            log_ph = st.empty()
            st.markdown("### 🖥️ 终端日志")
            sys_log_ph = st.empty()
        with col_r:
            st.markdown("### 🕸️ 拓扑路由高亮")
            g_ph = st.empty()

        with st.spinner("🔄 Agent 推理中..."):
            resp = run_streamed(query, st.session_state.app, log_ph, sys_log_ph, g_ph)

        st.markdown("---")
        st.subheader("✅ 最终输出")
        st.success(resp)
        st.info("💡 拓扑图展示 LangGraph 的 5 个节点：intent_analyzer → planner → validate → execute → synthesize")


if __name__ == "__main__":
    main()