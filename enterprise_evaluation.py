# -- coding: utf-8 --
""" 企业级智能体可观测治理、离线评估与在线哨兵系统 - enterprise_evaluation.py (Production Ready) """

import os
import json
import time
import html
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger("EnterpriseEvaluation")

class HyperparameterRegistry:
    """解决 rag_hyperparameters.json 配置文件未被现有代码使用的 Bug，增加热更新与并发安全"""
    DEFAULT_CONFIG = {
        "align_predicate_th": 0.82, "max_recall_score_gate": 0.80, "low_confidence_fuse_th": 0.60,
        "mcp_timeout_ms": 5000, "max_loop_depth": 3, "dense_retrieval_weight": 0.70,
        "sparse_retrieval_weight": 0.30, "temperature": 0.0
    }

    _config = {}
    _file_mtime = 0
    _lock = threading.Lock()

    @classmethod
    def load_parameters(cls, file_path: str = "rag_hyperparameters.json") -> Dict[str, Any]:
        with cls._lock:
            if os.path.exists(file_path):
                try:
                    current_mtime = os.path.getmtime(file_path)
                    # 【修复】支持热更新：只有文件修改时间变化时才重新读取，降低 IO 开销
                    if current_mtime > cls._file_mtime or not cls._config:
                        with open(file_path, "r", encoding="utf-8") as f:
                            user_cfg = json.load(f)
                        cls._config = {**cls.DEFAULT_CONFIG, **user_cfg}
                        cls._file_mtime = current_mtime
                        logger.info(f"⚙️ [Hyperparameter] 成功加载并覆盖参数配置自: {file_path}")
                except Exception as e:
                    logger.error(f"❌ [Hyperparameter] 参数文件加载失败, 使用默认配置: {e}")
                    cls._config = cls.DEFAULT_CONFIG.copy()
            else:
                cls._config = cls.DEFAULT_CONFIG.copy()
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(cls._config, f, indent=2, ensure_ascii=False)
                    logger.info(f"⚙️ [Hyperparameter] 已初始化默认配置文件至: {file_path}")
                except Exception as e:
                    logger.error(f"无法初始化配置文件: {e}")
            return cls._config

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        # 每次获取时检查是否需要热更新
        cls.load_parameters()
        with cls._lock:
            return cls._config.get(key, default)

class OfflineEvaluationSuite:
    """五维工业级离线指标评估套件"""

    def __init__(self, output_dir: str = "evaluation_reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results_db: List[Dict[str, Any]] = []

    def evaluate_run(self, query: str, final_state: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"📊 [Offline Eval] 正在审计查询: \"{query}\"")

        cache = final_state.get("resolved_cache", {})
        tool_call_log = final_state.get("tool_call_log", [])
        final_output = final_state.get("final_output", "")

        expected_answer = ground_truth.get("answer", "")
        answer_correctness = 0.0
        if expected_answer.lower() in final_output.lower():
            answer_correctness = 1.0
        elif any(alias.lower() in final_output.lower() for alias in ground_truth.get("aliases", [])):
            answer_correctness = 0.8

        has_graph_ctx = any(k.endswith("_graph_context") for k in cache.keys())
        has_full_ctx = any(k.endswith("_full_context") for k in cache.keys())
        evidence_score = 1.0 if (has_graph_ctx or has_full_ctx) else 0.0

        expected_tools = ground_truth.get("expected_tools", [])
        actual_tools = [t.get("tool") for t in tool_call_log]
        tool_match_count = sum(1 for t in actual_tools if t in expected_tools)
        tool_call_accuracy = (tool_match_count / max(1, len(expected_tools)))

        plan_steps = len(final_state.get("plan", []))
        completed_steps = final_state.get("step_idx", 0)
        hydration_rate = final_state.get("loop_audit_logs", {}).get("hydration_rate", 1.0)
        planning_score = (completed_steps / max(1, plan_steps)) * hydration_rate

        declared_confidence = final_state.get("max_recall_score", 0.5)
        calibration_error = abs(declared_confidence - answer_correctness)
        robustness_score = 1.0 - calibration_error

        eval_metrics = {
            "query": query,
            "answer_correctness": answer_correctness,
            "faithfulness_evidence": evidence_score,
            "tool_call_accuracy": tool_call_accuracy,
            "planning_hydration": planning_score,
            "robustness_calibration": robustness_score,
            "final_confidence": declared_confidence,
            "latency_ms": sum([t.get("elapsed", 0.0) for t in tool_call_log])
        }

        self.results_db.append(eval_metrics)
        return eval_metrics

    def generate_html_report(self, report_name: str = "agent_metrics_dashboard.html"):
        if not self.results_db: return

        total_cases = len(self.results_db)
        avg_correctness = sum(r["answer_correctness"] for r in self.results_db) / total_cases
        avg_faithfulness = sum(r["faithfulness_evidence"] for r in self.results_db) / total_cases
        avg_tool = sum(r["tool_call_accuracy"] for r in self.results_db) / total_cases
        avg_planning = sum(r["planning_hydration"] for r in self.results_db) / total_cases
        avg_robustness = sum(r["robustness_calibration"] for r in self.results_db) / total_cases

        # 【修复】引入 html.escape 防止 XSS 攻击和 HTML 结构破坏
        html_content = f"""
        <!DOCTYPE html>
        <html><head><meta charset="utf-8"><title>Enterprise RAG Agent Evaluation</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; margin: 40px; background: #f8f9fa; color: #333; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            .summary-container {{ display: flex; justify-content: space-between; margin-bottom: 30px; }}
            .summary-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); width: 18%; text-align: center; border-top: 4px solid #3498db; }}
            .summary-card h3 {{ margin: 0; color: #7f8c8d; font-size: 14px; }}
            .summary-card p {{ margin: 10px 0 0 0; font-size: 28px; font-weight: bold; color: #2c3e50; }}
            table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
            th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #eceff1; }}
            th {{ background-color: #34495e; color: white; }}
            tr:hover {{ background-color: #f1f5f9; }}
            .badge {{ padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
            .badge-green {{ background-color: #e8f5e9; color: #2e7d32; }}
            .badge-red {{ background-color: #ffebee; color: #c62828; }}
        </style></head><body>
            <h1>📊 Enterprise RAG Agent Offline Evaluation Dashboard</h1>
            <p><strong>测试用例数:</strong> {total_cases} | <strong>执行时间:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <div class="summary-container">
                <div class="summary-card" style="border-top-color: #2ecc71;"><h3>Answer Correctness</h3><p>{avg_correctness * 100:.1f}%</p></div>
                <div class="summary-card" style="border-top-color: #9b59b6;"><h3>Faithfulness</h3><p>{avg_faithfulness * 100:.1f}%</p></div>
                <div class="summary-card" style="border-top-color: #f1c40f;"><h3>Tool Accuracy</h3><p>{avg_tool * 100:.1f}%</p></div>
                <div class="summary-card" style="border-top-color: #3498db;"><h3>Planning Hydration</h3><p>{avg_planning * 100:.1f}%</p></div>
                <div class="summary-card" style="border-top-color: #e74c3c;"><h3>Robustness</h3><p>{avg_robustness * 100:.1f}%</p></div>
            </div>
            <h2>📋 Test Case Detailed Audits</h2>
            <table><thead><tr>
                <th>Query</th><th>Correctness</th><th>Faithfulness</th><th>Tool Accuracy</th><th>Hydration</th><th>Calibration</th><th>Latency</th>
            </tr></thead><tbody>
        """

        for r in self.results_db:
            # 【修复】对所有动态插入的文本进行 HTML 转义
            safe_query = html.escape(str(r['query']))
            badge_class = 'badge-green' if r['answer_correctness'] >= 0.8 else 'badge-red'

            html_content += f"""
                <tr>
                    <td style="font-weight: 500; max-width: 300px; word-wrap: break-word;">{safe_query}</td>
                    <td><span class="badge {badge_class}">{r['answer_correctness'] * 100:.0f}%</span></td>
                    <td>{r['faithfulness_evidence'] * 100:.0f}%</td>
                    <td>{r['tool_call_accuracy'] * 100:.0f}%</td>
                    <td>{r['planning_hydration'] * 100:.0f}%</td>
                    <td>{r['robustness_calibration'] * 100:.0f}%</td>
                    <td>{r['latency_ms']:.1f}ms</td>
                </tr>
            """

        html_content += "</tbody></table></body></html>"

        file_path = os.path.join(self.output_dir, report_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"💾 [Offline Eval] 报告生成成功! 路径: {file_path}")

class OnlineTelemetrySentinel:
    """对线上/压测中的每次请求进行微秒级强阻断轻量校验"""

    @staticmethod
    def inspect_tool_contract(step: Dict[str, Any], params: Dict[str, Any], required_keys: List[str] = None) -> Tuple[bool, Optional[str]]:
        """
        【轻量契约验证】：检验 Tool Call 运行时参数是否合规。
        【修复】：使其通用化，支持通过 required_keys 动态指定必填参数，不再硬编码 subj/obj。
        """
        step_id = step.get("step_id", "unknown")

        # 默认图谱实体链接校验
        if required_keys is None:
            required_keys = ["subj", "pred"]

        for key in required_keys:
            val = params.get(key)
            if val is None or str(val).strip() == "" or str(val).strip().lower() == "none":
                err_msg = f"Contract Breach in step {step_id}: Parameter '{key}' is null or empty."
                return False, err_msg

        return True, None

    @staticmethod
    def audit_online_degradation(step_id: str, matches_len: int, hydration_rate: float, execution_log: List[str], webhook_url: str = None):
        """检测系统是否发生了严重的“静默退化”并打印在线警报"""
        alerts = []

        if hydration_rate < 0.3:
            msg = \
                (f"🚨 [Online Alert] 【水合率坍塌报警】 step: {step_id} | 当前动态变量水合率仅为 {hydration_rate * 100:.1f}%，"
                f"可能发生大量指代消解丢失！")
            logger.critical(msg)
            alerts.append(msg)

        if matches_len == 0:
            msg = f"⚠️ [Online Telemetry] 【图检索落空警告】 step: {step_id} | Cypher 关系边物理匹配无记录。"
            logger.warning(msg)
            alerts.append(msg)

        # 【修复】增加企业级 Webhook 告警扩展点
        if alerts and webhook_url:
            try:
                import requests
                requests.post(webhook_url, json={"text": "\n".join(alerts)}, timeout=2)
            except Exception as e:
                logger.error(f"Webhook 告警发送失败: {e}")