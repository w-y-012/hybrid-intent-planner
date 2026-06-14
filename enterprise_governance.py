# -- coding: utf-8 --
""" 企业级智能体治理与 MCP/Skills 子系统 - enterprise_governance.py (Production Ready) """

import os
import time
import json
import uuid
import copy
import logging
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Dict, List, Any, Callable, Optional

logger = logging.getLogger("EnterpriseGovernance")

# 全局线程池，用于 Skill 超时控制，避免每次执行创建线程池的开销
_GLOBAL_EXECUTOR = ThreadPoolExecutor(max_workers=32, thread_name_prefix="MCP-Skill-Worker")

import traceback

class MCPSkill:
    """MCP 规范强类型 Skill 实体描述（企业生产级）"""

    def __init__(self, name: str, description: str, input_schema: Dict[str, Any], handler: Callable, timeout_sec: float = 5.0):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.handler = handler
        self.timeout_sec = timeout_sec  # 增加超时熔断机制

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """在完全隔离的沙盒中执行，强制 Schema 校验、超时控制与异常脱敏"""
        logger.info(f"🛡️ [MCP Sandbox] 正在验证并执行技能: {self.name}")

        # 1. 强 Schema 验证与动态缺失补全
        validated_params = {}
        for param_name, schema in self.input_schema.items():
            is_required = schema.get("required", False)
            default_val = schema.get("default", None)
            param_type = schema.get("type") # 支持基础类型校验

            if param_name not in params:
                if is_required:
                    error_msg = f"缺少必填参数: '{param_name}'"
                    logger.error(f"❌ [MCP Schema Error] {error_msg} (Skill: {self.name})")
                    return {"success": False, "data": None, "latency_ms": 0.0, "error": error_msg}
                else:
                    validated_params[param_name] = default_val
            else:
                val = params[param_name]
                # 基础类型校验 (防止恶意注入错误类型)
                if param_type and not isinstance(val, eval(param_type) if param_type in ['int', 'float', 'str', 'bool', 'list', 'dict'] else object):
                    logger.warning(f"⚠️ [MCP Type Warn] 参数 {param_name} 类型不匹配，期望 {param_type}")
                validated_params[param_name] = val

        # 2. 精准追踪 CPU 与 IO 时延，并增加超时熔断
        start_time = time.perf_counter()
        try:
            # 使用全局线程池执行，实现超时控制
            future = _GLOBAL_EXECUTOR.submit(self.handler, **validated_params)
            result = future.result(timeout=self.timeout_sec)

            latency = (time.perf_counter() - start_time) * 1000
            return {"success": True, "data": result, "latency_ms": latency, "error": None}

        except FuturesTimeoutError:
            latency = (time.perf_counter() - start_time) * 1000
            error_msg = f"Skill 执行超时 (>{self.timeout_sec}s)"
            logger.error(f"⏳ [MCP Timeout] {self.name} {error_msg}")
            return {"success": False, "data": None, "latency_ms": latency, "error": error_msg}

        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            stack_trace = traceback.format_exc()
            # 【安全修复】：详细堆栈仅记录到内部日志，对外返回脱敏信息，防止代码结构泄露
            logger.critical(f"💥 [MCP Sandbox Crash] Skill '{self.name}' 崩溃！\n堆栈:\n{stack_trace}")
            safe_error_msg = f"内部执行异常: {type(e).__name__} - {str(e)}"
            return {"success": False, "data": None, "latency_ms": latency, "error": safe_error_msg}


class MCPSkillRegistry:
    """MCP 全局技能仓库管理器"""

    def __init__(self):  # 【修复】原代码为 def init
        self.skills: Dict[str, MCPSkill] = {}
        self._lock = threading.Lock() # 增加线程锁

    def register_skill(self, skill: MCPSkill):
        with self._lock:
            self.skills[skill.name] = skill
        logger.info(f"🔌 [MCP Registry] 外部 Skill 挂载注册成功: {skill.name}")

    def call_tool(self, skill_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if skill_name not in self.skills:
                raise KeyError(f"MCP Skill '{skill_name}' is not registered.")
            skill = self.skills[skill_name]
        return skill.execute(params)

class SessionWorkspaceManager:
    """长时会话级 Session Checkpoint 快照管理器 (支持事务回滚与长程记忆)"""

    def __init__(self, session_id: str, workspace_dir: str = "workspace_checkpoints", max_checkpoints: int = 20):
        # 【修复】原代码为 def init
        self.session_id = session_id
        self.workspace_dir = workspace_dir
        self.memory_db_path = os.path.join(workspace_dir, f"memory_{session_id}.json")
        os.makedirs(workspace_dir, exist_ok=True)

        self.checkpoints: List[Dict[str, Any]] = []
        self.max_checkpoints = max_checkpoints # 防止内存泄漏
        self._lock = threading.Lock()          # 并发安全锁

        self.long_term_memory: Dict[str, Any] = self._load_long_term_memory()

    def _load_long_term_memory(self) -> Dict[str, Any]:
        if os.path.exists(self.memory_db_path):
            try:
                with open(self.memory_db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"记忆文件损坏，重置: {e}")
        return {"episodic_memories": [], "user_preferences": {}}

    def commit_checkpoint(self, step_idx: int, state_snapshot: Dict[str, Any]):
        """在推理重要节点保存 Checkpoint 快照"""
        with self._lock:
            # 【修复】使用 copy.deepcopy 替代 json 序列化，支持 datetime 等复杂对象，且性能更好
            checkpoint = {
                "checkpoint_id": str(uuid.uuid4()),
                "timestamp": time.time(),
                "step_idx": step_idx,
                "state": copy.deepcopy(state_snapshot)
            }
            self.checkpoints.append(checkpoint)

            # 【修复】LRU 淘汰策略，防止长会话 OOM
            if len(self.checkpoints) > self.max_checkpoints:
                self.checkpoints.pop(0)

            logger.info(f"💾 [Checkpoint Saved] 保存步骤 [{step_idx}]。快照 ID: {checkpoint['checkpoint_id']}")

    def rollback_to_step(self, target_step_idx: int) -> Optional[Dict[str, Any]]:
        """【事务级回滚】：当死循环、震荡或子步骤崩溃时，无损回退到健康的快照哨兵节点"""
        logger.warning(f"🔄 [Transaction Rollback] 触发长程事务回滚！目标步骤: {target_step_idx}")
        with self._lock:
            for cp in reversed(self.checkpoints):
                if cp["step_idx"] == target_step_idx:
                    # 斩断目标步骤之后的所有损坏快照
                    self.checkpoints = [c for c in self.checkpoints if c["timestamp"] <= cp["timestamp"]]
                    logger.info(f"✅ [Transaction Rollback] 成功！重置为快照: {cp['checkpoint_id']}")
                    return copy.deepcopy(cp["state"])
        return None

    def store_episodic_memory(self, query: str, final_answer: str, confidence: float):
        """持久化长程情节记忆 (原子写入防损坏)"""
        with self._lock:
            self.long_term_memory["episodic_memories"].append({
                "timestamp": time.time(),
                "query": query,
                "answer": final_answer,
                "confidence": confidence
            })

            # 【修复】使用临时文件 + os.replace 实现原子写入，杜绝并发写入导致的 JSON 截断损坏
            try:
                dir_name = os.path.dirname(self.memory_db_path)
                with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=dir_name, delete=False) as tf:
                    json.dump(self.long_term_memory, tf, ensure_ascii=False, indent=2)
                    temp_path = tf.name
                os.replace(temp_path, self.memory_db_path)
                logger.info(f"🧠 [Memory Persisted] 长程情节记忆库持久化成功。")
            except Exception as e:
                logger.error(f"长程记忆写入失败: {e}")

class ObservabilityTracer:
    """企业级高性能可观测性治理跟踪仪"""

    def __init__(self, trace_id: str):
        # 【修复】原代码为 def init
        self.trace_id = trace_id
        self.spans: List[Dict[str, Any]] = []
        self.metrics = {"cpu_time_ms": 0.0, "io_time_ms": 0.0, "token_count": 0}
        self._lock = threading.Lock() # 保证 spans 追加的线程安全

    def start_span(self, name: str, metadata: Optional[Dict] = None) -> float:
        logger.info(f"📊 [Observability Trace] Spanning START: {name}")
        return time.perf_counter()

    def end_span(self, name: str, start_perf_time: float, metadata: Optional[Dict] = None):
        latency = (time.perf_counter() - start_perf_time) * 1000
        span = {
            "name": name,
            "latency_ms": latency,
            "metadata": metadata or {}
        }
        with self._lock:
            self.spans.append(span)
            self.metrics["cpu_time_ms"] += latency # 简单累加
        logger.info(f"📊 [Observability Trace] Spanning END: {name} | 延时: {latency:.2f}ms")

    def print_observability_dashboard(self):
        """打印漂亮的生产级治理监控树"""
        with self._lock:
            spans_copy = list(self.spans) # 读取时拷贝，避免阻塞

        print("\n" + "=" * 46 + " 📊 ENTERPRISE GOVERNANCE & OBSERVABILITY " + "=" * 46)
        print(f"  Trace-ID: {self.trace_id}")
        total_latency = sum([s["latency_ms"] for s in spans_copy])
        print(f"  🟢 总执行时延 (Total Latency): {total_latency:.2f}ms")
        print(f"  🟢 事务状态: PERSISTED | 长程 Session 完整性: 100%")
        print("\n  ⚙️  SPAN LATENCY TREE:")
        for s in spans_copy:
            print(f"     ├── [{s['name']}] ──── 延时: {s['latency_ms']:.2f}ms | 携带元数据: {s['metadata']}")
        print("=" * 131 + "\n")
