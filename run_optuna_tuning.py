# -*- coding: utf-8 -*-
"""
超参自动标定器 - run_optuna_tuning.py
职责:
1. 运行贝叶斯搜索算法优化 RAG 执行器的所有决策参数与阈值。
2. 运行 Golden ground-truth 数据集，在 RAGAS 指标和 LoopAuditor 约束下进行评估。
3. 将最优标定参数自动保存。
"""
import quiet_logging  # 必须是第一行
import os
import re
import json
import uuid
import optuna
import logging
from typing import Dict, List, Any

# 引入 Agent 实体
from executor_agent import create_rag_graph, init_eval_agent_state
from functions_for_pipeline import RAGHyperparams, hyperparams

logging.getLogger("optuna").setLevel(logging.WARNING)
print("\033[1;36m=== 🚀 启动 RAG 决策阈值自动寻找最优解调优器 ===\033[0m")

# 1. 声明黄金数据集 (Golden Ground-Truth Evaluation Dataset)
GOLDEN_EVAL_DATASET = [
    {
        "query": "帮助反派的那个教授教什么课？",
        "expected_entity": "Quirinus Quirrell",
        "expected_relation": "TEACHES"
    },
    {
        "query": "谁击败了奇洛教授？",
        "expected_entity": "Harry Potter",
        "expected_relation": "DEFEATS"
    }
]

app_agent = create_rag_graph()


def objective(trial) -> float:
    # 2. 声明贝叶斯搜索空间
    trial_params = {
        "align_predicate_th": trial.suggest_float("align_predicate_th", 0.60, 0.85),
        "entity_link_th": trial.suggest_float("entity_link_th", 0.65, 0.90),
        "bridge_th": trial.suggest_float("bridge_th", 0.70, 0.95),
        "min_recall_gate": trial.suggest_float("min_recall_gate", 0.40, 0.70),

        "w_main_predicate_root": trial.suggest_int("w_main_predicate_root", 15, 30),
        "w_modifier_predicate": trial.suggest_int("w_modifier_predicate", -25, -10),
        "w_arg_bonus": trial.suggest_int("w_arg_bonus", 1, 5)
    }

    # 3. 将采样参数实时热载入到 RAG 运行时全局单例
    for k, v in trial_params.items():
        setattr(hyperparams, k, v)

    # 4. 并发/轮询运行测试集
    total_loss = 0.0
    for case in GOLDEN_EVAL_DATASET:
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        state_input = init_eval_agent_state(case["query"], tuning_mode=True)

        try:
            res_state = app_agent.invoke(state_input, config=config)

            # 读取 RAGAS 评估裁判得分与 LoopAuditor 指标
            faithfulness = res_state["ragas_metrics"]["faithfulness"]
            answer_relevance = res_state["ragas_metrics"]["answer_relevance"]

            loop_logs = res_state["loop_audit_logs"]
            step_count = res_state["step_idx"]

            oscillation_penalty = 5.0 if loop_logs["oscillation_detected"] else 0.0
            hydration_rate = loop_logs["hydration_rate"]

            # 构建帕累托最优损失函数 (Minimization Objective)
            # 我们希望 Faithfulness -> 1.0, AnswerRelevance -> 1.0, Hydration -> 1.0, Steps -> 0.0
            loss = (
                    4.0 * (1.0 - faithfulness) +
                    2.5 * (1.0 - answer_relevance) +
                    2.0 * (1.0 - hydration_rate) +
                    0.5 * (step_count / 6.0) +
                    oscillation_penalty
            )
            total_loss += loss
        except Exception as e:
            # 运行崩溃惩罚分
            total_loss += 15.0

    return total_loss / len(GOLDEN_EVAL_DATASET)


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15)  # 运行15轮快速采样，标定帕累托最优配置

    print("\n\033[1;32m🏆 贝叶斯搜索调优完毕！推荐最优超参推荐列表:\033[0m")
    best_params = study.best_params
    print(json.dumps(best_params, indent=2))

    # 5. 将最优参数自动重写并持久化为 JSON 供底座及 Agent 在运行时自适应热载入
    final_hyperparams = RAGHyperparams(**best_params)
    final_hyperparams.save()
    print(f"\n⚡ 最优超参已自动写入磁盘：rag_hyperparameters.json")
    print("\033[1;36m========================================================\033[0m")
