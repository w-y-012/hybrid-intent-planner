# quiet_logging.py
"""
统一日志清理模块 - 在程序入口第一行导入
"""

import os
import sys
import logging
import warnings


def silence_all_logging():
    """静默所有第三方库日志，但保留应用日志"""

    # 静默列表
    silent_libs = [
        'httpx', 'httpcore', 'urllib3', 'requests',
        'sentence_transformers', 'stanza', 'transformers',
        'torch', 'torchvision', 'torchaudio',
        'neo4j', 'qdrant_client', 'langchain', 'langgraph',
        'openai', 'pypdf', 'nltk', 'PIL',
        'matplotlib', 'tensorflow', 'keras',
        'grpc', 'asyncio', 'aiohttp',
    ]

    for lib in silent_libs:
        logger = logging.getLogger(lib)
        logger.setLevel(logging.ERROR)
        logger.propagate = False
        # 移除所有处理器
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    # 关闭警告
    warnings.filterwarnings("ignore")

    # 禁用 tqdm
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 【修复】不要覆盖根日志器的配置
    # 删除或注释掉这行：
    # logging.basicConfig(level=logging.WARNING, force=True)

    # 可选：设置一个合理的默认级别
    # 但让各个模块自己配置更好


# 自动执行
silence_all_logging()