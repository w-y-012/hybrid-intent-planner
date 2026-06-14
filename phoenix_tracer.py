import os, logging, warnings
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

warnings.filterwarnings("ignore", message=".*__path__.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def setup_phoenix_tracer(port=6006):
    """配置 OpenTelemetry 并启动 Phoenix 导出器"""
    endpoint = f"http://localhost:{port}/v1/traces"

    # 1. 创建 TracerProvider
    resource = Resource.create({"service.name": "plan-execute-agent", "deployment.environment": "dev"})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    # 2. 添加 OTLP 导出器 (指向 Phoenix)
    try:
        otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter, max_export_batch_size=64))
        logging.info(f"✅ 已配置 Phoenix OTLP 导出器: {endpoint}")
    except Exception as e:
        logging.warning(f"⚠️ OTLP 导出器配置失败: {e}")

    # 3. 添加控制台导出器 (调试兜底，确保你能在终端看到 Span)
    console_exporter = ConsoleSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(console_exporter, max_export_batch_size=1))

    # 4. 激活 LangChain 插桩
    LangChainInstrumentor().instrument()
    logging.info("✅ LangChain Instrumentation 已激活 | Trace 数据将输出至终端 & Phoenix")


# 模块加载时自动执行
setup_phoenix_tracer()
