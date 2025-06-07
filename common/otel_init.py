import logging
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter


def init_otel(service_name: str = "osiris-dgm-kernel"):
    """Initializes OpenTelemetry tracing."""
    try:
        provider = TracerProvider()
        processor = BatchSpanProcessor(OTLPSpanExporter())
        provider.add_span_processor(processor)
        trace_api.set_tracer_provider(provider)
        logging.info("OpenTelemetry initialized.")
        return trace_api.get_tracer(__name__)
    except Exception as e:
        logging.error(f"Failed to initialize OpenTelemetry: {e}")
        return None
