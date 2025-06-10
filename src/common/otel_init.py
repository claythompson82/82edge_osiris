import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

def init_otel(service_name: str = "osiris-default-service"):
    """Initializes OpenTelemetry tracing and sets it as the global provider."""
    try:
        provider = TracerProvider()
        exporter = OTLPSpanExporter()
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        logging.info(f"OpenTelemetry initialized for service: {service_name}")
        return trace.get_tracer(service_name)
    except Exception as e:
        logging.error(f"Failed to initialize OpenTelemetry: {e}")
        return None
