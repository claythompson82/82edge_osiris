import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

def init_otel(service_name: str = "osiris-default-service"):
    """Initializes OpenTelemetry tracing and sets it as the global provider."""
    try:
        # Create a TracerProvider
        provider = TracerProvider()

        # Create an OTLP Span Exporter (assumes OTLP endpoint is configured via env vars)
        exporter = OTLPSpanExporter()

        # Create a BatchSpanProcessor and add the exporter to it
        processor = BatchSpanProcessor(exporter)

        # Add the processor to the provider
        provider.add_span_processor(processor)

        # Set the configured provider as the global provider
        trace.set_tracer_provider(provider)

        logging.info(f"OpenTelemetry initialized for service: {service_name}")

        # Return a tracer from the global provider
        return trace.get_tracer(service_name)

    except Exception as e:
        logging.error(f"Failed to initialize OpenTelemetry: {e}")
        return None
