import os
import logging

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor

# Store the original FastAPIInstrumentor.instrument_app method
_original_fastapi_instrument_app = FastAPIInstrumentor.instrument_app
_original_fastapi_uninstrument_app = FastAPIInstrumentor.uninstrument_app


def init_otel(app=None):
    """
    Initializes OpenTelemetry if the OTEL_EXPORTER_OTLP_ENDPOINT environment variable is set.
    Sets up OTLP exporter, FastAPI instrumentation (if app is provided), and logging instrumentation.
    """
    otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

    if not otel_endpoint:
        logging.info(
            "OTEL_EXPORTER_OTLP_ENDPOINT not set, OpenTelemetry will not be initialized."
        )
        return

    logging.info(f"Initializing OpenTelemetry with OTLP endpoint: {otel_endpoint}")

    try:
        provider = TracerProvider()
        # Ensure OTLPSpanExporter endpoint is correctly formatted
        otlp_exporter = OTLPSpanExporter(endpoint=f"{otel_endpoint}/v1/traces")
        processor = BatchSpanProcessor(otlp_exporter)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

        # Instrument FastAPI if an app instance is provided
        if app:
            # Check if already instrumented to prevent double instrumentation in some dev environments (like uvicorn --reload)
            if not getattr(app, "_otel_instrumented", False):
                FastAPIInstrumentor.instrument_app(app)
                app._otel_instrumented = True  # Mark as instrumented
            else:
                logging.info("FastAPI app already instrumented. Skipping.")
        else:
            # If no app is provided on init, ensure that global instrumentation is set up if needed by other instrumentors
            # This part might need adjustment based on how non-FastAPI parts of the services are structured.
            # For now, we assume FastAPIInstrumentor is the primary concern for app-specific instrumentation.
            pass

        # Instrument logging
        # Check if already instrumented
        if not getattr(LoggingInstrumentor, "_otel_instrumented", False):
            LoggingInstrumentor().instrument()
            LoggingInstrumentor._otel_instrumented = True  # Mark as instrumented
        else:
            logging.info("Logging already instrumented. Skipping.")

        logging.info("OpenTelemetry initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize OpenTelemetry: {e}", exc_info=True)


if __name__ == "__main__":
    # Example usage (for testing purposes)
    logging.basicConfig(level=logging.INFO)
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"
    init_otel()

    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("example-span"):
        logging.info("This is a log message within a span.")
        print("Example span created. Check your OTLP collector.")
