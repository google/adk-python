from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.cloud_logging import CloudLoggingExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.cloud_monitoring import CloudMonitoringMetricsExporter
import logging

def setup_monitoring(app):
    # Set up trace provider
    trace.set_tracer_provider(TracerProvider())

    # Set up trace exporter
    trace.get_tracer_provider().add_span_processor(
        BatchSpanProcessor(CloudTraceSpanExporter())
    )

    # Set up metric provider
    reader = PeriodicExportingMetricReader(CloudMonitoringMetricsExporter())
    provider = MeterProvider(metric_readers=[reader])
    
    # Set up log provider
    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(CloudLoggingExporter()))
    handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)
    logging.getLogger().addHandler(handler)

    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app, meter_provider=provider)

    # Instrument requests
    RequestsInstrumentor().instrument()

    # Instrument logging
    LoggingInstrumentor().instrument(set_logging_format=True)
