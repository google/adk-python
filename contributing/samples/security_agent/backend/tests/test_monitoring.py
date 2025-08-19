from fastapi.testclient import TestClient
from backend.main import app as main_app
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from fastapi import FastAPI

def test_monitoring():
    # Set up in-memory exporter for testing
    tracer_provider = TracerProvider()
    trace.set_tracer_provider(tracer_provider)
    exporter = InMemorySpanExporter()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Create a new app instance for testing
    app = FastAPI()
    FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider)

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    client = TestClient(app)

    # Send a request to the health endpoint
    response = client.get("/health")
    assert response.status_code == 200

    # Check that a span was created
    spans = exporter.get_finished_spans()
    assert len(spans) > 0
    assert spans[0].name == "GET /health http send"
