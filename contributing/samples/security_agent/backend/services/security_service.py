import os
from google.adk import Agent
from opentelemetry import trace
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.context import attach, get_current

class SecurityService:
    def __init__(self):
        self.tracer = trace.get_tracer(__name__)
        try:
            # Ensure GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION are set for ADC
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'mgm-digitalconcierge')
            location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
            os.environ['GOOGLE_CLOUD_PROJECT'] = project_id
            os.environ['GOOGLE_CLOUD_LOCATION'] = location
            self.agent = Agent(
                model='gemini-2.0-flash-exp',
                name='security_agent',
            )
            print(f"✅ Vertex AI ADK Agent initialized for project: {project_id}")
        except Exception as e:
            print(f"❌ Failed to initialize Vertex AI ADK Agent: {e}")
            self.agent = None

    async def evaluate_vulnerability(self, text: str) -> dict:
        if not self.agent:
            return {"error": "ADK Agent not initialized."}

        with self.tracer.start_as_current_span("SecurityService.evaluate_vulnerability") as span:
            span.set_attribute("input.text_length", len(text))
            
            try:
                # Use the ADK agent to analyze the vulnerability
                prompt = f"Evaluate the following text for security vulnerabilities and suggest remediations: {text}"
                response = await self.agent.generate_content(prompt)
                
                result_text = ""
                for part in response.candidates[0].content.parts:
                    if part.text:
                        result_text += part.text
                        
                span.set_attribute("agent.response_length", len(result_text))
                span.set_status(trace.Status(trace.StatusCode.OK))
                return {"success": True, "evaluation": result_text}
            except Exception as e:
                span.set_status(trace.Status(trace.StatusCode.ERROR, description=str(e)))
                print(f"Error during vulnerability evaluation: {e}")
                return {"success": False, "error": str(e)}
