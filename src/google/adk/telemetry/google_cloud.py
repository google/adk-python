# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import enum
import logging
import os
import sys
from typing import Callable
from typing import cast
from typing import Optional
from typing import TYPE_CHECKING
import uuid

import google.auth
from google.auth.transport import mtls
from opentelemetry.sdk._logs import LogRecordProcessor
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor
from opentelemetry.sdk.metrics.export import MetricReader
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import OTELResourceDetector
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.util.types import AttributeValue

from ._agent_engine import _get_agent_engine_metrics_setup
from ._agent_engine import telemetry_user_agent_headers
from ._agent_engine_metric_exporter import MIN_EXPORT_INTERVAL_MS
from .setup import OTelHooks

if TYPE_CHECKING:
  from google.auth.credentials import Credentials
  from google.auth.transport.requests import AuthorizedSession
  from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

logger = logging.getLogger("google_adk." + __name__)

# cloud.resource_id is only defined in the private _incubating semconv package
# today; switch to the stable opentelemetry.semconv.attributes definition once
# the dependency floor is bumped past its promotion.
CLOUD_RESOURCE_ID = "cloud.resource_id"

_GCP_LOG_NAME_ENV_VARIABLE_NAME = "GOOGLE_CLOUD_DEFAULT_LOG_NAME"
_DEFAULT_LOG_NAME = "adk-otel"

_DEFAULT_TELEMETRY_TRACES_ENPOINT = "https://telemetry.googleapis.com/v1/traces"
_DEFAULT_MTLS_TELEMETRY_TRACES_ENPOINT = (
    "https://telemetry.mtls.googleapis.com/v1/traces"
)
_DEFAULT_TELEMETRY_METRICS_ENDPOINT = (
    "https://telemetry.googleapis.com/v1/metrics"
)
_DEFAULT_MTLS_TELEMETRY_METRICS_ENDPOINT = (
    "https://telemetry.mtls.googleapis.com/v1/metrics"
)


class _MtlsEndpoint(enum.Enum):
  """The mTLS endpoint setting."""

  AUTO = "auto"
  ALWAYS = "always"
  NEVER = "never"


def get_gcp_exporters(
    enable_cloud_tracing: bool = False,
    enable_cloud_metrics: bool = False,
    enable_cloud_logging: bool = False,
    google_auth: Optional[tuple[Credentials, str]] = None,
) -> OTelHooks:
  """Returns GCP OTel exporters to be used in the app.

  Args:
    enable_tracing: whether to enable tracing to Cloud Trace.
    enable_metrics: whether to enable reporting metrics to Cloud Monitoring.
    enable_logging: whether to enable sending logs to Cloud Logging.
    google_auth: optional custom credentials and project_id. google.auth.default() used when this is omitted.
  """

  credentials, project_id = (
      google_auth if google_auth is not None else google.auth.default()
  )
  if os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID"):
    # Try to convert project number to project ID to associate logs with traces.
    try:
      from google.cloud import resourcemanager_v3 as resourcemanager

      projects_client = resourcemanager.ProjectsClient(credentials=credentials)
      project = projects_client.get_project(name=f"projects/{project_id}")
      project_id = project.project_id
    except Exception:
      logger.warning(
          "Failed to convert project number to project ID. Your traces and logs"
          " may not be associated. To fix this, consider enabling the resource"
          " manager API and redeploying your agent.",
          exc_info=True,
      )
  if TYPE_CHECKING:
    credentials = cast(Credentials, credentials)
    project_id = cast(str, project_id)

  if not project_id:
    logger.warning(
        "Cannot determine GCP Project. OTel GCP Exporters cannot be set up."
        " Please make sure to log into correct GCP Project."
    )
    return OTelHooks()

  span_processors: list[SpanProcessor] = []
  if enable_cloud_tracing:
    span_processor = _get_gcp_span_exporter(credentials)
    span_processors.append(span_processor)

  metric_readers: list[MetricReader] = []
  if enable_cloud_metrics:
    if reader := _get_gcp_metrics_exporter((credentials, project_id)):
      metric_readers.append(reader)
    if agent_engine_metrics := _get_agent_engine_metrics_setup():
      span_processors.append(agent_engine_metrics.span_processor)

  log_record_processors: list[LogRecordProcessor] = []
  if enable_cloud_logging:
    logs_exporter = _get_gcp_logs_exporter(
        project_id=project_id,
    )
    if logs_exporter:
      log_record_processors.append(logs_exporter)

  return OTelHooks(
      span_processors=span_processors,
      metric_readers=metric_readers,
      log_record_processors=log_record_processors,
  )


def _get_gcp_span_exporter(credentials: Credentials) -> SpanProcessor:
  """Adds OTEL span exporter to telemetry.googleapis.com"""

  from google.auth.transport.requests import AuthorizedSession
  from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

  session = AuthorizedSession(credentials=credentials)

  endpoint = _get_telemetry_endpoint(
      session,
      _DEFAULT_TELEMETRY_TRACES_ENPOINT,
      _DEFAULT_MTLS_TELEMETRY_TRACES_ENPOINT,
  )

  return BatchSpanProcessor(
      OTLPSpanExporter(
          session=session,
          endpoint=endpoint,
          headers=telemetry_user_agent_headers(),
      )
  )


def _get_gcp_otlp_metric_exporter(
    google_auth: tuple[Credentials, str] | None = None,
) -> OTLPMetricExporter | None:
  """Returns a raw OTLP push metric exporter to telemetry.googleapis.com.

  This is the default GCP metric exporter (over Cloud Monitoring). It returns a
  bare push exporter so any metric reader can drain it -- a periodic reader for
  the local ``adk web`` path, or the request-driven reader on Agent Engine's
  request-billed runtime. Returns None if the OTLP exporter package is
  unavailable.

  Args:
    google_auth: optional custom credentials and project_id.
      google.auth.default() is used when this is omitted.
  """
  try:
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
  except (ImportError, AttributeError):
    logger.warning(
        "opentelemetry-exporter-otlp-proto-http is not installed; request-path"
        " metric export is disabled."
    )
    return None

  from google.auth.transport.requests import AuthorizedSession

  credentials, _ = (
      google_auth if google_auth is not None else google.auth.default()
  )
  session = AuthorizedSession(credentials=credentials)
  endpoint = _get_telemetry_endpoint(
      session,
      _DEFAULT_TELEMETRY_METRICS_ENDPOINT,
      _DEFAULT_MTLS_TELEMETRY_METRICS_ENDPOINT,
  )
  return OTLPMetricExporter(
      session=session,
      endpoint=endpoint,
      headers=telemetry_user_agent_headers(),
  )


def _get_telemetry_endpoint(
    session: AuthorizedSession, default_endpoint: str, mtls_endpoint: str
) -> str:
  """Configures the session for mTLS if enabled and returns the endpoint.

  Args:
    session: The AuthorizedSession to (maybe) configure for mTLS in place.
    default_endpoint: The plain telemetry.googleapis.com endpoint.
    mtls_endpoint: The telemetry.mtls.googleapis.com endpoint.

  Returns:
    The effective endpoint to export to.
  """
  if not _use_client_cert_effective():
    return default_endpoint
  client_cert_source = (
      mtls.default_client_cert_source()
      if mtls.has_default_client_cert_source()
      else None
  )
  session.configure_mtls_channel()
  return _get_api_endpoint(
      client_cert_source,
      default_endpoint=default_endpoint,
      mtls_endpoint=mtls_endpoint,
  )


def _get_gcp_metrics_exporter(
    google_auth: tuple[Credentials, str],
) -> MetricReader | None:
  """Returns the metric reader to install, or None if metrics are unavailable.

  On Agent Engine this is the request-driven reader (background export is
  starved by the request-billed runtime); elsewhere a periodic reader over the
  default OTLP metric exporter.
  """
  if agent_engine_metrics := _get_agent_engine_metrics_setup():
    return agent_engine_metrics.reader
  exporter = _get_gcp_otlp_metric_exporter(google_auth=google_auth)
  if exporter is None:
    return None
  return PeriodicExportingMetricReader(
      exporter,
      export_interval_millis=MIN_EXPORT_INTERVAL_MS,
  )


def _get_gcp_logs_exporter(
    project_id: str,
) -> LogRecordProcessor | None:
  if os.getenv("GOOGLE_CLOUD_AGENT_ENGINE_ID"):
    return _get_agent_engine_logs_exporter(
        project_id=project_id,
    )

  from opentelemetry.exporter.cloud_logging import CloudLoggingExporter

  default_log_name = os.environ.get(
      _GCP_LOG_NAME_ENV_VARIABLE_NAME, _DEFAULT_LOG_NAME
  )
  return BatchLogRecordProcessor(
      CloudLoggingExporter(
          project_id=project_id, default_log_name=default_log_name
      ),
  )


def _detect_cloud_resource_id(project_id: str | None) -> Optional[str]:
  """Detects the cloud resource ID."""
  location = os.getenv("GOOGLE_CLOUD_AGENT_ENGINE_LOCATION") or os.getenv(
      "GOOGLE_CLOUD_LOCATION"
  )
  agent_engine_id = os.getenv("GOOGLE_CLOUD_AGENT_ENGINE_ID")
  if project_id and location and agent_engine_id:
    return (
        f"//aiplatform.googleapis.com/projects/{project_id}"
        f"/locations/{location}/reasoningEngines/{agent_engine_id}"
    )
  return None


def get_gcp_resource(project_id: Optional[str] = None) -> Resource:
  """Returns OTEL with attributes specified in the following order (attributes specified later, overwrite those specified earlier):
  1. Populates gcp.project_id attribute from the project_id argument if present.
  2. OTELResourceDetector populates resource labels from environment variables like OTEL_SERVICE_NAME and OTEL_RESOURCE_ATTRIBUTES.
  3. GCP detector adds attributes corresponding to a correct monitored resource if ADK runs on one of supported platforms (e.g. GCE, GKE, CloudRun).

  Args:
    project_id: project id to fill out as `gcp.project_id` on the OTEL resource.
    This may be overwritten by OTELResourceDetector, if `gcp.project_id` is present in `OTEL_RESOURCE_ATTRIBUTES` env var.
  """
  agent_engine_id = os.getenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", "")
  cloud_resource_id = _detect_cloud_resource_id(project_id=project_id)
  resource_attributes: dict[str, AttributeValue] = {
      "cloud.provider": "gcp",
      "cloud.platform": "gcp.agent_engine",
      "service.name": agent_engine_id,
      "service.version": os.getenv(
          "GOOGLE_CLOUD_AGENT_ENGINE_RUNTIME_REVISION_ID", ""
      ),
      "service.instance.id": f"{uuid.uuid4().hex}-{os.getpid()}",
      "cloud.region": (
          os.getenv("GOOGLE_CLOUD_AGENT_ENGINE_LOCATION", "")
          or os.getenv("GOOGLE_CLOUD_LOCATION", "")
      ),
  }
  if project_id is not None:
    resource_attributes["gcp.project_id"] = project_id
    resource_attributes["cloud.account.id"] = project_id
  if cloud_resource_id is not None:
    resource_attributes[CLOUD_RESOURCE_ID] = cloud_resource_id

  if agent_engine_id:
    resource = Resource.create(attributes=resource_attributes).merge(
        OTELResourceDetector().detect()
    )
    return resource

  resource = Resource(
      attributes={"gcp.project_id": project_id}
      if project_id is not None
      else {}
  )
  resource = resource.merge(OTELResourceDetector().detect())
  try:
    from opentelemetry.resourcedetector.gcp_resource_detector import GoogleCloudResourceDetector

    resource = resource.merge(
        GoogleCloudResourceDetector(raise_on_error=False).detect()
    )
  except ImportError:
    logger.warning(
        "Cloud not import opentelemetry.resourcedetector.gcp_resource_detector"
        " GCE, GKE or CloudRun related resource attributes may be missing"
    )
  return resource


def _get_api_endpoint(
    client_cert_source: Callable[[], tuple[bytes, bytes]] | None = None,
    default_endpoint: str = _DEFAULT_TELEMETRY_TRACES_ENPOINT,
    mtls_endpoint: str = _DEFAULT_MTLS_TELEMETRY_TRACES_ENPOINT,
) -> str:
  """Returns API endpoint based on mTLS configuration and cert availability.

  Args:
      client_cert_source: A callable that returns the client certificate and
        key, or None.
      default_endpoint: The endpoint to use without mTLS.
      mtls_endpoint: The endpoint to use with mTLS.

  Returns:
      str: The API endpoint to be used.
  """
  use_mtls_endpoint_str = os.getenv(
      "GOOGLE_API_USE_MTLS_ENDPOINT", _MtlsEndpoint.AUTO.value
  ).lower()

  try:
    use_mtls_endpoint = _MtlsEndpoint(use_mtls_endpoint_str)
  except ValueError:
    logger.warning(
        "Environment variable `GOOGLE_API_USE_MTLS_ENDPOINT` must be one of "
        "%s. Defaulting to %s.",
        [e.value for e in _MtlsEndpoint],
        _MtlsEndpoint.AUTO.value,
    )
    use_mtls_endpoint = _MtlsEndpoint.AUTO

  if (use_mtls_endpoint is _MtlsEndpoint.ALWAYS) or (
      use_mtls_endpoint is _MtlsEndpoint.AUTO and client_cert_source
  ):
    return mtls_endpoint

  return default_endpoint


def _use_client_cert_effective() -> bool:
  """Returns whether client certificate should be used for mTLS.

  This checks if the google-auth version supports should_use_client_cert
  automatic mTLS enablement. Alternatively, it reads from the
  GOOGLE_API_USE_CLIENT_CERTIFICATE env var.

  Returns:
      bool: whether client certificate should be used for mTLS.
  """
  try:
    return bool(mtls.should_use_client_cert())
  except (ImportError, AttributeError):
    use_client_cert_str = os.getenv(
        "GOOGLE_API_USE_CLIENT_CERTIFICATE", "false"
    ).lower()
    if use_client_cert_str not in ("true", "false"):
      logger.warning(
          "Environment variable `GOOGLE_API_USE_CLIENT_CERTIFICATE` must be"
          " either `true` or `false`"
      )
    return use_client_cert_str == "true"


def _get_agent_engine_logs_exporter(
    *,
    project_id: str,
) -> LogRecordProcessor | None:
  """Configures logging for Agent Engine.

  Args:
    project_id: Project to which to write logs.
  """
  try:
    from opentelemetry.exporter import cloud_logging
  except (ImportError, AttributeError):
    logger.warning(
        "%s is not installed. Please call 'pip install %s'.",
        "opentelemetry-exporter-gcp-logging",
        "opentelemetry-exporter-gcp-logging",
    )
    logger.warning(
        "proceeding with logging disabled because not all packages for"
        " logging have been installed"
    )
    return None

  class _SimpleLogRecordProcessor(SimpleLogRecordProcessor):

    def force_flush(
        self, timeout_millis: int = 30000
    ) -> bool:  # pylint: disable=no-self-use
      _ = sys.stdout.flush()
      _ = sys.stderr.flush()
      return super().force_flush()

  return _SimpleLogRecordProcessor(
      cloud_logging.CloudLoggingExporter(
          project_id=project_id,
          default_log_name=os.getenv(
              "GCP_DEFAULT_LOG_NAME", "adk-on-agent-engine"
          ),
          structured_json_file=sys.stdout,
      ),
  )
