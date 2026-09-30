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

"""Describes where the agent runs, as an OpenTelemetry `Resource`.

`get_gcp_resource` is the only entry point. It merges one resource per source of
truth, in precedence order, each built by its own function:

  1. `_get_common_resource` -- what is true of every deployment.
  2. `_get_agent_runtime_resource` -- the Agent Runtime deployment.
  3. `_get_otel_detected_resource` -- what the operator stated through
     `OTEL_RESOURCE_ATTRIBUTES` and `OTEL_SERVICE_NAME`.
  4. `_get_gcp_detected_resource` -- what the platform says it is: GKE, Cloud
     Run, GCE, App Engine.
  5. `_get_urn_enriched_resource` -- the Agent Registry URN naming whatever the
     four above turned out to describe.

Every one of them returns a resource that may be empty, so merging it is a
no-op and the caller never branches. A source that has nothing to say because
this is not its platform says so at debug level; one that should have had
something and did not warns, because that is a deployment the user can fix.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Mapping
from typing import NamedTuple
from typing import TYPE_CHECKING
import uuid

from opentelemetry.sdk.resources import OTELResourceDetector
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv._incubating.attributes.cloud_attributes import CLOUD_ACCOUNT_ID
from opentelemetry.semconv._incubating.attributes.cloud_attributes import CLOUD_AVAILABILITY_ZONE
from opentelemetry.semconv._incubating.attributes.cloud_attributes import CLOUD_PLATFORM
from opentelemetry.semconv._incubating.attributes.cloud_attributes import CLOUD_PROVIDER
from opentelemetry.semconv._incubating.attributes.cloud_attributes import CLOUD_REGION
from opentelemetry.semconv._incubating.attributes.cloud_attributes import CLOUD_RESOURCE_ID
from opentelemetry.semconv._incubating.attributes.cloud_attributes import CloudPlatformValues
from opentelemetry.semconv._incubating.attributes.cloud_attributes import CloudProviderValues
from opentelemetry.semconv._incubating.attributes.faas_attributes import FAAS_NAME
from opentelemetry.semconv._incubating.attributes.k8s_attributes import K8S_CLUSTER_NAME
from opentelemetry.semconv._incubating.attributes.k8s_attributes import K8S_DEPLOYMENT_NAME
from opentelemetry.semconv._incubating.attributes.k8s_attributes import K8S_NAMESPACE_NAME
from opentelemetry.semconv.attributes.service_attributes import SERVICE_INSTANCE_ID
from opentelemetry.semconv.attributes.service_attributes import SERVICE_NAME
from opentelemetry.semconv.attributes.service_attributes import SERVICE_VERSION
from opentelemetry.util.types import AttributeValue

if TYPE_CHECKING:
  # pylint: disable-next=g-import-not-at-top
  from google.auth.credentials import Credentials

logger = logging.getLogger("google_adk." + __name__)

# Not in any released semconv versions. Replace with an import when available
# in the lower bound of the dependency.
_MAIN_AGENT_ID = "gen_ai.main_agent.id"

# gcp.project_id predates the semconv GCP attributes and has no constant there.
_GCP_PROJECT_ID = "gcp.project_id"

# Kubernetes names a Deployment's pods `<deployment>-<replicaset hash>-<pod
# hash>`, and tells the container only the pod name.
_POD_NAME_HASHES = re.compile(r"-[a-z0-9]+-[a-z0-9]{5}$")
_K8S_NAMESPACE_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"

# How many projects one process may hold resolved identifiers for. An agent
# reports to one project, or to a handful when a template hook redirects
# telemetry, so this is a ceiling against unbounded growth rather than a size
# anyone should reach.
_PROJECT_CACHE_SIZE = 32


def get_gcp_resource(project_id: str | None = None) -> Resource:
  """Returns the OTel resource describing this deployment.

  Later sources win: the platform's own answer overrides what the operator
  configured, which overrides the defaults, because a resource that disagrees
  with the platform it runs on cannot be mapped to a monitored resource.

  Args:
    project_id: the project telemetry is reported to. Used for `gcp.project_id`
      and `cloud.account.id`, either of which the OTel detector may override.

  Returns:
    The resource. Never empty: it always carries at least a service instance.
  """
  resource = _get_common_resource(project_id)
  resource = resource.merge(_get_agent_runtime_resource(project_id))
  resource = resource.merge(_get_otel_detected_resource())
  resource = resource.merge(_get_gcp_detected_resource())
  # Enrichment reads the three above, so it has to see the merged result.
  return resource.merge(_get_urn_enriched_resource(resource))


def _get_common_resource(project_id: str | None) -> Resource:
  """Returns what holds for every deployment, on any platform.

  Args:
    project_id: the project telemetry is reported to, if it is known.
  """
  # Unique per process, so two replicas of one deployment stay distinguishable
  # after every other attribute has turned out identical.
  attributes: dict[str, AttributeValue] = {
      SERVICE_INSTANCE_ID: f"{uuid.uuid4().hex}-{os.getpid()}",
  }
  if project_id:
    attributes[_GCP_PROJECT_ID] = project_id
    attributes[CLOUD_ACCOUNT_ID] = project_id
  else:
    logger.warning(
        "No project could be determined, so telemetry will not name one. Set"
        " GOOGLE_CLOUD_PROJECT, or pass a project explicitly, if it should."
    )
  return Resource(attributes=attributes)


def _get_agent_runtime_resource(project_id: str | None) -> Resource:
  """Returns the resource describing an Agent Runtime deployment.

  Empty off Agent Runtime, which is every other platform in this module.

  Args:
    project_id: the project telemetry is reported to, if it is known.
  """
  if not (agent_engine_id := os.getenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", "")):
    logger.debug("Not on Agent Runtime; no deployment of its own to describe.")
    return Resource.get_empty()

  location = os.getenv("GOOGLE_CLOUD_AGENT_ENGINE_LOCATION") or os.getenv(
      "GOOGLE_CLOUD_LOCATION"
  )
  if not location:
    logger.warning(
        "On Agent Runtime with no location set, so this deployment cannot be"
        " named. Set GOOGLE_CLOUD_AGENT_ENGINE_LOCATION or"
        " GOOGLE_CLOUD_LOCATION."
    )
  attributes: dict[str, AttributeValue] = {
      CLOUD_PROVIDER: CloudProviderValues.GCP.value,
      CLOUD_PLATFORM: CloudPlatformValues.GCP_AGENT_ENGINE.value,
      SERVICE_NAME: agent_engine_id,
      SERVICE_VERSION: os.getenv(
          "GOOGLE_CLOUD_AGENT_ENGINE_RUNTIME_REVISION_ID", ""
      ),
      CLOUD_REGION: location or "",
  }
  if project_id and location:
    attributes[CLOUD_RESOURCE_ID] = (
        f"//aiplatform.googleapis.com/projects/{project_id}"
        f"/locations/{location}/reasoningEngines/{agent_engine_id}"
    )
  # `Resource.create` rather than `Resource`, to pick up the `telemetry.sdk.*`
  # attributes the Agent Runtime resource has always carried.
  return Resource.create(attributes=attributes)


def _get_otel_detected_resource() -> Resource:
  """Returns what the operator declared through the OTel environment.

  Empty unless `OTEL_RESOURCE_ATTRIBUTES` or `OTEL_SERVICE_NAME` is set, which
  is how a deployment states anything this module cannot detect for itself --
  `k8s.namespace.name` through the Kubernetes downward API, say.
  """
  # `detect()` is untyped upstream, annotate to satisfy `no-any-return`.
  resource: Resource = OTELResourceDetector().detect()
  return resource


def _get_gcp_detected_resource() -> Resource:
  """Returns the resource the GCP detector describes this platform with.

  Empty on Agent Runtime, where the detector would describe the infrastructure
  underneath the deployment and, merging last, bury the deployment itself.
  """
  if os.getenv("GOOGLE_CLOUD_AGENT_ENGINE_ID"):
    # On Agent Runtime, whose own resource describes the deployment better.
    return Resource.get_empty()

  try:
    # pylint: disable-next=g-import-not-at-top
    from opentelemetry.resourcedetector.gcp_resource_detector import GoogleCloudResourceDetector

    detector = GoogleCloudResourceDetector(raise_on_error=False)
    # `detect()` is untyped upstream, annotate to satisfy `no-any-return`.
    resource: Resource = detector.detect()
    return resource
  except ImportError:
    logger.warning(
        "Could not import"
        " opentelemetry.resourcedetector.gcp_resource_detector GCE, GKE or"
        " CloudRun related resource attributes may be missing"
    )
  return Resource.get_empty()


def _get_urn_enriched_resource(resource: Resource) -> Resource:
  """Returns this deployment's Agent Registry URN, as a resource of its own.

  The URN is the name Agent Registry knows the agent by, so telemetry carrying
  it can be joined to the registered agent. Agent Registry derives the same URN
  from the resource name at ingestion, which is why this reconstructs it rather
  than inventing one:
  https://docs.cloud.google.com/agent-registry/concepts#agent-identifier

  All or nothing: a URN missing a segment is a foreign key that joins to the
  wrong agent, so a segment that could not be detected is warned about and no
  URN is emitted at all.

  Args:
    resource: everything detected so far, which the URN is built out of.
  """
  attributes = resource.attributes or {}
  if _MAIN_AGENT_ID in attributes:
    # A runtime that injects the URN itself knows what it deployed; this can
    # only reconstruct it.
    logger.debug("The URN is already set; leaving it as it is.")
    return Resource.get_empty()

  # The workload comes first because it decides whether there is a URN at all:
  # it is None off the platforms Agent Registry names, and empty when one of
  # its own segments is missing.
  workload = _urn_workload_segment(attributes)
  if not workload:
    return Resource.get_empty()

  project_id = _str_attr(attributes, CLOUD_ACCOUNT_ID)
  # One lookup covers both project segments, which are the same number twice.
  if not (number := _fetch_project(project_id).number):
    logger.warning(
        "The number of project %r is unknown, so no agent URN is set.",
        project_id,
    )
    return Resource.get_empty()
  if not (location := _urn_location_segment(attributes)):
    logger.warning(
        "Neither a zone nor a region could be detected, so no agent URN is set."
    )
    return Resource.get_empty()

  urn = ":".join([
      "urn:agent",
      _urn_publisher_segment(number),
      _urn_project_segment(number),
      location,
      workload,
  ])
  return Resource(attributes={_MAIN_AGENT_ID: urn})


def _urn_publisher_segment(project_number: str) -> str:
  """Returns who published the agent: the project, by number.

  Args:
    project_number: the project number, which the caller has already checked.
  """
  return f"projects-{project_number}"


def _urn_project_segment(project_number: str) -> str:
  """Returns the project the agent is deployed in, by number.

  The publisher repeated: Agent Registry scopes an agent to the project that
  published it, and names both in the URN.

  Args:
    project_number: the project number, which the caller has already checked.
  """
  return f"projects:{project_number}"


def _urn_location_segment(attributes: Mapping[str, AttributeValue]) -> str:
  """Returns where the agent is deployed: a zone, or failing that a region.

  Only a zonal GKE cluster reports a zone. Every other platform here reports a
  region, which is the form Agent Registry documents.

  Args:
    attributes: everything detected about this deployment.
  """
  if zone := _str_attr(attributes, CLOUD_AVAILABILITY_ZONE):
    return f"zones:{zone}"
  if region := _str_attr(attributes, CLOUD_REGION):
    return f"locations:{region}"
  return ""


def _urn_workload_segment(
    attributes: Mapping[str, AttributeValue],
) -> str | None:
  """Returns the segments naming the workload, which differ per platform.

  Args:
    attributes: everything detected about this deployment.

  Returns:
    The segments, empty if part of the workload could not be detected, or None
    on a platform Agent Registry names no agent on at all.
  """
  platform = _str_attr(attributes, CLOUD_PLATFORM)
  if platform == CloudPlatformValues.GCP_AGENT_ENGINE.value:
    return _urn_agent_runtime_segment(attributes)
  if platform == CloudPlatformValues.GCP_KUBERNETES_ENGINE.value:
    return _urn_gke_segment(attributes)
  if platform == CloudPlatformValues.GCP_CLOUD_RUN.value:
    return _urn_cloud_run_segment(attributes)
  logger.debug(
      "Agent Registry names no agent on %s, so there is no URN to build.",
      platform or "this platform",
  )
  return None


def _urn_agent_runtime_segment(
    attributes: Mapping[str, AttributeValue],
) -> str:
  """Returns the segments naming an Agent Runtime instance, or empty if unknown.

  Args:
    attributes: everything detected about this deployment.
  """
  # The instance ID is what the resource names the service after.
  if not (agent_engine_id := _str_attr(attributes, SERVICE_NAME)):
    logger.warning(
        "The Agent Runtime instance ID is unknown, so no agent URN is set."
    )
    return ""
  return f"aiplatform:reasoningEngines:{agent_engine_id}"


def _urn_gke_segment(attributes: Mapping[str, AttributeValue]) -> str:
  """Returns the segments naming a GKE Deployment, or empty if one is unknown.

  The GCP detector reports the cluster but neither the namespace nor the
  Deployment, so those fall back to what the pod itself can say.

  Args:
    attributes: everything detected about this deployment.
  """
  cluster = _str_attr(attributes, K8S_CLUSTER_NAME)
  namespace = _str_attr(attributes, K8S_NAMESPACE_NAME) or _k8s_namespace()
  deployment = _str_attr(attributes, K8S_DEPLOYMENT_NAME) or _k8s_deployment()
  for what, value, attribute in (
      ("cluster", cluster, K8S_CLUSTER_NAME),
      ("namespace", namespace, K8S_NAMESPACE_NAME),
      ("Deployment", deployment, K8S_DEPLOYMENT_NAME),
  ):
    if not value:
      logger.warning(
          "The GKE %s is unknown, so no agent URN is set. Set %s in"
          " OTEL_RESOURCE_ATTRIBUTES to state it.",
          what,
          attribute,
      )
  # After the loop, so an operator missing all three hears about all three.
  if not all((cluster, namespace, deployment)):
    return ""
  return (
      f"container:clusters:{cluster}"
      f":k8s:namespaces:{namespace}"
      f":apps:deployments:{deployment}"
  )


def _urn_cloud_run_segment(attributes: Mapping[str, AttributeValue]) -> str:
  """Returns the segments naming a Cloud Run service, or empty if unknown.

  Only services: the GCP detector recognises Cloud Run by `K_CONFIGURATION`,
  which a job does not set, so a job never reaches this and its documented
  `run:jobs:` URN is not built.

  Args:
    attributes: everything detected about this deployment.
  """
  if not (service := _str_attr(attributes, FAAS_NAME)):
    logger.warning(
        "The Cloud Run service name is unknown, so no agent URN is set."
    )
    return ""
  return f"run:services:{service}"


def _k8s_namespace() -> str:
  """Returns the namespace this pod runs in, per its service account token."""
  try:
    with open(_K8S_NAMESPACE_PATH, encoding="utf-8") as namespace:
      return namespace.read().strip()
  except OSError:
    logger.debug("This pod has no service account namespace to read.")
    return ""


def _k8s_deployment() -> str:
  """Returns the Deployment this pod belongs to, read off the pod name.

  Trimming the two hashes off HOSTNAME is the only way back to the Deployment
  without talking to the API server, and it holds for Deployments alone. Set
  `k8s.deployment.name` in OTEL_RESOURCE_ATTRIBUTES to say it outright.
  """
  deployment, trimmed = _POD_NAME_HASHES.subn("", os.getenv("HOSTNAME", ""))
  if not trimmed:
    logger.debug("HOSTNAME is not a name a Deployment would have given a pod.")
    return ""
  return deployment


def _str_attr(attributes: Mapping[str, AttributeValue], key: str) -> str:
  """Returns a resource attribute, or empty if it is absent or not a string.

  Args:
    attributes: everything detected about this deployment.
    key: the attribute to read.
  """
  value = attributes.get(key)
  return value if isinstance(value, str) else ""


class _Project(NamedTuple):
  """The two identifiers of a project, neither `projects/`-prefixed.

  A field that could not be determined is empty, so callers take whichever
  identifier they need and check it.
  """

  id: str
  number: str


def _fetch_project(
    project: str | None,
    credentials: Credentials | None = None,
) -> _Project:
  """Returns both identifiers of a project, given either one of them.

  Best effort. A project ID passed in is always returned; the number only ever
  comes from Resource Manager, so it is empty whenever that lookup fails, even
  though the value passed in may well have been the number.

  Memoized, because the exporters want the ID and the resource wants the
  number, and one lookup answers both. Without it each caller would pay for a
  lookup that only one of them used to need. Only successful lookups are kept,
  so a Resource Manager blip at startup costs one incomplete answer rather than
  an incomplete answer for the life of the process. The key is the project
  alone, not the credentials: an agent that resolves one project under two
  identities gets the first answer twice, which is the same project either way.

  Args:
    project: the project ID or project number, in either form.
    credentials: credentials for the lookup. Application Default Credentials
      are used when this is omitted, or when the answer is already memoized
      from a call that passed different ones.

  Returns:
    Both identifiers, each empty if it could not be determined.
  """
  if not project:
    return _Project(id="", number="")
  if cached := _fetch_project.cache.get(project):
    return cached

  # Only the lookup below tells a number from one of the roughly 20k legacy
  # project IDs that are all digits, so a failure leaves the number unknown
  # rather than reading digits as one: the number is what the URN is built
  # from, and an ID there is the one value worse than no URN at all. The ID
  # carries no such trap, and every caller of it falls back to the input.
  known = _Project(id="" if project.isdecimal() else project, number="")

  try:
    # pylint: disable-next=g-import-not-at-top
    from google.cloud import resourcemanager_v3 as resourcemanager

    projects_client = resourcemanager.ProjectsClient(credentials=credentials)
    fetched = projects_client.get_project(name=f"projects/{project}")
    # `name` is `projects/{number}`; strip it so both fields read the same way.
    resolved = _Project(
        id=fetched.project_id, number=fetched.name.split("/", 1)[1]
    )
  except Exception:  # pylint: disable=broad-exception-caught
    logger.warning(
        "Failed to look up the project's other identifier. Your traces and"
        " logs may not be associated, and the agent URN may be incomplete. To"
        " fix this, consider enabling the resource manager API and redeploying"
        " your agent.",
        exc_info=True,
    )
    return known

  if len(_fetch_project.cache) >= _PROJECT_CACHE_SIZE:
    # Evict the oldest rather than clear: a process this busy is redirecting
    # telemetry per request, and the project it started with is the one it is
    # least likely to need next. Two threads can pick the same key to evict, so
    # the second pop has to tolerate a miss.
    _ = _fetch_project.cache.pop(next(iter(_fetch_project.cache)), None)
  _fetch_project.cache[project] = resolved
  return resolved


# Memoization lives on the function rather than in a `functools.cache`
# decorator, which would key on the credentials as well as the project and
# would memoize failures just as readily as answers.
_fetch_project.cache: dict[str, _Project] = {}
