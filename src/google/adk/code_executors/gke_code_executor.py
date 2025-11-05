# Copyright 2025 Google LLC
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

import logging
import uuid

import kubernetes as k8s
from kubernetes.watch import Watch

from ..agents.invocation_context import InvocationContext
from .base_code_executor import BaseCodeExecutor
from .code_execution_utils import CodeExecutionInput
from .code_execution_utils import CodeExecutionResult

# Expose these for tests to monkeypatch.
client = k8s.client
config = k8s.config
ApiException = k8s.client.exceptions.ApiException

logger = logging.getLogger("google_adk." + __name__)


class GkeCodeExecutor(BaseCodeExecutor):
  """Executes Python code in a secure gVisor-sandboxed Pod on GKE.

  This executor securely runs code by dynamically creating a Kubernetes Job for
  each execution request. The user's code is mounted via a ConfigMap, and the
  Pod is hardened with a strict security context and resource limits.

  Key Features:
  - Sandboxed execution using the gVisor runtime.
  - Ephemeral, per-execution environments using Kubernetes Jobs.
  - Secure-by-default Pod configuration (non-root, no privileges).
  - Automatic garbage collection of completed Jobs and Pods via TTL.
  - Efficient, event-driven waiting using the Kubernetes watch API.

  RBAC Permissions:
  This executor requires a ServiceAccount with specific RBAC permissions. The
  Role granted to the ServiceAccount must include rules to manage Jobs,
  ConfigMaps, and Pod logs. Below is a minimal set of required permissions:

  rules:
  # For creating/deleting code ConfigMaps and patching ownerReferences
  - apiGroups: [""] # Core API Group
  resources: ["configmaps"]
  verbs: ["create", "delete", "get", "patch"]
  # For watching Job completion status
  - apiGroups: ["batch"]
  resources: ["jobs"]
  verbs: ["get", "list", "watch", "create", "delete"]
  # For retrieving logs from the completed Job's Pod
  - apiGroups: [""] # Core API Group
  resources: ["pods", "pods/log"]
  verbs: ["get", "list"]
  """

  namespace: str = "default"
  image: str = "python:3.11-slim"
  timeout_seconds: int = 300
  cpu_requested: str = "200m"
  mem_requested: str = "256Mi"
  # The maximum CPU the container can use, in "millicores". 1000m is 1 full CPU core.
  cpu_limit: str = "500m"
  mem_limit: str = "512Mi"

  kubeconfig_path: str | None = None
  kubeconfig_context: str | None = None

  _batch_v1: k8s.client.BatchV1Api
  _core_v1: k8s.client.CoreV1Api

  def __init__(
      self,
      kubeconfig_path: str | None = None,
      kubeconfig_context: str | None = None,
      **data,
  ):
    """Initializes the executor and the Kubernetes API clients.

    This constructor supports multiple authentication methods:
    1. Explicitly via a kubeconfig file path and context.
    2. Automatically via in-cluster service account (when running in GKE).
    3. Automatically via the default local kubeconfig file (~/.kube/config).
    """
    super().__init__(**data)
    self.kubeconfig_path = kubeconfig_path
    self.kubeconfig_context = kubeconfig_context

    if self.kubeconfig_path:
      try:
        logger.info(f"Using explicit kubeconfig from '{self.kubeconfig_path}'.")
        config.load_kube_config(
            config_file=self.kubeconfig_path, context=self.kubeconfig_context
        )
      except config.ConfigException as e:
        logger.error(
            f"Failed to load explicit kubeconfig from {self.kubeconfig_path}",
            exc_info=True,
        )
        raise RuntimeError(
            "Failed to configure Kubernetes client from provided path."
        ) from e
    else:
      try:
        config.load_incluster_config()
        logger.info("Using in-cluster Kubernetes configuration.")
      except config.ConfigException:
        try:
          logger.info(
              "In-cluster config not found. Falling back to default local"
              " kubeconfig."
          )
          config.load_kube_config()
        except config.ConfigException as e:
          logger.error(
              "Could not configure Kubernetes client automatically.",
              exc_info=True,
          )
          raise RuntimeError(
              "Failed to find any valid Kubernetes configuration."
          ) from e

    self._batch_v1 = client.BatchV1Api()
    self._core_v1 = client.CoreV1Api()

  def execute_code(
      self,
      invocation_context: InvocationContext,
      code_execution_input: CodeExecutionInput,
  ) -> CodeExecutionResult:
    """Orchestrates the secure execution of a code snippet on GKE."""
    job_name = f"adk-exec-{uuid.uuid4().hex[:10]}"
    configmap_name = f"code-src-{job_name}"

    try:
      # The execution process:
      # 1. Create a ConfigMap to mount LLM-generated code into the Pod.
      # 2. Create a Job that runs the code from the ConfigMap.
      # 3. Set the Job as the ConfigMap's owner for automatic cleanup.
      self._create_code_configmap(configmap_name, code_execution_input.code)
      job_manifest = self._create_job_manifest(
          job_name, configmap_name, invocation_context
      )
      created_job = self._batch_v1.create_namespaced_job(
          body=job_manifest, namespace=self.namespace
      )
      self._add_owner_reference(created_job, configmap_name)

      logger.info(
          f"Submitted Job '{job_name}' to namespace '{self.namespace}'."
      )
      logger.debug("Executing code:\n```\n%s\n```", code_execution_input.code)
      return self._watch_job_completion(job_name)

    except ApiException as e:
      logger.error(
          "A Kubernetes API error occurred during job"
          f" '{job_name}': {e.reason}",
          exc_info=True,
      )
      return CodeExecutionResult(stderr=f"Kubernetes API error: {e.reason}")
    except TimeoutError as e:
      logger.error(e, exc_info=True)
      logs = self._get_pod_logs(job_name)
      stderr = f"Executor timed out: {e}\n\nPod Logs:\n{logs}"
      return CodeExecutionResult(stderr=stderr)
    except Exception as e:
      logger.error(
          f"An unexpected error occurred during job '{job_name}': {e}",
          exc_info=True,
      )
      return CodeExecutionResult(
          stderr=f"An unexpected executor error occurred: {e}"
      )

  def _create_job_manifest(
      self,
      job_name: str,
      configmap_name: str,
      invocation_context: InvocationContext,
  ) -> k8s.client.V1Job:
    """Creates the complete V1Job object with security best practices."""
    # Define the container that will run the code.
    container = k8s.client.V1Container(
        name="code-runner",
        image=self.image,
        command=["python3", "/app/code.py"],
        volume_mounts=[
            k8s.client.V1VolumeMount(name="code-volume", mount_path="/app")
        ],
        # Enforce a strict security context.
        security_context=k8s.client.V1SecurityContext(
            run_as_non_root=True,
            run_as_user=1001,
            allow_privilege_escalation=False,
            read_only_root_filesystem=True,
            capabilities=k8s.client.V1Capabilities(drop=["ALL"]),
        ),
        # Set resource limits to prevent abuse.
        resources=k8s.client.V1ResourceRequirements(
            requests={"cpu": self.cpu_requested, "memory": self.mem_requested},
            limits={"cpu": self.cpu_limit, "memory": self.mem_limit},
        ),
    )

    # Use tolerations to request a gVisor node.
    pod_spec = k8s.client.V1PodSpec(
        restart_policy="Never",
        containers=[container],
        volumes=[
            k8s.client.V1Volume(
                name="code-volume",
                config_map=k8s.client.V1ConfigMapVolumeSource(
                    name=configmap_name
                ),
            )
        ],
        runtime_class_name="gvisor",  # Request the gVisor runtime.
        tolerations=[
            k8s.client.V1Toleration(
                key="sandbox.gke.io/runtime",
                operator="Equal",
                value="gvisor",
                effect="NoSchedule",
            )
        ],
    )

    job_spec = k8s.client.V1JobSpec(
        template=k8s.client.V1PodTemplateSpec(spec=pod_spec),
        backoff_limit=0,  # Do not retry the Job on failure.
        # Kubernetes TTL controller will handle Job/Pod cleanup.
        ttl_seconds_after_finished=600,  # Garbage collect after 10 minutes.
    )

    # Assemble and return the final Job object.
    annotations = {
        "adk.agent.google.com/invocation-id": invocation_context.invocation_id
    }
    return k8s.client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=k8s.client.V1ObjectMeta(
            name=job_name, annotations=annotations
        ),
        spec=job_spec,
    )

  def _watch_job_completion(self, job_name: str) -> CodeExecutionResult:
    """Uses the watch API to efficiently wait for job completion."""
    watch = Watch()
    try:
      for event in watch.stream(
          self._batch_v1.list_namespaced_job,
          namespace=self.namespace,
          field_selector=f"metadata.name={job_name}",
          timeout_seconds=self.timeout_seconds,
      ):
        job = event["object"]
        if job.status.succeeded:
          watch.stop()
          logger.info(f"Job '{job_name}' succeeded.")
          logs = self._get_pod_logs(job_name)
          return CodeExecutionResult(stdout=logs)
        if job.status.failed:
          watch.stop()
          logger.error(f"Job '{job_name}' failed.")
          logs = self._get_pod_logs(job_name)
          return CodeExecutionResult(stderr=f"Job failed. Logs:\n{logs}")

      # If the loop finishes without returning, the watch timed out.
      raise TimeoutError(
          f"Job '{job_name}' did not complete within {self.timeout_seconds}s."
      )
    finally:
      watch.stop()

  def _get_pod_logs(self, job_name: str) -> str:
    """Retrieves logs from the pod created by the specified job.

    Raises:
        RuntimeError: If the pod cannot be found or logs cannot be fetched.
    """
    try:
      pods = self._core_v1.list_namespaced_pod(
          namespace=self.namespace,
          label_selector=f"job-name={job_name}",
          limit=1,
      )
      if not pods.items:
        raise RuntimeError(
            f"Could not find Pod for Job '{job_name}' to retrieve logs."
        )

      pod_name = pods.items[0].metadata.name
      return self._core_v1.read_namespaced_pod_log(
          name=pod_name, namespace=self.namespace
      )
    except ApiException as e:
      raise RuntimeError(
          f"API error retrieving logs for job '{job_name}': {e.reason}"
      ) from e

  def _create_code_configmap(self, name: str, code: str) -> None:
    """Creates a ConfigMap to hold the Python code."""
    body = k8s.client.V1ConfigMap(
        metadata=k8s.client.V1ObjectMeta(name=name), data={"code.py": code}
    )
    self._core_v1.create_namespaced_config_map(
        namespace=self.namespace, body=body
    )

  def _add_owner_reference(
      self, owner_job: k8s.client.V1Job, configmap_name: str
  ) -> None:
    """Patches the ConfigMap to be owned by the Job for auto-cleanup."""
    owner_reference = k8s.client.V1OwnerReference(
        api_version=owner_job.api_version,
        kind=owner_job.kind,
        name=owner_job.metadata.name,
        uid=owner_job.metadata.uid,
        controller=True,
    )
    patch_body = {"metadata": {"ownerReferences": [owner_reference.to_dict()]}}

    try:
      self._core_v1.patch_namespaced_config_map(
          name=configmap_name,
          namespace=self.namespace,
          body=patch_body,
      )
      logger.info(
          f"Set Job '{owner_job.metadata.name}' as owner of ConfigMap"
          f" '{configmap_name}'."
      )
    except ApiException as e:
      logger.warning(
          f"Failed to set ownerReference on ConfigMap '{configmap_name}'. "
          f"Manual cleanup is required. Reason: {e.reason}"
      )
