import logging
import uuid

from google.adk.agents.invocation_context import InvocationContext
from google.adk.code_executors.base_code_executor import BaseCodeExecutor
from google.adk.code_executors.code_execution_utils import CodeExecutionInput, CodeExecutionResult

from kubernetes import client, config
from kubernetes.client.rest import ApiException
from kubernetes.watch import Watch

logger = logging.getLogger(__name__)

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
    This executor interacts with the Kubernetes API and requires a ServiceAccount 
    with specific RBAC permissions to function. The agent's pod needs permissions
    to create/watch Jobs, create/delete ConfigMaps, and list Pods to read logs.
    For a complete, working example of the required Role and RoleBinding, see the
    file at: contributing/samples/gke_agent_sandbox/deployment_rbac.yaml
    """
    namespace: str = "default"
    image: str = "python:3.11-slim"
    timeout_seconds: int = 300
    cpu_request: str = "200m"
    mem_request: str = "256Mi"
    cpu_limit: str = "500m"
    mem_limit: str = "512Mi"

    _batch_v1: client.BatchV1Api
    _core_v1: client.CoreV1Api

    def __init__(self, **data):
        """Initializes the executor and the Kubernetes API clients.

        This constructor supports overriding default class attributes (like
        'namespace', 'image', etc.) by passing them as keyword arguments. It
        automatically configures the Kubernetes client to work either within a
        cluster (in-cluster config) or locally using a kubeconfig file.
        """
        super().__init__(**data)
        try:
            config.load_incluster_config()
            logger.info("Using in-cluster Kubernetes configuration.")
        except config.ConfigException:
            logger.info("In-cluster config not found. Falling back to local kubeconfig.")
            config.load_kube_config()

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
            self._create_code_configmap(configmap_name, code_execution_input.code)
            job_manifest = self._create_job_manifest(job_name, configmap_name, invocation_context)

            self._batch_v1.create_namespaced_job(
                body=job_manifest, namespace=self.namespace
            )
            logger.info(f"Submitted Job '{job_name}' to namespace '{self.namespace}'.")
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
            return CodeExecutionResult(stderr=f"An unexpected executor error occurred: {e}")
        finally:
            # The Job is cleaned up by the TTL controller, and we ensure the ConfigMap is always deleted.
            self._cleanup_configmap(configmap_name)

    def _create_job_manifest(self, job_name: str, configmap_name: str, invocation_context: InvocationContext) -> client.V1Job:
        """Creates the complete V1Job object with security best practices."""
        # Define the container that will run the code.
        container = client.V1Container(
            name="code-runner",
            image=self.image,
            command=["python3", "/app/code.py"],
            volume_mounts=[
                client.V1VolumeMount(name="code-volume", mount_path="/app")
            ],
            # Enforce a strict security context.
            security_context=client.V1SecurityContext(
                run_as_non_root=True,
                run_as_user=1001,
                allow_privilege_escalation=False,
                read_only_root_filesystem=True,
                capabilities=client.V1Capabilities(drop=["ALL"]),
            ),
            # Set resource limits to prevent abuse.
            resources=client.V1ResourceRequirements(
                requests={"cpu": self.cpu_request, "memory": self.mem_request},
                limits={"cpu": self.cpu_limit, "memory": self.mem_limit},
            ),
        )
        
        # Use tolerations to request a gVisor node.
        pod_spec = client.V1PodSpec(
            restart_policy="Never",
            containers=[container],
            volumes=[
                client.V1Volume(
                    name="code-volume",
                    config_map=client.V1ConfigMapVolumeSource(name=configmap_name),
                )
            ],
            runtime_class_name="gvisor",  # Request the gVisor runtime.
            tolerations=[
                client.V1Toleration(
                    key="sandbox.gke.io/runtime",
                    operator="Equal",
                    value="gvisor",
                    effect="NoSchedule",
                )
            ],
        )

        job_spec = client.V1JobSpec(
            template=client.V1PodTemplateSpec(spec=pod_spec),
            backoff_limit=0,  # Do not retry the Job on failure.
            # Kubernetes TTL controller will handle Job/Pod cleanup.
            ttl_seconds_after_finished=600,  # Garbage collect after 10 minutes.
        )
        
        # Assemble and return the final Job object.
        annotations = {
            "adk.agent.google.com/invocation-id": invocation_context.invocation_id
        }
        return client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(name=job_name, annotations=annotations),
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
                namespace=self.namespace, label_selector=f"job-name={job_name}", limit=1
            )
            if not pods.items:
                raise RuntimeError(f"Could not find Pod for Job '{job_name}' to retrieve logs.")

            pod_name = pods.items[0].metadata.name
            return self._core_v1.read_namespaced_pod_log(
                name=pod_name, namespace=self.namespace
            )
        except ApiException as e:
            raise RuntimeError(f"API error retrieving logs for job '{job_name}': {e.reason}") from e

    def _create_code_configmap(self, name: str, code: str) -> None:
        """Creates a ConfigMap to hold the Python code."""
        body = client.V1ConfigMap(
            metadata=client.V1ObjectMeta(name=name), data={"code.py": code}
        )
        self._core_v1.create_namespaced_config_map(namespace=self.namespace, body=body)

    def _cleanup_configmap(self, name: str) -> None:
        """Deletes a ConfigMap."""
        try:
            self._core_v1.delete_namespaced_config_map(name=name, namespace=self.namespace)
            logger.info(f"Cleaned up ConfigMap '{name}'.")
        except ApiException as e:
            if e.status != 404:
                logger.warning(f"Could not delete ConfigMap '{name}': {e.reason}")
