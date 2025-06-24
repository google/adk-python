import logging
import uuid
from typing import Any

from google.adk.agents.invocation_context import InvocationContext
from google.adk.code_executors.base_code_executor import BaseCodeExecutor
from google.adk.code_executors.code_execution_utils import CodeExecutionInput, CodeExecutionResult

from kubernetes import client, config
from kubernetes.client.rest import ApiException
from kubernetes.watch import Watch

logger = logging.getLogger(__name__)

class GkeCodeExecutor(BaseCodeExecutor):
    """
    A secure, robust, and efficient code executor that runs Python code in a
    sandboxed gVisor Pod on GKE.

    Features includes:
    - Secure code execution via ConfigMaps and a strict security context.
    - Kubernetes-native job and pod garbage collection via TTL.
    - Efficient, event-driven waiting using the Kubernetes watch API.
    - Explicit resource limits to prevent abuse.
    """
    namespace: str = "default"
    image: str = "python:3.11-slim"
    timeout_seconds: int = 3000
    cpu_limit: str = "500m"
    mem_limit: str = "512Mi"
    use_gvisor_sandbox: bool = True

    _batch_v1: Any = None
    _core_v1: Any = None

    def __init__(self, **data):
        """
        Initializes the Pydantic model and the Kubernetes clients.
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
        """
        Orchestrates the secure execution of a code snippet on GKE.
        """
        job_name = f"adk-exec-{uuid.uuid4().hex[:10]}"
        configmap_name = f"code-src-{job_name}"

        try:
            # 1. Create a ConfigMap to hold the code securely.
            self._create_code_configmap(configmap_name, code_execution_input.code)
            # 2. Create the Job manifest with all security features.
            job_manifest = self._create_job_manifest(job_name, configmap_name)
            # 3. Create and run the Job on the cluster.
            self._batch_v1.create_namespaced_job(
                body=job_manifest, namespace=self.namespace
            )
            logger.info(f"Submitted Job '{job_name}' to namespace '{self.namespace}'.")
            # 4. Efficiently watch for the Job's completion.
            return self._watch_job_completion(job_name)

        except Exception as e:
            logger.error(
                f"An unexpected error occurred during execution of job '{job_name}': {e}",
                exc_info=True,
            )
            return CodeExecutionResult(stderr=f"Executor failed: {e}")
        finally:
            # 5. Always clean up the ConfigMap. The Job is cleaned up by Kubernetes.
            self._cleanup_configmap(configmap_name)

    def _create_job_manifest(self, job_name: str, configmap_name: str) -> client.V1Job:
        """Creates the complete V1Job object with security best practices."""
        # Define the container that will run the code.
        container = client.V1Container(
            name="code-runner",
            image=self.image,
            command=["python3", "/app/code.py"],
            volume_mounts=[
                client.V1VolumeMount(name="code-volume", mount_path="/app")
            ],
            # BEST PRACTICE: Enforce a strict security context.
            security_context=client.V1SecurityContext(
                run_as_non_root=True,
                run_as_user=1001,
                allow_privilege_escalation=False,
                read_only_root_filesystem=True,
                capabilities=client.V1Capabilities(drop=["ALL"]),
            ),
            # BEST PRACTICE: Set resource limits to prevent abuse.
            resources=client.V1ResourceRequirements(
                requests={"cpu": "100m", "memory": "128Mi"},
                limits={"cpu": self.cpu_limit, "memory": self.mem_limit},
            ),
        )
        
        # Pod Spec Customization for A/B Testing
        pod_spec_args = {
            "restart_policy": "Never",
            "containers": [container],
            "volumes": [
                client.V1Volume(
                    name="code-volume",
                    config_map=client.V1ConfigMapVolumeSource(name=configmap_name),
                )
            ],
        }
        
        if self.use_gvisor_sandbox:
            pod_spec_args["runtime_class_name"] = "gvisor"
            pod_spec_args["node_selector"] = {
                "cloud.google.com/gke-nodepool": "gvisor-nodepool"
            }
            pod_spec_args["tolerations"] = [
                client.V1Toleration(
                    key="sandbox.gke.io/runtime",
                    operator="Equal",
                    value="gvisor",
                    effect="NoSchedule",
                )
            ]
        else:
            pod_spec_args["node_selector"] = {
                "cloud.google.com/gke-nodepool": "standard-nodepool"
            }

        # Define the pod spec, mounting the code and targeting gVisor.
        pod_spec = client.V1PodSpec(**pod_spec_args)

        # Define the Job specification.
        job_spec = client.V1JobSpec(
            template=client.V1PodTemplateSpec(spec=pod_spec),
            backoff_limit=0,  # Do not retry the Job on failure.
            # BEST PRACTICE: Let the Kubernetes TTL controller handle cleanup.
            # This is more robust than client-side cleanup.
            ttl_seconds_after_finished=600,  # Garbage collect after 10 minutes.
        )
        
        # Assemble and return the final Job object.
        return client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(name=job_name),
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
        """Retrieves logs from the pod created by the specified job."""
        try:
            pods = self._core_v1.list_namespaced_pod(
                namespace=self.namespace, label_selector=f"job-name={job_name}", limit=1
            )
            if not pods.items:
                return "Error: Could not find pod for job."
            pod_name = pods.items[0].metadata.name
            
            return self._core_v1.read_namespaced_pod_log(
                name=pod_name, namespace=self.namespace
            )
        except ApiException as e:
            logger.error(f"Could not retrieve logs for job '{job_name}': {e}")
            return f"Error retrieving logs: {e.reason}"

    def _create_code_configmap(self, name: str, code: str) -> None:
        """Creates a ConfigMap to hold the Python code."""
        body = client.V1ConfigMap(
            metadata=client.V1ObjectMeta(name=name), data={"code.py": code}
        )
        self._core_v1.create_namespaced_config_map(
            namespace=self.namespace, body=body
        )

    def _cleanup_configmap(self, name: str) -> None:
        """Deletes a ConfigMap."""
        try:
            self._core_v1.delete_namespaced_config_map(name=name, namespace=self.namespace)
            logger.info(f"Cleaned up ConfigMap '{name}'.")
        except ApiException as e:
            if e.status != 404:
                logger.warning(f"Could not delete ConfigMap '{name}': {e.reason}")
