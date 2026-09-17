# GKE Agent Sandbox RBAC

## Introduction

This directory is not a runnable agent. It holds the Kubernetes manifest that
`GkeCodeExecutor` needs in order to run generated code as Jobs on a GKE
cluster. The companion agent is
[`code_execution/gke_sandbox_agent.py`](../../code_execution/code_execution/gke_sandbox_agent.py).

`deployment_rbac.yaml` creates four objects in one namespace:

1. Namespace `agent-sandbox`
1. ServiceAccount `adk-agent-sa`
1. Role `adk-agent-role`, granting create/get/watch/list/delete on `jobs`,
   create/get/list/patch on `configmaps` (`patch` sets the ownerReference that
   lets each code ConfigMap be garbage collected with its Job), get/list/delete
   on `pods`, and get/list on `pods/log`
1. RoleBinding `adk-agent-binding`, binding the Role to the ServiceAccount

## How to Use

1. Apply the manifest to your cluster:

   ```bash
   kubectl apply -f contributing/samples/integrations/gke_agent_sandbox/deployment_rbac.yaml
   ```

1. Run the agent workload as `adk-agent-sa` in the `agent-sandbox` namespace,
   for example by setting `serviceAccountName: adk-agent-sa` on its Pod spec.

1. Pass the matching namespace when constructing the executor.
   `GkeCodeExecutor.namespace` defaults to `default`, so it must be set
   explicitly:

   ```python
   gke_executor = GkeCodeExecutor(namespace="agent-sandbox")
   ```

If you change the namespace, change it in both places — the manifest and the
executor — or the executor's API calls will be denied.
