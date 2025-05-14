from typing import Any, Callable, Optional

from pydantic import BaseModel

from google.adk.approval.approval_grant import ApprovalAction
from google.adk.tools import BaseTool


class ApprovalPolicy(BaseModel):
  policy_name: str | None = None
  actions: list[ApprovalAction]

  def get_resources(self, args: dict[str, Any]):
    raise NotImplementedError("get_resources not implemented")


TOOL_NAMESPACE = "tool"
AGENT_NAMESPACE = "agent"
AGENTS_NAMESPACE = f"{AGENT_NAMESPACE}:agents"


class FunctionToolPolicy(ApprovalPolicy):
  """A tool policy"""

  resource_mappers: Callable[[dict[str, Any]], list[str]]

  @property
  def tool_name(self) -> Optional[str]:
    if self.policy_name.startswith(f"{TOOL_NAMESPACE}:"):
      return self.policy_name[len(f"{TOOL_NAMESPACE}:") :]
    else:
      return None

  @staticmethod
  def format_name(tool_name) -> str:
    return f"{TOOL_NAMESPACE}:{tool_name}"

  def get_resources(self, args: dict[str, Any]) -> list[str]:
    return self.resource_mappers(args)

  def get_action_resources(
      self, args: dict[str, Any]
  ) -> list[tuple[ApprovalAction, str]]:
    return [
        (action, resource)
        for action in self.actions
        for resource in self.get_resources(args)
    ]


class ApprovalPolicyRegistry(object):
  """Singleton Policy Registry"""

  TOOL_POLICIES: list[FunctionToolPolicy] = []

  @classmethod
  def register_tool_policy(cls, policy):
    if policy.policy_name is None:
      raise ValueError("Policy name cannot be None")
    cls.TOOL_POLICIES.append(policy)

  @classmethod
  def get_tool_policies(cls, tool_name):
    return [
        policy for policy in cls.TOOL_POLICIES if policy.tool_name == tool_name
    ]


register_tool_policy = ApprovalPolicyRegistry.register_tool_policy


def register_policy_for_tool(
    tool: BaseTool | Callable,
    policy: FunctionToolPolicy,
):
  if isinstance(tool, BaseTool):
    tool_name = tool.name
  else:
    tool_name = tool.__name__
  policy = FunctionToolPolicy(
      policy_name=FunctionToolPolicy.format_name(tool_name=tool_name),
      actions=policy.actions,
      resource_mappers=policy.resource_mappers,
  )
  register_tool_policy(policy)


def tool_policy(
    actions: list[ApprovalAction],
    resources: Callable[[dict[str, Any]], list[str]],
) -> Callable[[BaseTool | Callable], BaseTool | Callable]:
  def register(tool):
    if isinstance(tool, BaseTool):
      tool_name = tool.name
    else:
      tool_name = tool.__name__
    policy = FunctionToolPolicy(
        policy_name=FunctionToolPolicy.format_name(tool_name=tool_name),
        actions=actions,
        resource_mappers=resources,
    )
    register_tool_policy(policy)
    return tool

  return register


def resource_parameters(namespace: str, parameters: list[str]):
  mapping = {parameter: (namespace + ":{}").format for parameter in parameters}
  return resource_parameter_map(**mapping)


def resource_parameter_map(**mapping: Callable[[Any], str]):
  def resources_map(args):
    return [v(args[k]) for k, v in mapping.items()]

  return resources_map
