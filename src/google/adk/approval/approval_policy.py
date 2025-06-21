
"""Defines the structure for approval policies and a registry to manage them.

This module provides:
- `ApprovalPolicy`: An abstract base class for defining policies that require approval.
- `FunctionToolPolicy`: A concrete policy implementation specifically for tools (functions)
  that defines actions and how to map tool arguments to resource strings.
- `ApprovalPolicyRegistry`: A singleton registry to store and retrieve tool policies.
- Decorators and helper functions (`@tool_policy`, `register_policy_for_tool`,
  `resource_parameters`, `resource_parameter_map`) for easily defining and registering policies.

Policies determine what actions on what resources require approval before a tool can be executed.
"""
from __future__ import annotations
from typing import Any, Callable, Optional

from pydantic import BaseModel

from google.adk.approval.approval_grant import ApprovalAction
from google.adk.tools import BaseTool


class ApprovalPolicy(BaseModel):
  """Abstract base model for an approval policy.

  An approval policy defines a set of actions that, when applied to resources derived
  from tool arguments, require an explicit approval grant.
  """
  policy_name: str | None = None
  """An optional unique name for the policy, often linking it to a specific tool or agent feature."""
  actions: list[ApprovalAction]
  """The list of actions (e.g., 'tool:files:read', 'agent:delegate') that this policy governs."""

  def get_resources(self, args: dict[str, Any]) -> list[str]:
    """Abstract method to derive resource strings from tool arguments.

    Subclasses must implement this to define how tool invocation arguments
    map to specific resource identifiers that the policy's actions apply to.

    Args:
        args: A dictionary of arguments passed to a tool call.

    Returns:
        A list of resource strings.

    Raises:
        NotImplementedError: If the subclass does not override this method.
    """
    raise NotImplementedError("get_resources not implemented")


TOOL_NAMESPACE = "tool"
AGENT_NAMESPACE = "agent"
AGENTS_NAMESPACE = f"{AGENT_NAMESPACE}:agents"


class FunctionToolPolicy(ApprovalPolicy):
  """A policy specifically designed for function tools.

  It links tool names to actions and provides a mechanism (`resource_mappers`)
  to extract resource strings from the arguments passed to the tool.

  Attributes:
      resource_mappers: A callable that takes tool arguments (dict) and returns a list of resource strings.
  """

  resource_mappers: Callable[[dict[str, Any]], list[str]]
  """A function that maps tool arguments to a list of resource strings.
  For example, for a file reading tool, this might extract the file path.
  """

  @property
  def tool_name(self) -> Optional[str]:
    """Extracts the tool name if the policy_name follows the 'tool:<name>' format."""
    if self.policy_name.startswith(f"{TOOL_NAMESPACE}:"):
      return self.policy_name[len(f"{TOOL_NAMESPACE}:") :]
    else:
      return None

  @staticmethod
  def format_name(tool_name: str) -> str:
    """Formats a policy name for a tool, ensuring it's prefixed with the tool namespace."""
    return f"{TOOL_NAMESPACE}:{tool_name}"

  def get_resources(self, args: dict[str, Any]) -> list[str]:
    """Derives resource strings from tool arguments using the `resource_mappers` function."""
    return self.resource_mappers(args)

  def get_action_resources(
      self, args: dict[str, Any]
  ) -> list[tuple[ApprovalAction, str]]:
    """Generates all action-resource pairs covered by this policy for given tool arguments.

    Args:
        args: The arguments passed to the tool call.

    Returns:
        A list of tuples, where each tuple is (action_string, resource_string).
    """
    return [
        (action, resource)
        for action in self.actions
        for resource in self.get_resources(args)
    ]


class ApprovalPolicyRegistry(object):
  """A singleton registry for managing `FunctionToolPolicy` instances.

  This registry stores policies associated with tool names, allowing the system
  to look up relevant policies when a tool is about to be executed.
  """

  tool_policies: list[FunctionToolPolicy] = []
  """Static list holding all registered `FunctionToolPolicy` instances."""

  @classmethod
  def register_tool_policy(cls, policy: FunctionToolPolicy):
    """Registers a `FunctionToolPolicy`.

    Ensures that the policy has a name and adds it to the global list if not already present.

    Args:
        policy: The `FunctionToolPolicy` instance to register.

    Raises:
        ValueError: If `policy.policy_name` is None.
    """
    if policy.policy_name is None:
      raise ValueError("Policy name cannot be None")
    for existing_policy in cls.tool_policies:
      if existing_policy.policy_name != policy.policy_name:
        continue
      if existing_policy.actions != policy.actions:
        continue
      return
    cls.tool_policies.append(policy)

  @classmethod
  def get_tool_policies(cls, tool_name: str) -> list[FunctionToolPolicy]:
    """Retrieves all registered policies associated with a given tool name.

    Args:
        tool_name: The name of the tool.

    Returns:
        A list of `FunctionToolPolicy` instances whose `tool_name` matches.
    """
    return [
        policy for policy in cls.tool_policies if policy.tool_name == tool_name
    ]


register_tool_policy = ApprovalPolicyRegistry.register_tool_policy
"""Alias for `ApprovalPolicyRegistry.register_tool_policy` for convenient access."""


def register_policy_for_tool(
    tool: BaseTool | Callable,
    policy: FunctionToolPolicy,
):
  """Registers a given policy for a specific tool.

  This helper function constructs the correct policy name based on the tool's name
  (whether it's a `BaseTool` instance or a callable) and then registers the policy.

  Args:
      tool: The tool (either a `BaseTool` subclass or a callable) to associate the policy with.
      policy: The `FunctionToolPolicy` to register. The `policy_name` attribute of this
              object will be overridden.
  """
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
  """Decorator to associate an approval policy with a tool function or `BaseTool` class.

  This decorator simplifies the creation and registration of `FunctionToolPolicy` objects.

  Args:
      actions: A list of `ApprovalAction` strings that the policy governs.
      resources: A callable that takes the tool's arguments (a dict) and returns a list
                 of resource strings to which the actions apply.

  Returns:
      A decorator function that, when applied to a tool, registers a policy for it.
  """
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


def resource_parameters(namespace: str, parameters: list[str]) -> Callable[[dict[str, Any]], list[str]]:
  """Creates a resource mapper function that extracts resource strings from specified tool arguments.

  This is a convenience function for a common pattern where resource identifiers are
  directly derived from the values of certain tool parameters, prefixed with a namespace.

  Example:
      `resource_parameters("tool:files", ["path"])` would create a mapper that, for a tool call
      `my_tool(path="/data/f.txt")`, returns `["tool:files:/data/f.txt"]`.

  Args:
      namespace: The namespace string to prefix to each parameter value (e.g., "tool:my_tool_namespace").
      parameters: A list of parameter names from the tool's arguments whose values will be used
                  to construct resource strings.

  Returns:
      A callable suitable for use as the `resources` argument in `@tool_policy` or
      the `resource_mappers` attribute of `FunctionToolPolicy`.
  """
  mapping = {parameter: (namespace + ":{}").format for parameter in parameters}
  return resource_parameter_map(**mapping)


def resource_parameter_map(**mapping: Callable[[Any], str]) -> Callable[[dict[str, Any]], list[str]]:
  """Creates a resource mapper function based on a direct mapping of argument names to formatters.

  This provides a flexible way to construct resource strings by applying specific formatting
  functions to the values of named tool arguments.

  Example:
      `resource_parameter_map(filePath=lambda x: f"files:{x}", dirPath=lambda x: f"dirs:{x}")`
      If a tool is called with `my_tool(filePath="/a.txt", otherArg=1)`, this mapper would return
      `["files:/a.txt"]` (if `dirPath` was not provided).

  Args:
      **mapping: Keyword arguments where each key is a tool argument name, and the value is a
                 callable that takes the argument's value and returns a formatted resource string part.

  Returns:
      A callable suitable for use as the `resources` argument in `@tool_policy` or
      the `resource_mappers` attribute of `FunctionToolPolicy`.
  """
  def resources_map(args):
    return [v(args[k]) for k, v in mapping.items()]

  return resources_map
