from crewai.tools import tool as crew_ai_tool
from langchain_core.tools import tool as lg_tool

from google.adk import Agent
from google.adk.approval.approval_grant import ApprovalAction
from google.adk.approval.approval_policy import ApprovalPolicyRegistry, FunctionToolPolicy, resource_parameters, tool_policy
from google.adk.tools import FunctionTool
from google.adk.tools.agent_tool import AgentTool, ApprovedAgentTool
from google.adk.tools.crewai_tool import CrewaiTool
from google.adk.tools.langchain_tool import LangchainTool
from tests.unittests import utils
from tests.unittests.testing_utils import MockModel


def test_tool_policy__simple_policy(tmp_path):
  LOCAL_FILE_NAMESPACE = "tool:local_file"
  LOCAL_FILE_READ_PERMISSION = ApprovalAction(f"{LOCAL_FILE_NAMESPACE}:read")

  @tool_policy(
      actions=[LOCAL_FILE_READ_PERMISSION],
      resources=resource_parameters(LOCAL_FILE_NAMESPACE, ["path"]),
  )
  def read_file(*, path: str):
    try:
      with open(path, "r") as f:
        return f.read()
    except Exception as e:
      return str(e)

  function_tool = FunctionTool(read_file)
  assert function_tool.name == "read_file"
  policies = ApprovalPolicyRegistry.get_tool_policies(function_tool.name)
  assert len(policies) == 1
  policy: FunctionToolPolicy = policies[0]
  assert policy.tool_name == "read_file"

  assert len(policy.actions) == 1
  action: ApprovalAction = policy.actions[0]
  assert action == "tool:local_file:read"

  path = str(tmp_path / "file-to-read.txt")
  args = dict(path=path)
  assert policy.get_resources(args) == [f"tool:local_file:{path}"]


def test_tool_policy__policy_with_two_actions(tmp_path):
  LOCAL_FILE_NAMESPACE = "tool:local_file"
  LOCAL_FILE_READ_ACTION = ApprovalAction(f"{LOCAL_FILE_NAMESPACE}:read")
  LOCAL_FILE_WRITE_ACTION = ApprovalAction(f"{LOCAL_FILE_NAMESPACE}:write")

  @tool_policy(
      actions=[LOCAL_FILE_READ_ACTION, LOCAL_FILE_WRITE_ACTION],
      resources=resource_parameters(LOCAL_FILE_NAMESPACE, ["path"]),
  )
  def str_replace(path: str, old_str: str, new_str: str):
    try:
      with open(path, "r") as f:
        content = f.read()
      new_content = content.replace(old_str, new_str)
      if new_content != content:
        with open(path, "w") as f:
          return f.write(new_content)
      else:
        return "Error: could not find string to replace"
    except Exception as e:
      return str(e)

  function_tool = FunctionTool(str_replace)
  assert function_tool.name == "str_replace"
  policies = ApprovalPolicyRegistry.get_tool_policies(function_tool.name)
  assert len(policies) == 1
  policy = policies[0]
  assert policy.tool_name == "str_replace"

  assert len(policy.actions) == 2
  read_action, write_action = policy.actions
  assert read_action == "tool:local_file:read"
  assert write_action == "tool:local_file:write"

  path = str(tmp_path / "file-to-read.txt")
  args = dict(path=path, old_str="something", new_str="something else")
  assert policy.get_resources(args) == [f"tool:local_file:{path}"]


def test_tool_policy__tool_with_two_policies(tmp_path):
  GIT_NAMESPACE = "tool:git"
  GIT_READ = ApprovalAction(f"{GIT_NAMESPACE}:read")
  GITLAB_NAMESPACE = "tool:gitlab"
  GITLAB_READ = ApprovalAction(f"{GITLAB_NAMESPACE}:read")

  @tool_policy(
      actions=[GIT_READ],
      resources=lambda args: [
          f"{GIT_NAMESPACE}:{args['repo_path']}",
      ],
  )
  @tool_policy(
      actions=[GITLAB_READ],
      resources=lambda args: [
          f"{GITLAB_NAMESPACE}:*",
      ],
  )
  def gitlab_get_mr_for_repo(repo_path: str):
    ...

  function_tool = FunctionTool(gitlab_get_mr_for_repo)
  assert function_tool.name == "gitlab_get_mr_for_repo"
  policies = ApprovalPolicyRegistry.get_tool_policies(function_tool.name)
  assert len(policies) == 2
  gitlab_policy, git_policy = policies
  assert git_policy.tool_name == "gitlab_get_mr_for_repo"
  assert gitlab_policy.tool_name == "gitlab_get_mr_for_repo"

  assert len(git_policy.actions) == 1
  assert git_policy.actions[0] == "tool:git:read"
  assert gitlab_policy.actions[0] == "tool:gitlab:read"

  repo_path = str(tmp_path / "some-repo")
  args = dict(repo_path=repo_path)
  assert git_policy.get_resources(args) == [f"tool:git:{repo_path}"]
  assert gitlab_policy.get_resources(args) == [f"tool:gitlab:*"]


def test_tool_policy__agent_tool():
  agent = Agent(name="useful_agent", model=MockModel.create(responses=[]))
  agent_tool = ApprovedAgentTool(agent)
  assert agent_tool.name == "useful_agent"
  policies = ApprovalPolicyRegistry.get_tool_policies(agent_tool.name)
  assert len(policies) == 1
  policy = policies[0]
  assert policy.tool_name == "useful_agent"

  assert len(policy.actions) == 1
  assert policy.actions[0] == "tool:agent:use"
  assert policy.get_resources({}) == [f"tool:agents:useful_agent"]


def test_tool_policy__langchain_tool(tmp_path):
  GIT_NAMESPACE = "tool:git"
  GIT_READ = ApprovalAction(f"{GIT_NAMESPACE}:read")
  GITLAB_NAMESPACE = "tool:gitlab"
  GITLAB_READ = ApprovalAction(f"{GITLAB_NAMESPACE}:read")

  @lg_tool
  def gitlab_get_mr_for_repo(repo_path: str):
    """Tool to get gitlab Merge Request for a Repo"""
    ...

  langchain_tool = LangchainTool(
      gitlab_get_mr_for_repo,
      policies=[
          FunctionToolPolicy(
              actions=[GIT_READ],
              resource_mappers=lambda args: [
                  f"{GIT_NAMESPACE}:{args['repo_path']}",
              ],
          ),
          FunctionToolPolicy(
              actions=[GITLAB_READ],
              resource_mappers=lambda args: [
                  f"{GITLAB_NAMESPACE}:*",
              ],
          ),
      ],
  )
  assert langchain_tool.name == "gitlab_get_mr_for_repo"
  policies = ApprovalPolicyRegistry.get_tool_policies(langchain_tool.name)
  assert len(policies) == 2
  git_policy, gitlab_policy = policies
  assert git_policy.tool_name == "gitlab_get_mr_for_repo"
  assert gitlab_policy.tool_name == "gitlab_get_mr_for_repo"

  assert len(git_policy.actions) == 1
  assert git_policy.actions[0] == "tool:git:read"
  assert gitlab_policy.actions[0] == "tool:gitlab:read"

  repo_path = str(tmp_path / "some-repo")
  args = dict(repo_path=repo_path)
  assert git_policy.get_resources(args) == [f"tool:git:{repo_path}"]
  assert gitlab_policy.get_resources(args) == [f"tool:gitlab:*"]


def test_tool_policy__crewai_tool(tmp_path):
  GIT_NAMESPACE = "tool:git"
  GIT_READ = ApprovalAction(f"{GIT_NAMESPACE}:read")
  GITLAB_NAMESPACE = "tool:gitlab"
  GITLAB_READ = ApprovalAction(f"{GITLAB_NAMESPACE}:read")

  @crew_ai_tool
  def gitlab_get_mr_for_repo(repo_path: str):
    """Tool to get gitlab Merge Request for a Repo"""
    ...

  crewai_tool = CrewaiTool(
      gitlab_get_mr_for_repo,
      name="",
      description="",
      policies=[
          FunctionToolPolicy(
              actions=[GIT_READ],
              resource_mappers=lambda args: [
                  f"{GIT_NAMESPACE}:{args['repo_path']}",
              ],
          ),
          FunctionToolPolicy(
              actions=[GITLAB_READ],
              resource_mappers=lambda args: [
                  f"{GITLAB_NAMESPACE}:*",
              ],
          ),
      ],
  )
  assert crewai_tool.name == "gitlab_get_mr_for_repo"
  policies = ApprovalPolicyRegistry.get_tool_policies(crewai_tool.name)
  assert len(policies) == 2
  git_policy, gitlab_policy = policies
  assert git_policy.tool_name == "gitlab_get_mr_for_repo"
  assert gitlab_policy.tool_name == "gitlab_get_mr_for_repo"

  assert len(git_policy.actions) == 1
  assert git_policy.actions[0] == "tool:git:read"
  assert gitlab_policy.actions[0] == "tool:gitlab:read"

  repo_path = str(tmp_path / "some-repo")
  args = dict(repo_path=repo_path)
  assert git_policy.get_resources(args) == [f"tool:git:{repo_path}"]
  assert gitlab_policy.get_resources(args) == [f"tool:gitlab:*"]
