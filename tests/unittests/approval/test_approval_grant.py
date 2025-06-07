import uuid

import pytest

from google.adk.approval.approval_grant import ApprovalAction, ApprovalActor, ApprovalGrant
from google.adk.approval.approval_policy import ApprovalPolicyRegistry, resource_parameters, tool_policy
from google.adk.tools import FunctionTool, ToolContext


@pytest.fixture()
def read_file_tool():
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

  return read_file


def test_approval_grant__once(read_file_tool, tmp_path):
  tool = FunctionTool(read_file_tool)

  user_id = str(uuid.uuid4())
  session_id = str(uuid.uuid4())
  function_call_id = str(uuid.uuid4())
  path = str(tmp_path / "file-to-read.txt")
  args = dict(path=path)

  policies = ApprovalPolicyRegistry.get_tool_policies(tool.name)
  assert len(policies) == 1

  policy = policies[0]
  grant = ApprovalGrant(
      effect="allow",
      actions=policy.actions,
      resources=policy.get_resources(args),
      grantee=ApprovalActor(
          id=f"{tool.name}:{function_call_id}",
          type="tool",
          on_behalf_of=ApprovalActor(
              id=session_id,
              type="agent",
              on_behalf_of=ApprovalActor(
                  id=user_id,
                  type="user",
              ),
          ),
      ),
      grantor=ApprovalActor(
          id=user_id,
          type="user",
      ),
  )

  assert grant
