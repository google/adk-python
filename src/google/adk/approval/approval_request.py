from google.genai.types import FunctionCall
from pydantic import BaseModel

from google.adk.approval.approval_grant import ApprovalAction, ApprovalActor, ApprovalEffect, ApprovalGrant, ApprovalResource


class ApprovalChallenge(BaseModel):
  """Effect the actions on the resources to the grantee by the grantor until the expiration."""

  actions: list[ApprovalAction]
  """The actions to which the grant will effect."""
  resources: list[ApprovalResource]
  """The resources that this grant affects."""


class ApprovalStatus(BaseModel):
  effect: ApprovalEffect
  challenges: list[ApprovalChallenge]


class ApprovalDenied(ValueError):

  def __init__(self, denied_challenges: list[ApprovalChallenge]):
    super().__init__()
    self.denied_challenges = denied_challenges


class ApprovalRequest(BaseModel):
  function_call: FunctionCall
  challenges: list[ApprovalChallenge]
  grantee: ApprovalActor
  """Who the grant applies to."""


class ApprovalResponse(BaseModel):
  grants: list[ApprovalGrant]
