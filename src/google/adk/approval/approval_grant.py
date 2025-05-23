
"""Defines the core data structures for representing approval grants.

This module includes classes for:
- `ApprovalActor`: Represents an entity (user, agent, tool) involved in an approval.
- `ApprovalEffect`: Enumerates the possible outcomes of an approval (allow, deny, challenge).
- `ApprovalGrant`: Encapsulates the details of a permission grant, including the
  effect, actions, resources, grantee, grantor, and optional expiration.
"""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel


class ApprovalActor(BaseModel):
  id: str
  """A unique identifier for the actor (e.g., user ID, agent session ID, tool call ID)."""
  type: str = Literal["user", "agent", "tool"]
  """The type of the actor."""
  on_behalf_of: ApprovalActor | None = None
  """The actor on whose behalf this actor is operating, if any (e.g., an agent acting on behalf of a user)."""


class ApprovalEffect(str, Enum):
  allow = "allow"
  """Indicates that the requested action is permitted."""
  deny = "deny"
  """Indicates that the requested action is explicitly forbidden."""
  challenge = "challenge"
  """Indicates that further information or confirmation is required before allowing or denying."""


ApprovalAction = str
"""Type alias for an action string (e.g., 'tool:read_file', 'agent:use')."""
ApprovalResource = str
"""Type alias for a resource string (e.g., 'tool:files:/path/to/file', 'agent:agent_name')."""


class ApprovalGrant(BaseModel):
  """Effect the actions on the resources to the grantee by the grantor until the expiration."""

  effect: Literal[ApprovalEffect.allow, ApprovalEffect.deny]
  """The effect of this grant, either allowing or denying the specified actions on the resources."""
  actions: list[ApprovalAction]
  """A list of actions (e.g., 'tool:read_file') that this grant permits or denies."""
  resources: list[ApprovalResource]
  """A list of resources (e.g., 'tool:files:/path/to/data.txt') to which this grant applies."""
  grantee: ApprovalActor
  """The actor (user, agent, or tool) to whom the permissions are granted."""
  grantor: ApprovalActor
  """The actor who authorized this grant (e.g., an end-user or a delegating agent)."""
  expiration_time: Optional[datetime] = None
  """The optional time after which this grant is no longer valid. If None, the grant does not expire."""

  comment: Optional[str] = None
  """An optional comment from the grantor, often used to explain the reason for a denial or to provide context for an approval."""
