# Copyright 2026 Google LLC
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

"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

Database Schema Version: [FILL IN, e.g. v2]
Compatible ADK Versions: [FILL IN, e.g. >=1.25.0]

Rollback Notes:
  [FILL IN: Any special considerations for downgrade, or "Standard rollback"]
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
${imports if imports else ""}

revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
  ${upgrades if upgrades else "pass"}


def downgrade() -> None:
  ${downgrades if downgrades else "pass"}