# Copyright 2025 Google LLC
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

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import field_validator

from ...features import experimental
from ...features import FeatureName


class WriteMode(Enum):
  """Write mode indicating what levels of write operations are allowed in BigQuery."""

  BLOCKED = 'blocked'
  """No write operations are allowed.

  This mode implies that only read (i.e. SELECT query) operations are allowed.
  """

  PROTECTED = 'protected'
  """Only protected write operations are allowed in a BigQuery session.

  In this mode write operations in the anonymous dataset of a BigQuery session
  are allowed. For example, a temporary table can be created, manipulated and
  deleted in the anonymous dataset during Agent interaction, while protecting
  permanent tables from being modified or deleted. To learn more about BigQuery
  sessions, see https://cloud.google.com/bigquery/docs/sessions-intro.
  """

  ALLOWED = 'allowed'
  """All write operations are allowed."""


@experimental(FeatureName.BIG_QUERY_TOOL_CONFIG)
class BigQueryToolConfig(BaseModel):
  """Configuration for BigQuery tools."""

  # Forbid any fields not defined in the model
  model_config = ConfigDict(extra='forbid')

  write_mode: WriteMode = WriteMode.BLOCKED
  """Write mode for BigQuery tools.

  By default, the tool will allow only read operations. This behaviour may
  change in future versions.
  """

  maximum_bytes_billed: Optional[int] = None
  """Maximum number of bytes to bill for a query.

  In BigQuery on-demand pricing, charges are rounded up to the nearest MB, with
  a minimum 10 MB data processed per table referenced by the query, and with a
  minimum 10 MB data processed per query. So this value must be set >=10485760.
  """

  max_query_result_rows: int = 50
  """Maximum number of rows to return from a query.

  By default, the query result will be limited to 50 rows.
  """

  application_name: Optional[str] = None
  """Name of the application using the BigQuery tools.

  By default, no particular application name will be set in the BigQuery
  interaction. But if the tool user (agent builder) wants to differentiate
  their application/agent for tracking or support purpose, they can set this
  field. If set, this value will be added to the user_agent in BigQuery API calls, and also to the BigQuery job labels with the key
  "adk-bigquery-application-name".
  """

  compute_project_id: Optional[str] = None
  """GCP project ID to use for the BigQuery compute operations.

  This can be set as a guardrail to ensure that the tools perform the compute
  operations (such as query execution) in a specific project.
  """

  location: Optional[str] = None
  """BigQuery location to use for the data and compute.

  This can be set if the BigQuery tools are expected to process data in a
  particular BigQuery location. If not set, then location would be automatically
  determined based on the data location in the query. For all supported
  locations, see https://cloud.google.com/bigquery/docs/locations.
  """

  job_labels: Optional[dict[str, str]] = None
  """Labels to apply to BigQuery jobs for tracking and monitoring.

  These labels will be added to all BigQuery jobs executed by the tools.
  Labels must be key-value pairs where both keys and values are strings.
  Labels can be used for billing, monitoring, and resource organization.
  For more information about labels, see 
  https://cloud.google.com/bigquery/docs/labels-intro.
  """

  @field_validator('maximum_bytes_billed')
  @classmethod
  def validate_maximum_bytes_billed(cls, v):
    """Validate the maximum bytes billed."""
    if v and v < 10_485_760:
      raise ValueError(
          'In BigQuery on-demand pricing, charges are rounded up to the nearest'
          ' MB, with a minimum 10 MB data processed per table referenced by the'
          ' query, and with a minimum 10 MB data processed per query. So'
          ' max_bytes_billed must be set >=10485760.'
      )
    return v

  @field_validator('application_name')
  @classmethod
  def validate_application_name(cls, v):
    """Validate the application name."""
    if v and ' ' in v:
      raise ValueError('Application name should not contain spaces.')
    return v

  @field_validator('job_labels')
  @classmethod
  def validate_job_labels(cls, v):
    """Validate that job_labels keys are not empty."""
    if v is not None:
      for key in v.keys():
        if not key:
          raise ValueError('Label keys cannot be empty.')
    return v
