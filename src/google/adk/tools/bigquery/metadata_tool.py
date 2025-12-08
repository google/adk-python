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

from google.auth.credentials import Credentials
from google.cloud import bigquery

from . import client
from .config import BigQueryToolConfig


def list_dataset_ids(
    project_id: str, credentials: Credentials, settings: BigQueryToolConfig
) -> list[str]:
  """List BigQuery dataset ids in a Google Cloud project.

  Args:
      project_id (str): The Google Cloud project id.
      credentials (Credentials): The credentials to use for the request.

  Returns:
      list[str]: List of the BigQuery dataset ids present in the project.

  Examples:
      >>> list_dataset_ids("bigquery-public-data")
      ['america_health_rankings',
       'american_community_survey',
       'aml_ai_input_dataset',
       'austin_311',
       'austin_bikeshare',
       'austin_crime',
       'austin_incidents',
       'austin_waste',
       'baseball',
       'bbc_news']
  """
  try:
    bq_client = client.get_bigquery_client(
        project=project_id,
        credentials=credentials,
        location=settings.location,
        user_agent=[settings.application_name, "list_dataset_ids"],
    )

    datasets = []
    for dataset in bq_client.list_datasets(project_id):
      datasets.append(dataset.dataset_id)
    return datasets
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


def get_dataset_info(
    project_id: str,
    dataset_id: str,
    credentials: Credentials,
    settings: BigQueryToolConfig,
) -> dict:
  """Get metadata information about a BigQuery dataset.

  Args:
      project_id (str): The Google Cloud project id containing the dataset.
      dataset_id (str): The BigQuery dataset id.
      credentials (Credentials): The credentials to use for the request.

  Returns:
      dict: Dictionary representing the properties of the dataset.

  Examples:
      >>> get_dataset_info("bigquery-public-data", "cdc_places")
      {
        "kind": "bigquery#dataset",
        "etag": "fz9BaiXKgbGi53EpI2rJug==",
        "id": "bigquery-public-data:cdc_places",
        "selfLink": "https://content-bigquery.googleapis.com/bigquery/v2/projects/bigquery-public-data/datasets/cdc_places",
        "datasetReference": {
          "datasetId": "cdc_places",
          "projectId": "bigquery-public-data"
        },
        "description": "Local Data for Better Health, County Data",
        "access": [
          {
            "role": "WRITER",
            "specialGroup": "projectWriters"
          },
          {
            "role": "OWNER",
            "specialGroup": "projectOwners"
          },
          {
            "role": "OWNER",
            "userByEmail": "some-redacted-email@bigquery-public-data.iam.gserviceaccount.com"
          },
          {
            "role": "READER",
            "specialGroup": "projectReaders"
          }
        ],
        "creationTime": "1640891845643",
        "lastModifiedTime": "1640891845643",
        "location": "US",
        "type": "DEFAULT",
        "maxTimeTravelHours": "168"
      }
  """
  try:
    bq_client = client.get_bigquery_client(
        project=project_id,
        credentials=credentials,
        location=settings.location,
        user_agent=[settings.application_name, "get_dataset_info"],
    )
    dataset = bq_client.get_dataset(
        bigquery.DatasetReference(project_id, dataset_id)
    )
    return dataset.to_api_repr()
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


def list_table_ids(
    project_id: str,
    dataset_id: str,
    credentials: Credentials,
    settings: BigQueryToolConfig,
) -> list[str]:
  """List table ids in a BigQuery dataset.

  Args:
      project_id (str): The Google Cloud project id containing the dataset.
      dataset_id (str): The BigQuery dataset id.
      credentials (Credentials): The credentials to use for the request.

  Returns:
      list[str]: List of the tables ids present in the dataset.

  Examples:
      >>> list_table_ids("bigquery-public-data", "cdc_places")
      ['chronic_disease_indicators',
       'local_data_for_better_health_county_data']
  """
  try:
    bq_client = client.get_bigquery_client(
        project=project_id,
        credentials=credentials,
        location=settings.location,
        user_agent=[settings.application_name, "list_table_ids"],
    )

    tables = []
    for table in bq_client.list_tables(
        bigquery.DatasetReference(project_id, dataset_id)
    ):
      tables.append(table.table_id)
    return tables
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


def get_table_info(
    project_id: str,
    dataset_id: str,
    table_id: str,
    credentials: Credentials,
    settings: BigQueryToolConfig,
) -> dict:
  """Get metadata information about a BigQuery table.

  Args:
      project_id (str): The Google Cloud project id containing the dataset.
      dataset_id (str): The BigQuery dataset id containing the table.
      table_id (str): The BigQuery table id.
      credentials (Credentials): The credentials to use for the request.

  Returns:
      dict: Dictionary representing the properties of the table.

  Examples:
      >>> get_table_info("bigquery-public-data", "cdc_places", "local_data_for_better_health_county_data")
      {
        "kind": "bigquery#table",
        "etag": "wx23aDqmgc39oUSiNuYTAA==",
        "id": "bigquery-public-data:cdc_places.local_data_for_better_health_county_data",
        "selfLink": "https://content-bigquery.googleapis.com/bigquery/v2/projects/bigquery-public-data/datasets/cdc_places/tables/local_data_for_better_health_county_data",
        "tableReference": {
          "projectId": "bigquery-public-data",
          "datasetId": "cdc_places",
          "tableId": "local_data_for_better_health_county_data"
        },
        "description": "Local Data for Better Health, County Data",
        "schema": {
          "fields": [
            {
              "name": "year",
              "type": "INTEGER",
              "mode": "NULLABLE"
            },
            {
              "name": "stateabbr",
              "type": "STRING",
              "mode": "NULLABLE"
            },
            {
              "name": "statedesc",
              "type": "STRING",
              "mode": "NULLABLE"
            },
            {
              "name": "locationname",
              "type": "STRING",
              "mode": "NULLABLE"
            },
            {
              "name": "datasource",
              "type": "STRING",
              "mode": "NULLABLE"
            },
            {
              "name": "category",
              "type": "STRING",
              "mode": "NULLABLE"
            },
            {
              "name": "measure",
              "type": "STRING",
              "mode": "NULLABLE"
            },
            {
              "name": "data_value_unit",
              "type": "STRING",
              "mode": "NULLABLE"
            },
            {
              "name": "data_value_type",
              "type": "STRING",
              "mode": "NULLABLE"
            },
            {
              "name": "data_value",
              "type": "FLOAT",
              "mode": "NULLABLE"
            }
          ]
        },
        "numBytes": "234849",
        "numLongTermBytes": "0",
        "numRows": "1000",
        "creationTime": "1640891846119",
        "lastModifiedTime": "1749427268137",
        "type": "TABLE",
        "location": "US",
        "numTimeTravelPhysicalBytes": "285737",
        "numTotalLogicalBytes": "234849",
        "numActiveLogicalBytes": "234849",
        "numLongTermLogicalBytes": "0",
        "numTotalPhysicalBytes": "326557",
        "numActivePhysicalBytes": "326557",
        "numLongTermPhysicalBytes": "0",
        "numCurrentPhysicalBytes": "40820"
      }
  """
  try:
    bq_client = client.get_bigquery_client(
        project=project_id,
        credentials=credentials,
        location=settings.location,
        user_agent=[settings.application_name, "get_table_info"],
    )
    return bq_client.get_table(
        bigquery.TableReference(
            bigquery.DatasetReference(project_id, dataset_id), table_id
        )
    ).to_api_repr()
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


def list_connections(
    project_id: str,
    location: str,
    credentials: Credentials,
    settings: BigQueryToolConfig,
) -> list[dict]:
  """List BigQuery connections in a Google Cloud project and location.

  BigQuery connections are used to connect to external data sources like
  Vertex AI for AI functions (AI.CLASSIFY, AI.IF, AI.SCORE) and remote
  models (AI.GENERATE_TEXT, AI.GENERATE_EMBEDDING).

  Args:
      project_id (str): The Google Cloud project id.
      location (str): The location/region (e.g., 'us', 'eu', 'us-central1').
      credentials (Credentials): The credentials to use for the request.
      settings (BigQueryToolConfig): The BigQuery tool settings.

  Returns:
      list[dict]: List of connections with their properties.

  Examples:
      >>> list_connections("my-project", "us")
      [
        {
          "name": "projects/my-project/locations/us/connections/my_ai_connection",
          "connection_id": "my_ai_connection",
          "location": "us",
          "connection_type": "CLOUD_RESOURCE",
          "friendly_name": "My AI Connection",
          "description": "Connection for Vertex AI"
        }
      ]
  """
  try:
    from google.cloud import bigquery_connection_v1

    connection_client = bigquery_connection_v1.ConnectionServiceClient(
        credentials=credentials
    )
    parent = f"projects/{project_id}/locations/{location}"

    connections = []
    for conn in connection_client.list_connections(parent=parent):
      # Extract connection_id from the full name
      # Format: projects/{project}/locations/{location}/connections/{connection_id}
      parts = conn.name.split("/")
      connection_id = parts[-1] if parts else conn.name

      conn_info = {
          "name": conn.name,
          "connection_id": connection_id,
          "location": location,
          "friendly_name": conn.friendly_name or "",
          "description": conn.description or "",
      }

      # Add connection type based on which field is set
      if conn.cloud_resource:
        conn_info["connection_type"] = "CLOUD_RESOURCE"
        if conn.cloud_resource.service_account_id:
          conn_info["service_account"] = conn.cloud_resource.service_account_id
      elif conn.cloud_sql:
        conn_info["connection_type"] = "CLOUD_SQL"
      elif conn.spark:
        conn_info["connection_type"] = "SPARK"
      else:
        conn_info["connection_type"] = "UNKNOWN"

      connections.append(conn_info)

    return connections
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


def _grant_vertex_ai_role(
    project_id: str,
    service_account: str,
    credentials: Credentials,
) -> dict:
  """Grant the Vertex AI User role to a service account.

  Args:
      project_id (str): The Google Cloud project id.
      service_account (str): The service account email to grant the role to.
      credentials (Credentials): The credentials to use for the request.

  Returns:
      dict: Status of the IAM operation.
  """
  try:
    from google.cloud import resourcemanager_v3
    from google.iam.v1 import iam_policy_pb2, policy_pb2

    # Create the Resource Manager client
    client = resourcemanager_v3.ProjectsClient(credentials=credentials)

    # Get the current IAM policy
    resource = f"projects/{project_id}"
    request = iam_policy_pb2.GetIamPolicyRequest(resource=resource)
    policy = client.get_iam_policy(request=request)

    # Check if the binding already exists
    role = "roles/aiplatform.user"
    member = f"serviceAccount:{service_account}"

    binding_exists = False
    for binding in policy.bindings:
      if binding.role == role:
        if member in binding.members:
          binding_exists = True
          break
        else:
          # Add member to existing role binding
          binding.members.append(member)
          binding_exists = True
          break

    # If role binding doesn't exist, create a new one
    if not binding_exists:
      new_binding = policy_pb2.Binding(role=role, members=[member])
      policy.bindings.append(new_binding)

    # Set the updated IAM policy
    set_request = iam_policy_pb2.SetIamPolicyRequest(
        resource=resource, policy=policy
    )
    client.set_iam_policy(request=set_request)

    return {
        "status": "SUCCESS",
        "message": f"Granted {role} to {service_account}",
    }
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": f"Failed to grant IAM role: {str(ex)}",
    }


def create_connection(
    project_id: str,
    location: str,
    connection_id: str,
    credentials: Credentials,
    settings: BigQueryToolConfig,
    friendly_name: str = "",
    description: str = "",
    grant_vertex_ai_role: bool = True,
) -> dict:
  """Create a BigQuery Cloud Resource connection for Vertex AI.

  This creates a connection that can be used with AI functions like
  AI.CLASSIFY, AI.IF, AI.SCORE and remote models for AI.GENERATE_TEXT,
  AI.GENERATE_EMBEDDING.

  By default, this function also grants the connection's service account
  the "Vertex AI User" role, which is required to use AI functions.

  Args:
      project_id (str): The Google Cloud project id.
      location (str): The location/region (e.g., 'us', 'eu', 'us-central1').
      connection_id (str): The connection id to create (e.g., 'my_ai_connection').
      credentials (Credentials): The credentials to use for the request.
      settings (BigQueryToolConfig): The BigQuery tool settings.
      friendly_name (str): Optional friendly name for the connection.
      description (str): Optional description for the connection.
      grant_vertex_ai_role (bool): If True (default), automatically grants the
          Vertex AI User role to the connection's service account.

  Returns:
      dict: The created connection details including the service account.

  Examples:
      >>> create_connection("my-project", "us", "my_ai_connection")
      {
        "status": "SUCCESS",
        "connection_id": "my_ai_connection",
        "location": "us",
        "full_connection_path": "us.my_ai_connection",
        "service_account": "bqcx-123456789-xxxx@gcp-sa-bigquery-condel.iam.gserviceaccount.com",
        "iam_status": "Granted roles/aiplatform.user to service account"
      }
  """
  try:
    from google.cloud import bigquery_connection_v1

    connection_client = bigquery_connection_v1.ConnectionServiceClient(
        credentials=credentials
    )
    parent = f"projects/{project_id}/locations/{location}"

    # Create a Cloud Resource connection (for Vertex AI)
    connection = bigquery_connection_v1.Connection(
        friendly_name=friendly_name or connection_id,
        description=description or "BigQuery connection for Vertex AI",
        cloud_resource=bigquery_connection_v1.CloudResourceProperties(),
    )

    created_conn = connection_client.create_connection(
        parent=parent,
        connection_id=connection_id,
        connection=connection,
    )

    service_account = ""
    if created_conn.cloud_resource and created_conn.cloud_resource.service_account_id:
      service_account = created_conn.cloud_resource.service_account_id

    result = {
        "status": "SUCCESS",
        "connection_id": connection_id,
        "location": location,
        "full_connection_path": f"{location}.{connection_id}",
        "service_account": service_account,
    }

    # Automatically grant Vertex AI User role if requested
    if grant_vertex_ai_role and service_account:
      iam_result = _grant_vertex_ai_role(project_id, service_account, credentials)
      if iam_result["status"] == "SUCCESS":
        result["iam_status"] = f"Granted roles/aiplatform.user to {service_account}"
      else:
        result["iam_status"] = "FAILED"
        result["iam_error"] = iam_result["error_details"]
        result["manual_command"] = (
            f"gcloud projects add-iam-policy-binding {project_id} "
            f"--member='serviceAccount:{service_account}' "
            f"--role='roles/aiplatform.user'"
        )
    elif not grant_vertex_ai_role:
      result["note"] = (
          "IAM role not granted. To use AI functions, grant the Vertex AI User role: "
          f"gcloud projects add-iam-policy-binding {project_id} "
          f"--member='serviceAccount:{service_account}' "
          f"--role='roles/aiplatform.user'"
      )

    return result
  except Exception as ex:
    error_msg = str(ex)
    if "already exists" in error_msg.lower():
      return {
          "status": "ERROR",
          "error_details": f"Connection '{connection_id}' already exists in {location}. Use list_connections to see existing connections.",
      }
    return {
        "status": "ERROR",
        "error_details": error_msg,
    }


def get_job_info(
    project_id: str,
    job_id: str,
    credentials: Credentials,
    settings: BigQueryToolConfig,
) -> dict:
  """Get metadata information about a BigQuery job. Including slot usage,
     job configuration, job statistics, job status, original query etc.

  Args:
      project_id (str): The Google Cloud project id containing the job.
      job_id (str): The BigQuery job id.
      credentials (Credentials): The credentials to use for the request.
      settings (BigQueryToolConfig): The BigQuery tool settings.

  Returns:
      dict: Dictionary representing the properties of the job.

  Examples:
      >>> user may give job id in format of: project_id:region.job_id
      like bigquery-public-data:US.bquxjob_12345678_1234567890
      >>> get_job_info("bigquery-public-data", "bquxjob_12345678_1234567890")
      {
        "get_job_info_response": {
          "configuration": {
            "jobType": "QUERY",
            "query": {
              "destinationTable": {
                "datasetId": "_fd6de55d5d5c13fcfb0449cbf933bb695b2c3085",
                "projectId": "projectid",
                "tableId": "anonfbbe65d6_9782_469b_9f56_1392560314b2"
              },
              "priority": "INTERACTIVE",
              "query": "SELECT * FROM `projectid.dataset_id.table_id` WHERE TIMESTAMP_TRUNC(_PARTITIONTIME, DAY) = TIMESTAMP(\"2025-10-29\") LIMIT 1000",
              "useLegacySql": false,
              "writeDisposition": "WRITE_TRUNCATE"
            }
          },
          "etag": "EdeYv9sdcO7tD9HsffvcuQ==",
          "id": "projectid:US.job-id",
          "jobCreationReason": {
            "code": "REQUESTED"
          },
          "jobReference": {
            "jobId": "job-id",
            "location": "US",
            "projectId": "projectid"
          },
          "kind": "bigquery#job",
          "principal_subject": "user:abc@google.com",
          "selfLink": "https://bigquery.googleapis.com/bigquery/v2/projects/projectid/jobs/job-id?location=US",
          "statistics": {
            "creationTime": 1761760370152,
            "endTime": 1761760371250,
            "finalExecutionDurationMs": "489",
            "query": {
              "billingTier": 1,
              "cacheHit": false,
              "estimatedBytesProcessed": "5597805",
              "metadataCacheStatistics": {
                "tableMetadataCacheUsage": [
                  {
                    "explanation": "Table does not have CMETA.",
                    "tableReference": {
                      "datasetId": "datasetId",
                      "projectId": "projectid",
                      "tableId": "tableId"
                    },
                    "unusedReason": "OTHER_REASON"
                  }
                ]
              },
              "queryPlan": [
                {
                  "completedParallelInputs": "3",
                  "computeMode": "BIGQUERY",
                  "computeMsAvg": "13",
                  "computeMsMax": "15",
                  "computeRatioAvg": 0.054852320675105488,
                  "computeRatioMax": 0.063291139240506333,
                  "endMs": "1761760370422",
                  "id": "0",
                  "name": "S00: Input",
                  "parallelInputs": "8",
                  "readMsAvg": "18",
                  "readMsMax": "21",
                  "readRatioAvg": 0.0759493670886076,
                  "readRatioMax": 0.088607594936708861,
                  "recordsRead": "1690",
                  "recordsWritten": "1690",
                  "shuffleOutputBytes": "1031149",
                  "shuffleOutputBytesSpilled": "0",
                  "slotMs": "157",
                  "startMs": "1761760370388",
                  "status": "COMPLETE",
                  "steps": [
                    {
                      "kind": "READ",
                      "substeps": [
                        "$2:extendedFields.$is_not_null, $3:extendedFields.traceId, $4:span.$is_not_null, $5:span.spanKind, $6:span.endTime, $7:span.startTime, $8:span.parentSpanId, $9:span.spanId, $10:span.name, $11:span.childSpanCount.$is_not_null, $12:span.childSpanCount.value, $13:span.sameProcessAsParentSpan.$is_not_null, $14:span.sameProcessAsParentSpan.value, $15:span.status.$is_not_null, $16:span.status.message, $17:span.status.code",
                        "FROM projectid.dataset_id.table_id",
                        "WHERE equal(timestamp_trunc($1, 3), 1761696000.000000000)"
                      ]
                    },
                    {
                      "kind": "LIMIT",
                      "substeps": [
                        "1000"
                      ]
                    },
                    {
                      "kind": "WRITE",
                      "substeps": [
                        "$2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17",
                        "TO __stage00_output"
                      ]
                    }
                  ],
                  "waitMsAvg": "1",
                  "waitMsMax": "1",
                  "waitRatioAvg": 0.0042194092827004216,
                  "waitRatioMax": 0.0042194092827004216,
                  "writeMsAvg": "2",
                  "writeMsMax": "2",
                  "writeRatioAvg": 0.0084388185654008432,
                  "writeRatioMax": 0.0084388185654008432
                },
                {
                  "completedParallelInputs": "1",
                  "computeMode": "BIGQUERY",
                  "computeMsAvg": "22",
                  "computeMsMax": "22",
                  "computeRatioAvg": 0.092827004219409287,
                  "computeRatioMax": 0.092827004219409287,
                  "endMs": "1761760370428",
                  "id": "1",
                  "inputStages": [
                    "0"
                  ],
                  "name": "S01: Compute+",
                  "parallelInputs": "1",
                  "readMsAvg": "0",
                  "readMsMax": "0",
                  "readRatioAvg": 0,
                  "readRatioMax": 0,
                  "recordsRead": "1001",
                  "recordsWritten": "1000",
                  "shuffleOutputBytes": "800157",
                  "shuffleOutputBytesSpilled": "0",
                  "slotMs": "29",
                  "startMs": "1761760370398",
                  "status": "COMPLETE",
                  "steps": [
                    {
                      "kind": "READ",
                      "substeps": [
                        "$2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17",
                        "FROM __stage00_output"
                      ]
                    },
                    {
                      "kind": "COMPUTE",
                      "substeps": [
                        "$130 := MAKE_STRUCT($3, $2)",
                        "$131 := MAKE_STRUCT($10, $9, $8, MAKE_STRUCT($29, $28, $27), $7, $6, MAKE_STRUCT(...), MAKE_STRUCT(...), MAKE_STRUCT(...), ...)"
                      ]
                    },
                    {
                      "kind": "LIMIT",
                      "substeps": [
                        "1000"
                      ]
                    },
                    {
                      "kind": "WRITE",
                      "substeps": [
                        "$130, $131",
                        "TO __stage01_output"
                      ]
                    }
                  ],
                  "waitMsAvg": "7",
                  "waitMsMax": "7",
                  "waitRatioAvg": 0.029535864978902954,
                  "waitRatioMax": 0.029535864978902954,
                  "writeMsAvg": "4",
                  "writeMsMax": "4",
                  "writeRatioAvg": 0.016877637130801686,
                  "writeRatioMax": 0.016877637130801686
                },
                {
                  "completedParallelInputs": "1",
                  "computeMode": "BIGQUERY",
                  "computeMsAvg": "33",
                  "computeMsMax": "33",
                  "computeRatioAvg": 0.13924050632911392,
                  "computeRatioMax": 0.13924050632911392,
                  "endMs": "1761760370745",
                  "id": "2",
                  "inputStages": [
                    "1"
                  ],
                  "name": "S02: Output",
                  "parallelInputs": "1",
                  "readMsAvg": "0",
                  "readMsMax": "0",
                  "readRatioAvg": 0,
                  "readRatioMax": 0,
                  "recordsRead": "1000",
                  "recordsWritten": "1000",
                  "shuffleOutputBytes": "459829",
                  "shuffleOutputBytesSpilled": "0",
                  "slotMs": "106",
                  "startMs": "1761760370667",
                  "status": "COMPLETE",
                  "steps": [
                    {
                      "kind": "READ",
                      "substeps": [
                        "$130, $131",
                        "FROM __stage01_output"
                      ]
                    },
                    {
                      "kind": "WRITE",
                      "substeps": [
                        "$130, $131",
                        "TO __stage02_output"
                      ]
                    }
                  ],
                  "waitMsAvg": "237",
                  "waitMsMax": "237",
                  "waitRatioAvg": 1,
                  "waitRatioMax": 1,
                  "writeMsAvg": "55",
                  "writeMsMax": "55",
                  "writeRatioAvg": 0.2320675105485232,
                  "writeRatioMax": 0.2320675105485232
                }
              ],
              "referencedTables": [
                {
                  "datasetId": "dataset_id",
                  "projectId": "projectid",
                  "tableId": "table_id"
                }
              ],
              "statementType": "SELECT",
              "timeline": [
                {
                  "completedUnits": "5",
                  "elapsedMs": "492",
                  "estimatedRunnableUnits": "0",
                  "pendingUnits": "5",
                  "totalSlotMs": "293"
                }
              ],
              "totalBytesBilled": "10485760",
              "totalBytesProcessed": "5597805",
              "totalPartitionsProcessed": "2",
              "totalSlotMs": "293",
              "transferredBytes": "0"
            },
            "startTime": 1761760370268,
            "totalBytesProcessed": "5597805",
            "totalSlotMs": "293"
          },
          "status": {
            "state": "DONE"
          },
          "user_email": "abc@google.com"
        }
      }
  """
  try:
    bq_client = client.get_bigquery_client(
        project=project_id,
        credentials=credentials,
        location=settings.location,
        user_agent=[settings.application_name, "get_job_info"],
    )

    job = bq_client.get_job(job_id)
    # We need to use _properties to get the job info because it contains all
    # the job info.
    # pylint: disable=protected-access
    return job._properties
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }
