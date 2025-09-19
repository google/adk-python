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
        user_agent=settings.application_name,
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
        user_agent=settings.application_name,
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
        user_agent=settings.application_name,
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
        user_agent=settings.application_name,
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
