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

from google.cloud import bigquery
from google.oauth2.credentials import Credentials

from ...tools.bigquery import client


def list_dataset_ids(project_id: str, credentials: Credentials) -> list[str]:
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
    bq_client = client.get_bigquery_client(credentials=credentials)

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
    project_id: str, dataset_id: str, credentials: Credentials
) -> dict:
  """Get metadata information about a BigQuery dataset.

  Args:
      project_id (str): The Google Cloud project id containing the dataset.
      dataset_id (str): The BigQuery dataset id.
      credentials (Credentials): The credentials to use for the request.

  Returns:
      dict: Dictionary representing the properties of the dataset.

  Examples:
      >>> get_dataset_info("bigquery-public-data", "penguins")
      {
        "kind": "bigquery#dataset",
        "etag": "PNC5907iQbzeVcAru/2L3A==",
        "id": "bigquery-public-data:ml_datasets",
        "selfLink":
          "https://bigquery.googleapis.com/bigquery/v2/projects/bigquery-public-data/datasets/ml_datasets",
        "datasetReference": {
            "datasetId": "ml_datasets",
            "projectId": "bigquery-public-data"
        },
        "access": [
            {
                "role": "OWNER",
                "groupByEmail": "cloud-datasets-eng@google.com"
            },
            {
                "role": "READER",
                "iamMember": "allUsers"
            },
            {
                "role": "READER",
                "groupByEmail": "bqml-eng@google.com"
            }
        ],
        "creationTime": "1553208775542",
        "lastModifiedTime": "1686338918114",
        "location": "US",
        "type": "DEFAULT",
        "maxTimeTravelHours": "168"
      }
  """
  try:
    bq_client = client.get_bigquery_client(credentials=credentials)
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
    project_id: str, dataset_id: str, credentials: Credentials
) -> list[str]:
  """List table ids in a BigQuery dataset.

  Args:
      project_id (str): The Google Cloud project id containing the dataset.
      dataset_id (str): The BigQuery dataset id.
      credentials (Credentials): The credentials to use for the request.

  Returns:
      list[str]: List of the tables ids present in the dataset.

  Examples:
      >>> list_table_ids("bigquery-public-data", "ml_datasets")
      ['census_adult_income',
       'credit_card_default',
       'holidays_and_events_for_forecasting',
       'iris',
       'penguins',
       'ulb_fraud_detection']
  """
  try:
    bq_client = client.get_bigquery_client(credentials=credentials)

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
    project_id: str, dataset_id: str, table_id: str, credentials: Credentials
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
      >>> get_table_info("bigquery-public-data", "ml_datasets", "penguins")
      {
        "kind": "bigquery#table",
        "etag": "X0ZkRohSGoYvWemRYEgOHA==",
        "id": "bigquery-public-data:ml_datasets.penguins",
        "selfLink":
        "https://bigquery.googleapis.com/bigquery/v2/projects/bigquery-public-data/datasets/ml_datasets/tables/penguins",
        "tableReference": {
            "projectId": "bigquery-public-data",
            "datasetId": "ml_datasets",
            "tableId": "penguins"
        },
        "schema": {
            "fields": [
                {
                    "name": "species",
                    "type": "STRING",
                    "mode": "REQUIRED"
                },
                {
                    "name": "island",
                    "type": "STRING",
                    "mode": "NULLABLE"
                },
                {
                    "name": "culmen_length_mm",
                    "type": "FLOAT",
                    "mode": "NULLABLE"
                },
                {
                    "name": "culmen_depth_mm",
                    "type": "FLOAT",
                    "mode": "NULLABLE"
                },
                {
                    "name": "flipper_length_mm",
                    "type": "FLOAT",
                    "mode": "NULLABLE"
                },
                {
                    "name": "body_mass_g",
                    "type": "FLOAT",
                    "mode": "NULLABLE"
                },
                {
                    "name": "sex",
                    "type": "STRING",
                    "mode": "NULLABLE"
                }
            ]
        },
        "numBytes": "28947",
        "numLongTermBytes": "28947",
        "numRows": "344",
        "creationTime": "1619804743188",
        "lastModifiedTime": "1634584675234",
        "type": "TABLE",
        "location": "US",
        "numTimeTravelPhysicalBytes": "0",
        "numTotalLogicalBytes": "28947",
        "numActiveLogicalBytes": "0",
        "numLongTermLogicalBytes": "28947",
        "numTotalPhysicalBytes": "5350",
        "numActivePhysicalBytes": "0",
        "numLongTermPhysicalBytes": "5350",
        "numCurrentPhysicalBytes": "5350"
      }
  """
  try:
    bq_client = client.get_bigquery_client(credentials=credentials)
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
