# Google API Tools Sample

## Introduction

This sample tests and demos Google API tools available in the
`google.adk.tools.google_api_tool` module. We pick the following BigQuery API
tools for this sample agent:

1. `bigquery_datasets_list`: List user's datasets.

2. `bigquery_datasets_get`: Get a dataset's details.

3. `bigquery_datasets_insert`: Create a new dataset.

4. `bigquery_tables_list`: List all tables in a dataset.

5. `bigquery_tables_get`: Get a table's details.

6. `bigquery_tables_insert`: Insert a new table into a dataset.

## How to use

1. Follow https://developers.google.com/identity/protocols/oauth2#1.-obtain-oauth-2.0-credentials-from-the-dynamic_data.setvar.console_name. to get your client id and client secret.
  Be sure to choose "web" as your client type.

2. Configure your `.env` file to add two variables:

  * OAUTH_CLIENT_ID={your client id}
  * OAUTH_CLIENT_SECRET={your client secret}

  Note: don't create a separate `.env` file , instead put it to the same `.env` file that stores your Vertex AI or Dev ML credentials

3. Follow https://developers.google.com/identity/protocols/oauth2/web-server#creatingcred to add http://localhost/dev-ui/ to "Authorized redirect URIs".

  Note: localhost here is just a hostname that you use to access the dev ui, replace it with the actual hostname you use to access the dev ui.

4. For 1st run, allow popup for localhost in Chrome.

## Sample prompt

* `Do I have any datasets in project sean-dev-agent ?`
* `Do I have any tables under it ?`
* `could you get me the details of this table ?`
* `Can you help to create a new dataset in the same project? id : sean_test , location: us`
* `could you show me the details of this new dataset ?`
* `could you create a new table under this dataset ? table name : sean_test_table. column1 : name is id , type is integer, required. column2 : name is info , type is string, required. column3 : name is backup , type is string, optional.`
