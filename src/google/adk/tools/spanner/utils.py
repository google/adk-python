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

from __future__ import annotations

import asyncio
import itertools
import json
import logging
from typing import Generator
from typing import Iterable
from typing import Optional
from typing import TYPE_CHECKING

from google.auth.credentials import Credentials
from google.cloud.spanner_admin_database_v1.types import DatabaseDialect

from . import client
from ...features import experimental
from ...features import FeatureName
from ..tool_context import ToolContext
from .settings import QueryResultMode
from .settings import SpannerToolSettings
from .settings import SpannerVectorStoreSettings

if TYPE_CHECKING:
  from google.cloud import spanner
  from google.genai import Client

logger = logging.getLogger("google_adk." + __name__)

DEFAULT_MAX_EXECUTED_QUERY_RESULT_ROWS = 50


def execute_sql(
    project_id: str,
    instance_id: str,
    database_id: str,
    query: str,
    credentials: Credentials,
    settings: SpannerToolSettings,
    tool_context: ToolContext,
    params: Optional[dict] = None,
    params_types: Optional[dict] = None,
) -> dict:
  """Utility function to run a Spanner Read-Only query in the spanner database and return the result.

  Args:
      project_id (str): The GCP project id in which the spanner database
        resides.
      instance_id (str): The instance id of the spanner database.
      database_id (str): The database id of the spanner database.
      query (str): The Spanner SQL query to be executed.
      credentials (Credentials): The credentials to use for the request.
      settings (SpannerToolSettings): The settings for the tool.
      tool_context (ToolContext): The context for the tool.
      params (dict): values for parameter replacement.  Keys must match the
        names used in ``query``.
      params_types (dict): maps explicit types for one or more param values.

  Returns:
      dict: Dictionary with the result of the query.
            If the result contains the key "result_is_likely_truncated" with
            value True, it means that there may be additional rows matching the
            query not returned in the result.
  """

  try:
    # Get Spanner client
    spanner_client = client.get_spanner_client(
        project=project_id, credentials=credentials
    )
    instance = spanner_client.instance(instance_id)
    database = instance.database(database_id)

    if database.database_dialect == DatabaseDialect.POSTGRESQL:
      return {
          "status": "ERROR",
          "error_details": "PostgreSQL dialect is not supported.",
      }

    with database.snapshot() as snapshot:
      result_set = snapshot.execute_sql(
          sql=query, params=params, param_types=params_types
      )
      rows = []
      counter = (
          settings.max_executed_query_result_rows
          if settings and settings.max_executed_query_result_rows > 0
          else DEFAULT_MAX_EXECUTED_QUERY_RESULT_ROWS
      )
      if settings and settings.query_result_mode is QueryResultMode.DICT_LIST:
        result_set = result_set.to_dict_list()

      for row in result_set:
        try:
          # if the json serialization of the row succeeds, use it as is
          json.dumps(row)
        except (TypeError, ValueError, OverflowError):
          row = str(row)

        rows.append(row)
        counter -= 1
        if counter <= 0:
          break

      result = {"status": "SUCCESS", "rows": rows}
      if counter <= 0:
        result["result_is_likely_truncated"] = True
      return result
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


def embed_contents(
    vertex_ai_embedding_model_name: str,
    contents: list[str],
    output_dimensionality: Optional[int] = None,
    genai_client: Client | None = None,
) -> list[list[float]]:
  """Embed the given contents into list of vectors using the Vertex AI embedding model endpoint."""
  try:
    from google.genai import Client
    from google.genai.types import EmbedContentConfig

    genai_client = genai_client or Client()
    config = EmbedContentConfig()
    if output_dimensionality:
      config.output_dimensionality = output_dimensionality
    response = genai_client.models.embed_content(
        model=vertex_ai_embedding_model_name,
        contents=contents,
        config=config,
    )
    return [list(e.values) for e in response.embeddings]
  except Exception as ex:
    raise RuntimeError(f"Failed to embed content: {ex!r}") from ex


async def embed_contents_async(
    vertex_ai_embedding_model_name: str,
    contents: list[str],
    output_dimensionality: Optional[int] = None,
    genai_client: Client | None = None,
) -> list[list[float]]:
  """Embed the given contents into list of vectors using the Vertex AI embedding model endpoint."""
  try:
    from google.genai import Client
    from google.genai.types import EmbedContentConfig

    genai_client = genai_client or Client()
    config = EmbedContentConfig()
    if output_dimensionality:
      config.output_dimensionality = output_dimensionality
    response = await genai_client.aio.models.embed_content(
        model=vertex_ai_embedding_model_name,
        contents=contents,
        config=config,
    )
    return [list(e.values) for e in response.embeddings]
  except Exception as ex:
    raise RuntimeError(f"Failed to embed content: {ex!r}") from ex


@experimental(FeatureName.SPANNER_VECTOR_STORE)
class SpannerVectorStore:
  """A class for orchestrating and providing utility functions for a Spanner vector store.

  This class provides utility functions for setting up and adding contents to a
  vector store table in a Google Cloud Spanner database, based on the given
  Spanner tool settings.
  """

  DEFAULT_VECTOR_STORE_ID_COLUMN_NAME = "id"
  SPANNER_VECTOR_STORE_USER_AGENT = "adk-spanner-vector-store"

  def __init__(
      self,
      settings: SpannerToolSettings,
      credentials: Credentials | None = None,
      spanner_client: spanner.Client | None = None,
      genai_client: Client | None = None,
  ):
    """Initializes the SpannerVectorStore with validated settings and clients.

    This constructor sets up the connection to a specific Spanner database and
    configures the necessary clients for vector operations.

    Args:
      settings (SpannerToolSettings): The settings for the tool.
      credentials (Credentials | None): Credentials for Spanner operations. This
        is used to initialize a new Spanner client only if `spanner_client`
        is not explicitly provided.
      spanner_client (spanner.Client | None): An pre-configured `spanner.Client`
        instance. If not provided, a new client will be created.
      genai_client (Client | None): Google GenAI client used for
        generating vector embeddings.
    """

    if not settings.vector_store_settings:
      raise ValueError("Spanner vector store settings are not set.")

    self._settings = settings

    if not spanner_client:
      self._spanner_client = client.get_spanner_client(
          project=self._vector_store_settings.project_id,
          credentials=credentials,
      )
    else:
      self._spanner_client = spanner_client
      client_user_agent = self._spanner_client._client_info.user_agent
      if not client_user_agent:
        self._spanner_client._client_info.user_agent = client.USER_AGENT
      elif client.USER_AGENT not in client_user_agent:
        self._spanner_client._client_info.user_agent = " ".join(
            [client_user_agent, client.USER_AGENT]
        )
    self._spanner_client._client_info.user_agent = " ".join([
        self._spanner_client._client_info.user_agent,
        self.SPANNER_VECTOR_STORE_USER_AGENT,
    ])

    instance = self._spanner_client.instance(
        self._vector_store_settings.instance_id
    )
    if not instance.exists():
      raise ValueError(
          "Instance id {} doesn't exist.".format(
              self._vector_store_settings.instance_id
          )
      )
    self._database = instance.database(self._vector_store_settings.database_id)
    if not self._database.exists():
      raise ValueError(
          "Database id {} doesn't exist.".format(
              self._vector_store_settings.database_id
          )
      )

    self._genai_client = genai_client

  @property
  def _vector_store_settings(self) -> SpannerVectorStoreSettings:
    """Returns the Spanner vector store settings."""

    if self._settings.vector_store_settings is None:
      raise ValueError("Spanner vector store settings are not set.")
    return self._settings.vector_store_settings

  def _create_vector_store_table_ddl(
      self,
      dialect: DatabaseDialect,
  ) -> str:
    """Creates the DDL statements necessary to define a vector store table in Spanner.

    The vector store table is created based on the given settings.
    - **id_column** (STRING or text): The default primary key, typically a UUID.
      Note: This column is only included in the DDL when `primary_key_columns`
      is not specified in the settings.
    - **content_column** (STRING or text): The source text content used to
      generate the embedding.
    - **embedding_column** (ARRAY<FLOAT32> or float4[]): The vector embedding
      column corresponding to the content.
    - **additional_columns_to_setup** (provided in the settings): Additional
      columns to be added to the vector store table.

    Args:
      dialect: The database dialect (e.g., GOOGLE_STANDARD_SQL or POSTGRESQL)
        governing the DDL syntax.

    Returns:
      A DDL statement string defining the vector store table.
    """

    primary_key_columns = self._vector_store_settings.primary_key_columns or [
        self.DEFAULT_VECTOR_STORE_ID_COLUMN_NAME
    ]

    column_definitions = []

    if self._vector_store_settings.primary_key_columns is None:
      if dialect == DatabaseDialect.POSTGRESQL:
        column_definitions.append(
            f"{self.DEFAULT_VECTOR_STORE_ID_COLUMN_NAME} varchar(36) DEFAULT"
            " spanner.generate_uuid()"
        )
      else:
        column_definitions.append(
            f"{self.DEFAULT_VECTOR_STORE_ID_COLUMN_NAME} STRING(36) DEFAULT"
            " (GENERATE_UUID())"
        )

    # Additional Columns
    if self._vector_store_settings.additional_columns_to_setup:
      for column in self._vector_store_settings.additional_columns_to_setup:
        null_stmt = "" if column.is_nullable else " NOT NULL"
        column_definitions.append(f"{column.name} {column.type}{null_stmt}")

    # Content and Embedding Columns
    if dialect == DatabaseDialect.POSTGRESQL:
      column_definitions.append(
          f"{self._vector_store_settings.content_column} text"
      )
      column_definitions.append(
          f"{self._vector_store_settings.embedding_column} float4[] "
          f"VECTOR LENGTH {self._vector_store_settings.vector_length}"
      )
    else:
      column_definitions.append(
          f"{self._vector_store_settings.content_column} STRING(MAX)"
      )
      column_definitions.append(
          f"{self._vector_store_settings.embedding_column} "
          f"ARRAY<FLOAT32>(vector_length=>{self._vector_store_settings.vector_length})"
      )

    inner_ddl = ",\n  ".join(column_definitions)
    pk_stmt = ", ".join(primary_key_columns)

    if dialect == DatabaseDialect.POSTGRESQL:
      return (
          f"CREATE TABLE IF NOT EXISTS {self._vector_store_settings.table_name}"
          f" (\n  {inner_ddl},\n  PRIMARY KEY({pk_stmt})\n)"
      )
    else:
      return (
          f"CREATE TABLE IF NOT EXISTS {self._vector_store_settings.table_name}"
          f" (\n  {inner_ddl}\n) PRIMARY KEY({pk_stmt})"
      )

  def _create_ann_vector_search_index_ddl(
      self,
      dialect: DatabaseDialect,
  ) -> str:
    """Create a DDL statement to create a vector search index for ANN.

    Args:
      dialect: The database dialect (e.g., GOOGLE_STANDARD_SQL or POSTGRESQL)
        governing the DDL syntax.

    Returns:
      A DDL statement string to create the vector search index.
    """

    # This is only required when the nearest neighbors search algorithm is
    # APPROXIMATE_NEAREST_NEIGHBORS.
    if not self._vector_store_settings.vector_search_index_settings:
      raise ValueError("Vector search index settings are not set.")

    if dialect != DatabaseDialect.GOOGLE_STANDARD_SQL:
      raise ValueError(
          "ANN is only supported for the Google Standard SQL dialect."
      )

    index_columns = [self._vector_store_settings.embedding_column]
    if (
        self._vector_store_settings.vector_search_index_settings.additional_key_columns
    ):
      index_columns.extend(
          self._vector_store_settings.vector_search_index_settings.additional_key_columns
      )

    statement = (
        "CREATE VECTOR INDEX IF NOT EXISTS"
        f" {self._vector_store_settings.vector_search_index_settings.index_name}\n\tON"
        f" {self._vector_store_settings.table_name}({', '.join(index_columns)})"
    )

    if (
        self._vector_store_settings.vector_search_index_settings.additional_storing_columns
    ):
      statement += (
          "\n\tSTORING"
          f" ({', '.join(self._vector_store_settings.vector_search_index_settings.additional_storing_columns)})"
      )

    statement += (
        f"\n\tWHERE {self._vector_store_settings.embedding_column} IS NOT NULL"
    )

    options_segments = [
        f"distance_type='{self._vector_store_settings.distance_type}'"
    ]

    if (
        getattr(
            self._vector_store_settings.vector_search_index_settings,
            "tree_depth",
            0,
        )
        > 0
    ):
      tree_depth = (
          self._vector_store_settings.vector_search_index_settings.tree_depth
      )
      if tree_depth not in (2, 3):
        raise ValueError(
            f"Vector search index settings: tree_depth: {tree_depth} must be"
            " either 2 or 3"
        )
      options_segments.append(
          f"tree_depth={self._vector_store_settings.vector_search_index_settings.tree_depth}"
      )

    if (
        self._vector_store_settings.vector_search_index_settings.num_branches
        is not None
        and self._vector_store_settings.vector_search_index_settings.num_branches
        > 0
    ):
      options_segments.append(
          f"num_branches={self._vector_store_settings.vector_search_index_settings.num_branches}"
      )

    if self._vector_store_settings.vector_search_index_settings.num_leaves > 0:
      options_segments.append(
          f"num_leaves={self._vector_store_settings.vector_search_index_settings.num_leaves}"
      )

    statement += "\n\tOPTIONS(" + ", ".join(options_segments) + ")"

    return statement.strip()

  def create_vector_store(self):
    """Creates a new vector store within the Google Cloud Spanner database.

    Raises:
        RuntimeError: If the DDL statement execution against Spanner fails.
    """
    try:
      ddl = self._create_vector_store_table_ddl(self._database.database_dialect)
      logger.debug(
          "Executing DDL statement to create vector store table: %s", ddl
      )
      operation = self._database.update_ddl([ddl])

      # Wait for completion
      logger.info("Waiting for update database operation to complete...")
      operation.result()

      logger.debug(
          "Successfully created the vector store table: %s in Spanner"
          " database: projects/%s/instances/%s/databases/%s",
          self._vector_store_settings.table_name,
          self._vector_store_settings.project_id,
          self._vector_store_settings.instance_id,
          self._vector_store_settings.database_id,
      )
    except Exception as e:
      logger.error("Failed to create the vector store. Error: %s", e)
      raise

  def create_vector_search_index(self):
    """Creates a vector search index within the Google Cloud Spanner database.

    Raises:
        RuntimeError: If the DDL statement execution against Spanner fails.
    """
    try:
      if not self._vector_store_settings.vector_search_index_settings:
        logger.warning("No vector search index settings found.")
        return

      ddl = self._create_ann_vector_search_index_ddl(
          self._database.database_dialect
      )
      logger.debug(
          "Executing DDL statement to create vector search index: %s", ddl
      )
      operation = self._database.update_ddl([ddl])

      # Wait for completion
      logger.info("Waiting for update database operation to complete...")
      operation.result()

      logger.debug(
          "Successfully created the vector search index: %s in Spanner"
          " database: projects/%s/instances/%s/databases/%s",
          self._vector_store_settings.vector_search_index_settings.index_name,
          self._vector_store_settings.project_id,
          self._vector_store_settings.instance_id,
          self._vector_store_settings.database_id,
      )

    except Exception as e:
      logger.error("Failed to create the vector search index. Error: %s", e)
      raise

  async def create_vector_store_async(self):
    """Asynchronously creates a new vector store within the Google Cloud Spanner database.

    Raises:
        RuntimeError: If the DDL statement execution against Spanner fails.
    """
    await asyncio.to_thread(self.create_vector_store)

  async def create_vector_search_index_async(self):
    """Asynchronously creates a vector search index within the Google Cloud Spanner database.

    Raises:
        RuntimeError: If the DDL statement execution against Spanner fails.
    """
    await asyncio.to_thread(self.create_vector_search_index)

  def _prepare_and_validate_batches(
      self,
      contents: Iterable[str],
      additional_columns_values: Iterable[dict] | None,
      batch_size: int,
  ) -> Generator[tuple[list[str], list[dict], int], None, None]:
    """Prepares and validates batches of contents and additional columns for insertion into the vector store."""
    content_iter = iter(contents)

    value_iter = (
        iter(additional_columns_values)
        if additional_columns_values is not None
        else itertools.repeat({})
    )

    batches = iter(lambda: list(itertools.islice(content_iter, batch_size)), [])

    for index, content_batch in enumerate(batches):
      actual_index = index * batch_size
      value_batch = list(itertools.islice(value_iter, len(content_batch)))

      if len(value_batch) < len(content_batch):
        raise ValueError(
            f"Data mismatch: ended at index {actual_index}. Expected"
            f" {len(content_batch)} values for this batch, but got"
            f" {len(value_batch)}."
        )

      yield (content_batch, value_batch, actual_index)

    if additional_columns_values is not None:
      if next(value_iter, None) is not None:
        raise ValueError(
            "additional_columns_values contains more items than contents."
        )

  def add_contents(
      self,
      contents: Iterable[str],
      *,
      additional_columns_values: Iterable[dict] | None = None,
      batch_size: int = 200,
  ):
    """Adds text contents to the vector store.

    Performs batch embedding generation and subsequent insertion of the contents
    into the vector store table in the Google Cloud Spanner database.

    Args:
        contents (Iterable[str]): An iterable collection of string contents to
          be added to the vector store.
        additional_columns_values (Iterable[dict] | None): An optional iterable
          of dictionary containing values for additional columns to be stored
          with the content row. Keys must match column names.
        batch_size (int): The maximum number of items to process and insert in a
          single batch. Defaults to 200.
    """
    total_rows = 0
    try:
      self._database.reload()

      cols = [
          c.name
          for c in self._vector_store_settings.additional_columns_to_setup or []
      ]

      batch_gen = self._prepare_and_validate_batches(
          contents, additional_columns_values, batch_size
      )

      for content_b, extra_b, batch_index in batch_gen:
        logger.debug(
            "Embedding content batch %d to %d (size: %d)...",
            batch_index,
            batch_index + len(content_b),
            len(content_b),
        )
        embeddings = embed_contents(
            self._vector_store_settings.vertex_ai_embedding_model_name,
            content_b,
            self._vector_store_settings.vector_length,
            self._genai_client,
        )

        logger.debug(
            "Committing batch mutation %d to %d (size: %d).",
            batch_index,
            batch_index + len(content_b),
            len(content_b),
        )
        mutation_rows = [
            # [content, embedding, ...additional_columns]
            [c, e, *map(extra.get, cols)]
            for c, e, extra in zip(content_b, embeddings, extra_b)
        ]
        with self._database.batch() as batch:
          batch.insert_or_update(
              table=self._vector_store_settings.table_name,
              columns=[
                  self._vector_store_settings.content_column,
                  self._vector_store_settings.embedding_column,
              ]
              + cols,
              values=mutation_rows,
          )

        total_rows += len(mutation_rows)

      logger.debug(
          "Successfully added %d contents to the vector store table: %s in"
          " Spanner database: projects/%s/instances/%s/databases/%s",
          total_rows,
          self._vector_store_settings.table_name,
          self._vector_store_settings.project_id,
          self._vector_store_settings.instance_id,
          self._vector_store_settings.database_id,
      )

    except Exception as e:
      logger.error(
          "Failed to finish adding contents to the vector store table: %s in"
          " Spanner database: projects/%s/instances/%s/databases/%s. Total"
          " rows added: %d. Error: %s",
          self._vector_store_settings.table_name,
          self._vector_store_settings.project_id,
          self._vector_store_settings.instance_id,
          self._vector_store_settings.database_id,
          total_rows,
          e,
      )
      raise

  async def add_contents_async(
      self,
      contents: Iterable[str],
      *,
      additional_columns_values: Iterable[dict] | None = None,
      batch_size: int = 200,
  ):
    """Asynchronously adds text contents to the vector store.

    Performs batch embedding generation and subsequent insertion of the contents
    into the vector store table in the Google Cloud Spanner database.

    Args:
        contents (Iterable[str]): An iterable collection of string contents to
          be added to the vector store.
        additional_columns_values (Iterable[dict] | None): An optional iterable
          of dictionary containing values for additional columns to be stored
          with the content row. Keys must match column names.
        batch_size (int): The maximum number of items to process and insert in a
          single batch. Defaults to 200.
    """
    total_rows = 0
    try:
      await asyncio.to_thread(self._database.reload)

      cols = [
          c.name
          for c in self._vector_store_settings.additional_columns_to_setup or []
      ]

      batch_gen = self._prepare_and_validate_batches(
          contents, additional_columns_values, batch_size
      )

      for content_b, extra_b, batch_index in batch_gen:
        logger.debug(
            "Embedding content batch %d to %d (size: %d)...",
            batch_index,
            batch_index + len(content_b),
            len(content_b),
        )
        embeddings = await embed_contents_async(
            self._vector_store_settings.vertex_ai_embedding_model_name,
            content_b,
            self._vector_store_settings.vector_length,
            self._genai_client,
        )

        logger.debug(
            "Committing batch mutation %d to %d (size: %d).",
            batch_index,
            batch_index + len(content_b),
            len(content_b),
        )
        mutation_rows = [
            # [content, embedding, ...additional_columns]
            [c, e, *map(extra.get, cols)]
            for c, e, extra in zip(content_b, embeddings, extra_b)
        ]

        def _commit_batch(columns, rows_to_commit):
          with self._database.batch() as batch:
            batch.insert_or_update(
                table=self._vector_store_settings.table_name,
                columns=[
                    self._vector_store_settings.content_column,
                    self._vector_store_settings.embedding_column,
                ]
                + columns,
                values=rows_to_commit,
            )

        await asyncio.to_thread(_commit_batch, cols, mutation_rows)
        total_rows += len(mutation_rows)

      logger.debug(
          "Successfully added %d contents to the vector store table: %s in"
          " Spanner database: projects/%s/instances/%s/databases/%s",
          total_rows,
          self._vector_store_settings.table_name,
          self._vector_store_settings.project_id,
          self._vector_store_settings.instance_id,
          self._vector_store_settings.database_id,
      )

    except Exception as e:
      logger.error(
          "Failed to finish adding contents to the vector store table: %s in"
          " Spanner database: projects/%s/instances/%s/databases/%s. Total"
          " rows added: %d. Error: %s",
          self._vector_store_settings.table_name,
          self._vector_store_settings.project_id,
          self._vector_store_settings.instance_id,
          self._vector_store_settings.database_id,
          total_rows,
          e,
      )
      raise
