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

"""Data source tools for the Data Analysis Agent."""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Union, BinaryIO
import urllib.parse
import urllib.request

import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FileUploadRequest(BaseModel):
    """Request to upload a file for data analysis."""
    
    file_path: str = Field(..., description="The path to the file to upload.")
    file_type: str = Field("auto", description="The type of the file (csv, json, excel, etc.).")
    options: Dict[str, Any] = Field(
        default_factory=dict, description="Additional options for loading the file."
    )
    max_file_size_mb: float = Field(
        10.0, description="Maximum file size in megabytes."
    )


class FileUploadResponse(BaseModel):
    """Response from uploading a file."""
    
    success: bool = Field(..., description="Whether the file was uploaded successfully.")
    message: str = Field(..., description="A message describing the result.")
    data_id: Optional[str] = Field(
        None, description="An identifier for the loaded data."
    )
    file_info: Dict[str, Any] = Field(
        default_factory=dict, description="Information about the uploaded file."
    )


class GoogleSheetRequest(BaseModel):
    """Request to load data from a Google Sheet."""
    
    sheet_url: str = Field(..., description="The URL of the Google Sheet.")
    sheet_name: Optional[str] = Field(
        None, description="The name of the sheet to load. If None, loads the first sheet."
    )
    range: Optional[str] = Field(
        None, description="The range of cells to load (e.g., 'A1:D10')."
    )
    options: Dict[str, Any] = Field(
        default_factory=dict, description="Additional options for loading the sheet."
    )


class GoogleSheetResponse(BaseModel):
    """Response from loading a Google Sheet."""
    
    success: bool = Field(..., description="Whether the sheet was loaded successfully.")
    message: str = Field(..., description="A message describing the result.")
    data_id: Optional[str] = Field(
        None, description="An identifier for the loaded data."
    )
    sheet_info: Dict[str, Any] = Field(
        default_factory=dict, description="Information about the loaded sheet."
    )


class DatabaseConnectionRequest(BaseModel):
    """Request to connect to a database."""
    
    connection_type: str = Field(..., description="The type of database connection (mysql, postgresql, sqlite, etc.).")
    connection_string: str = Field(..., description="The connection string for the database.")
    query: str = Field(..., description="The SQL query to execute.")
    options: Dict[str, Any] = Field(
        default_factory=dict, description="Additional options for the database connection."
    )


class DatabaseConnectionResponse(BaseModel):
    """Response from connecting to a database."""
    
    success: bool = Field(..., description="Whether the connection was successful.")
    message: str = Field(..., description="A message describing the result.")
    data_id: Optional[str] = Field(
        None, description="An identifier for the loaded data."
    )
    connection_info: Dict[str, Any] = Field(
        default_factory=dict, description="Information about the database connection."
    )


class APIDataRequest(BaseModel):
    """Request to load data from an API."""
    
    api_url: str = Field(..., description="The URL of the API.")
    method: str = Field("GET", description="The HTTP method to use (GET, POST, etc.).")
    headers: Dict[str, str] = Field(
        default_factory=dict, description="Headers to include in the request."
    )
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters to include in the request."
    )
    body: Optional[Dict[str, Any]] = Field(
        None, description="Body to include in the request (for POST, PUT, etc.)."
    )
    response_format: str = Field(
        "json", description="The expected format of the response (json, csv, etc.)."
    )


class APIDataResponse(BaseModel):
    """Response from loading data from an API."""
    
    success: bool = Field(..., description="Whether the data was loaded successfully.")
    message: str = Field(..., description="A message describing the result.")
    data_id: Optional[str] = Field(
        None, description="An identifier for the loaded data."
    )
    api_info: Dict[str, Any] = Field(
        default_factory=dict, description="Information about the API request."
    )


class DataSourceTools:
    """Tools for loading data from various sources."""

    def __init__(self, data_store: Dict[str, pd.DataFrame], temp_dir: str):
        """Initialize the data source tools.
        
        Args:
            data_store: A dictionary mapping data IDs to DataFrames.
            temp_dir: A temporary directory for storing uploaded files.
        """
        self._data_store = data_store
        self._temp_dir = temp_dir
    
    async def upload_file(self, request: FileUploadRequest) -> FileUploadResponse:
        """Upload a file for data analysis.
        
        Args:
            request: The request containing the file path and type.
            
        Returns:
            A response indicating whether the file was uploaded successfully.
        """
        try:
            # Check if the file exists
            if not os.path.exists(request.file_path):
                return FileUploadResponse(
                    success=False,
                    message=f"File '{request.file_path}' not found.",
                )
            
            # Check the file size
            file_size_mb = os.path.getsize(request.file_path) / (1024 * 1024)
            if file_size_mb > request.max_file_size_mb:
                return FileUploadResponse(
                    success=False,
                    message=f"File size ({file_size_mb:.2f} MB) exceeds the maximum allowed size ({request.max_file_size_mb} MB).",
                )
            
            # Determine the file type if set to auto
            file_type = request.file_type.lower()
            if file_type == "auto":
                if request.file_path.endswith(".csv"):
                    file_type = "csv"
                elif request.file_path.endswith(".json"):
                    file_type = "json"
                elif request.file_path.endswith((".xls", ".xlsx")):
                    file_type = "excel"
                else:
                    return FileUploadResponse(
                        success=False,
                        message=f"Could not determine file type for '{request.file_path}'.",
                    )
            
            # Load the file based on the type
            if file_type == "csv":
                df = pd.read_csv(request.file_path, **request.options)
            elif file_type == "json":
                df = pd.read_json(request.file_path, **request.options)
            elif file_type == "excel":
                df = pd.read_excel(request.file_path, **request.options)
            else:
                return FileUploadResponse(
                    success=False,
                    message=f"Unsupported file type '{file_type}'.",
                )
            
            # Generate a unique ID for the data
            data_id = f"data_{len(self._data_store) + 1}"
            self._data_store[data_id] = df
            
            # Copy the file to the temporary directory
            file_name = os.path.basename(request.file_path)
            temp_file_path = os.path.join(self._temp_dir, file_name)
            with open(request.file_path, "rb") as src_file, open(temp_file_path, "wb") as dst_file:
                dst_file.write(src_file.read())
            
            # Gather file information
            file_info = {
                "original_path": request.file_path,
                "temp_path": temp_file_path,
                "file_type": file_type,
                "file_size_mb": file_size_mb,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            }
            
            return FileUploadResponse(
                success=True,
                message=f"Successfully uploaded and loaded file '{request.file_path}'.",
                data_id=data_id,
                file_info=file_info,
            )
        except Exception as e:
            logger.exception("Error uploading file")
            return FileUploadResponse(
                success=False,
                message=f"Error uploading file: {str(e)}",
            )
    
    async def load_google_sheet(self, request: GoogleSheetRequest) -> GoogleSheetResponse:
        """Load data from a Google Sheet.
        
        Args:
            request: The request containing the Google Sheet URL and options.
            
        Returns:
            A response indicating whether the sheet was loaded successfully.
        """
        try:
            # Extract the sheet ID from the URL
            url_parts = urllib.parse.urlparse(request.sheet_url)
            if "docs.google.com/spreadsheets" not in url_parts.netloc + url_parts.path:
                return GoogleSheetResponse(
                    success=False,
                    message=f"Invalid Google Sheet URL: '{request.sheet_url}'.",
                )
            
            # Extract the sheet ID from the URL
            path_parts = url_parts.path.split("/")
            sheet_id = None
            for i, part in enumerate(path_parts):
                if part == "d" and i + 1 < len(path_parts):
                    sheet_id = path_parts[i + 1]
                    break
            
            if not sheet_id:
                return GoogleSheetResponse(
                    success=False,
                    message=f"Could not extract sheet ID from URL: '{request.sheet_url}'.",
                )
            
            # Construct the export URL
            export_format = "csv"
            export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format={export_format}"
            
            # Add sheet name if provided
            if request.sheet_name:
                export_url += f"&gid={request.sheet_name}"
            
            # Download the sheet
            with urllib.request.urlopen(export_url) as response:
                sheet_data = response.read()
            
            # Load the data into a DataFrame
            df = pd.read_csv(io.BytesIO(sheet_data), **request.options)
            
            # Generate a unique ID for the data
            data_id = f"data_{len(self._data_store) + 1}"
            self._data_store[data_id] = df
            
            # Gather sheet information
            sheet_info = {
                "sheet_url": request.sheet_url,
                "sheet_id": sheet_id,
                "sheet_name": request.sheet_name,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            }
            
            return GoogleSheetResponse(
                success=True,
                message=f"Successfully loaded Google Sheet '{request.sheet_url}'.",
                data_id=data_id,
                sheet_info=sheet_info,
            )
        except Exception as e:
            logger.exception("Error loading Google Sheet")
            return GoogleSheetResponse(
                success=False,
                message=f"Error loading Google Sheet: {str(e)}",
            )
    
    async def connect_to_database(self, request: DatabaseConnectionRequest) -> DatabaseConnectionResponse:
        """Connect to a database and load data.
        
        Args:
            request: The request containing the database connection information.
            
        Returns:
            A response indicating whether the connection was successful.
        """
        try:
            # Check the connection type
            connection_type = request.connection_type.lower()
            
            # Connect to the database and execute the query
            if connection_type == "sqlite":
                import sqlite3
                conn = sqlite3.connect(request.connection_string)
                df = pd.read_sql_query(request.query, conn, **request.options)
                conn.close()
            
            elif connection_type == "mysql":
                try:
                    import pymysql
                    conn = pymysql.connect(request.connection_string)
                    df = pd.read_sql_query(request.query, conn, **request.options)
                    conn.close()
                except ImportError:
                    return DatabaseConnectionResponse(
                        success=False,
                        message="MySQL connection requires the pymysql package. Please install it with 'pip install pymysql'.",
                    )
            
            elif connection_type == "postgresql":
                try:
                    import psycopg2
                    conn = psycopg2.connect(request.connection_string)
                    df = pd.read_sql_query(request.query, conn, **request.options)
                    conn.close()
                except ImportError:
                    return DatabaseConnectionResponse(
                        success=False,
                        message="PostgreSQL connection requires the psycopg2 package. Please install it with 'pip install psycopg2-binary'.",
                    )
            
            else:
                return DatabaseConnectionResponse(
                    success=False,
                    message=f"Unsupported database connection type: '{connection_type}'.",
                )
            
            # Generate a unique ID for the data
            data_id = f"data_{len(self._data_store) + 1}"
            self._data_store[data_id] = df
            
            # Gather connection information (excluding sensitive information)
            connection_info = {
                "connection_type": connection_type,
                "query": request.query,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            }
            
            return DatabaseConnectionResponse(
                success=True,
                message=f"Successfully connected to {connection_type} database and executed query.",
                data_id=data_id,
                connection_info=connection_info,
            )
        except Exception as e:
            logger.exception("Error connecting to database")
            return DatabaseConnectionResponse(
                success=False,
                message=f"Error connecting to database: {str(e)}",
            )
    
    async def load_api_data(self, request: APIDataRequest) -> APIDataResponse:
        """Load data from an API.
        
        Args:
            request: The request containing the API URL and options.
            
        Returns:
            A response indicating whether the data was loaded successfully.
        """
        try:
            # Prepare the request
            method = request.method.upper()
            headers = request.headers.copy()
            
            # Add default headers if not provided
            if "User-Agent" not in headers:
                headers["User-Agent"] = "DataAnalysisAgent/1.0"
            if method in ["POST", "PUT", "PATCH"] and "Content-Type" not in headers:
                headers["Content-Type"] = "application/json"
            
            # Prepare the URL with parameters for GET requests
            url = request.api_url
            if method == "GET" and request.params:
                url_parts = list(urllib.parse.urlparse(url))
                query = dict(urllib.parse.parse_qsl(url_parts[4]))
                query.update(request.params)
                url_parts[4] = urllib.parse.urlencode(query)
                url = urllib.parse.urlunparse(url_parts)
            
            # Prepare the request
            req = urllib.request.Request(url, method=method)
            for key, value in headers.items():
                req.add_header(key, value)
            
            # Add the body for non-GET requests
            if method != "GET" and request.body:
                data = json.dumps(request.body).encode("utf-8")
                req.data = data
            
            # Send the request
            with urllib.request.urlopen(req) as response:
                response_data = response.read()
                response_code = response.getcode()
                response_headers = dict(response.info())
            
            # Parse the response based on the format
            response_format = request.response_format.lower()
            if response_format == "json":
                data = json.loads(response_data)
                df = pd.json_normalize(data)
            elif response_format == "csv":
                df = pd.read_csv(io.BytesIO(response_data))
            else:
                return APIDataResponse(
                    success=False,
                    message=f"Unsupported response format: '{response_format}'.",
                )
            
            # Generate a unique ID for the data
            data_id = f"data_{len(self._data_store) + 1}"
            self._data_store[data_id] = df
            
            # Gather API information
            api_info = {
                "api_url": request.api_url,
                "method": method,
                "response_code": response_code,
                "response_format": response_format,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            }
            
            return APIDataResponse(
                success=True,
                message=f"Successfully loaded data from API '{request.api_url}'.",
                data_id=data_id,
                api_info=api_info,
            )
        except Exception as e:
            logger.exception("Error loading API data")
            return APIDataResponse(
                success=False,
                message=f"Error loading API data: {str(e)}",
            )

