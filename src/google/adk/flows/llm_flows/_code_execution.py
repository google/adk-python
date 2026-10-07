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

"""Backward compatibility module for code execution.

Code execution processing has moved to
``google.adk.flows.llm_flows.extensions._code_execution``. This module re-exports
all symbols for backward compatibility.
"""

from __future__ import annotations

from .extensions._code_execution import _CodeExecutionRequestProcessor as _CodeExecutionRequestProcessor
from .extensions._code_execution import _CodeExecutionResponseProcessor as _CodeExecutionResponseProcessor
from .extensions._code_execution import _DATA_FILE_HELPER_LIB as _DATA_FILE_HELPER_LIB
from .extensions._code_execution import _DATA_FILE_UTIL_MAP as _DATA_FILE_UTIL_MAP
from .extensions._code_execution import _extract_and_replace_inline_files as _extract_and_replace_inline_files
from .extensions._code_execution import _get_data_file_preprocessing_code as _get_data_file_preprocessing_code
from .extensions._code_execution import _get_or_set_execution_id as _get_or_set_execution_id
from .extensions._code_execution import _NON_BUILTIN_EXECUTOR_INSTRUCTION as _NON_BUILTIN_EXECUTOR_INSTRUCTION
from .extensions._code_execution import _post_process_code_execution_result as _post_process_code_execution_result
from .extensions._code_execution import _run_post_processor as _run_post_processor
from .extensions._code_execution import _run_pre_processor as _run_pre_processor
from .extensions._code_execution import DataFileUtil as DataFileUtil
from .extensions._code_execution import get_content_as_bytes as get_content_as_bytes
from .extensions._code_execution import logger as logger
from .extensions._code_execution import request_processor as request_processor
from .extensions._code_execution import response_processor as response_processor

__all__ = [
    'DataFileUtil',
    '_CodeExecutionRequestProcessor',
    '_CodeExecutionResponseProcessor',
    '_DATA_FILE_HELPER_LIB',
    '_DATA_FILE_UTIL_MAP',
    '_NON_BUILTIN_EXECUTOR_INSTRUCTION',
    '_extract_and_replace_inline_files',
    '_get_data_file_preprocessing_code',
    '_get_or_set_execution_id',
    '_post_process_code_execution_result',
    '_run_post_processor',
    '_run_pre_processor',
    'get_content_as_bytes',
    'logger',
    'request_processor',
    'response_processor',
]
