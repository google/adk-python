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

from functools import wraps


def tool_error_notifier(target_exception):
    """
    A decorator that wraps a tool function and catches a specific exception.
    If the specified exception occurs, it returns True, signaling an error
    back to the LLM agent.

    Args:
        target_exception (Type[Exception]): The exception class to catch.

    Returns:
        Callable: A decorated function that returns True if the specified
        exception is raised, otherwise returns the function result.
    """
    def _decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except target_exception as e:
                return {
                    {
                        "status": "error",
                        "error": e.__class__.__name__,
                        "message": str(e)
                    }
                }
        return _wrapper
    return _decorator
