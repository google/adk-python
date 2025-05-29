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

import abc

from .example import Example


# A class that provides examples for a given query.
class BaseExampleProvider(abc.ABC):
  """Base class for example providers.

  This class defines the interface for providing examples for a given query.
  """

  @abc.abstractmethod
  def get_examples(self, query: str) -> list[Example]:
    """Returns a list of examples for a given query.

    Args:
        query: The query to get examples for.

    Returns:
        A list of Example objects.
    """
