#!/bin/bash
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

# Autoformat ADK codebase.

if ! command -v isort &> /dev/null
then
    echo "isort not found, refer to CONTRIBUTING.md to set up dev environment first."
    exit
fi

if ! command -v pyink &> /dev/null
then
    echo "pyink not found, refer to CONTRIBUTING.md to set up dev environment first."
    exit
fi

echo '---------------------------------------'
echo '|  Organizing imports for src/...'
echo '---------------------------------------'

isort src/
echo 'All done! ✨ 🍰 ✨'

echo '---------------------------------------'
echo '|  Organizing imports for tests/...'
echo '---------------------------------------'

isort tests/
echo 'All done! ✨ 🍰 ✨'

echo '---------------------------------------'
echo '|  Organizing imports for contributing/...'
echo '---------------------------------------'

isort contributing/
echo 'All done! ✨ 🍰 ✨'

echo '---------------------------------------'
echo '|  Auto-formatting src/...'
echo '---------------------------------------'

find -L src/ -type f -name "*.py" -exec pyink --config pyproject.toml {} +

echo '---------------------------------------'
echo '|  Auto-formatting tests/...'
echo '---------------------------------------'

find -L tests/ -type f -name "*.py" -exec pyink --config pyproject.toml {} +

echo '---------------------------------------'
echo '|  Auto-formatting contributing/...'
echo '---------------------------------------'

find -L contributing/ -type f -name "*.py" -exec pyink --config pyproject.toml {} +
