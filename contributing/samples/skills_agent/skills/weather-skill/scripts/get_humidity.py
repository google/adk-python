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

import argparse


def get_humidity(location: str) -> str:
  """Fetch live humidity for a given location. (Simulated)"""
  print(f"Fetching live humidity for {location}...")
  return "45% (Simulated)"


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--location", type=str, default="Mountain View")
  args = parser.parse_args()

  print(get_humidity(args.location))
