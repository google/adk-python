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

parser = argparse.ArgumentParser()
parser.add_argument("--op", choices=["add", "sub", "mul"], required=True)
parser.add_argument("--a", type=float, required=True)
parser.add_argument("--b", type=float, required=True)
args = parser.parse_args()

if args.op == "add":
  print(f"Result: {args.a + args.b}")
elif args.op == "sub":
  print(f"Result: {args.a - args.b}")
elif args.op == "mul":
  print(f"Result: {args.a * args.b}")
