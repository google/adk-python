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

"""In-memory stand-in for the pymongo calls MongoDbSessionService makes."""

from __future__ import annotations

from collections.abc import Iterator
import copy
from typing import Any


class DuplicateKeyError(Exception):
  """Fake pymongo.errors.DuplicateKeyError matched by class name."""


class OperationFailure(Exception):
  """Fake pymongo.errors.OperationFailure carrying an error code."""

  def __init__(self, message: str, *, code: int) -> None:
    super().__init__(message)
    self.code = code


def _doc_key(doc_id: Any) -> Any:
  if isinstance(doc_id, dict):
    return tuple(doc_id.items())
  return doc_id


def _check_bson_field_names(value: Any) -> None:
  if isinstance(value, dict):
    for key, child in value.items():
      if "." in key or key.startswith("$"):
        raise ValueError(f"Invalid BSON field name: {key!r}")
      _check_bson_field_names(child)
  elif isinstance(value, list):
    for item in value:
      _check_bson_field_names(item)


def _matches(doc: dict[str, Any], query: dict[str, Any]) -> bool:
  for key, condition in query.items():
    value = doc.get(key)
    if isinstance(condition, dict) and any(
        op.startswith("$") for op in condition
    ):
      for op, bound in condition.items():
        if op == "$gte":
          if value is None or value < bound:
            return False
        elif op == "$in":
          if value not in bound:
            return False
        else:
          raise NotImplementedError(f"Unsupported query operator: {op}")
    elif value != condition:
      return False
  return True


class FakeCursor:

  def __init__(self, docs: list[dict[str, Any]]) -> None:
    self._docs = docs

  def sort(
      self, key_or_list: str | list[tuple[str, int]], direction: int = 1
  ) -> FakeCursor:
    keys = (
        [(key_or_list, direction)]
        if isinstance(key_or_list, str)
        else list(key_or_list)
    )
    for key, dir_ in reversed(keys):
      self._docs.sort(key=lambda doc: doc.get(key, 0), reverse=dir_ < 0)
    return self

  def limit(self, count: int) -> FakeCursor:
    self._docs = self._docs[:count]
    return self

  def __iter__(self) -> Iterator[dict[str, Any]]:
    for doc in self._docs:
      yield copy.deepcopy(doc)


class FakeCollection:

  def __init__(self) -> None:
    self._docs: dict[Any, dict[str, Any]] = {}
    self._indexes: dict[str, dict[str, Any]] = {
        "_id_": {"key": [("_id", 1)]},
    }

  def create_index(self, keys: list[tuple[str, int]], *, name: str) -> str:
    keys = list(keys)
    existing = self._indexes.get(name)
    if existing is not None and existing["key"] != keys:
      raise OperationFailure(
          f"An existing index has the same name as the requested index: {name}",
          code=86,
      )
    for other_name, spec in self._indexes.items():
      if other_name != name and spec["key"] == keys:
        raise OperationFailure(
            f"Index already exists with a different name: {other_name}",
            code=85,
        )
    self._indexes[name] = {"key": keys}
    return name

  def index_information(self) -> dict[str, dict[str, Any]]:
    return copy.deepcopy(self._indexes)

  def find_one(
      self, query: dict[str, Any], *, session: Any = None
  ) -> dict[str, Any] | None:
    del session
    for doc in self._docs.values():
      if _matches(doc, query):
        return copy.deepcopy(doc)
    return None

  def find(self, query: dict[str, Any], *, session: Any = None) -> FakeCursor:
    del session
    return FakeCursor(
        [copy.deepcopy(d) for d in self._docs.values() if _matches(d, query)]
    )

  def insert_one(self, doc: dict[str, Any], *, session: Any = None) -> None:
    del session
    _check_bson_field_names(doc)
    key = _doc_key(doc["_id"])
    if key in self._docs:
      raise DuplicateKeyError("E11000 duplicate key error", 11000)
    self._docs[key] = copy.deepcopy(doc)

  def update_one(
      self,
      query: dict[str, Any],
      update: dict[str, Any],
      *,
      upsert: bool = False,
      session: Any = None,
  ) -> None:
    del session
    _check_bson_field_names(update.get("$set", {}))
    target = None
    for doc in self._docs.values():
      if _matches(doc, query):
        target = doc
        break
    if target is None:
      if not upsert:
        return
      doc_id = query["_id"]
      target = {"_id": copy.deepcopy(doc_id)}
      self._docs[_doc_key(doc_id)] = target
    for key, value in update.get("$set", {}).items():
      target[key] = copy.deepcopy(value)
    for key, delta in update.get("$inc", {}).items():
      target[key] = target.get(key, 0) + delta

  def replace_one(
      self,
      query: dict[str, Any],
      replacement: dict[str, Any],
      *,
      upsert: bool = False,
      session: Any = None,
  ) -> None:
    del session
    _check_bson_field_names(replacement)
    doc_id = query["_id"]
    key = _doc_key(doc_id)
    if key in self._docs or upsert:
      new_doc = copy.deepcopy(replacement)
      new_doc["_id"] = copy.deepcopy(doc_id)
      self._docs[key] = new_doc

  def find_one_and_update(
      self,
      query: dict[str, Any],
      update: dict[str, Any],
      *,
      return_document: bool = False,
      session: Any = None,
  ) -> dict[str, Any] | None:
    del return_document, session
    _check_bson_field_names(update.get("$set", {}))
    for doc in self._docs.values():
      if _matches(doc, query):
        for key, value in update.get("$set", {}).items():
          doc[key] = copy.deepcopy(value)
        for key, delta in update.get("$inc", {}).items():
          doc[key] = doc.get(key, 0) + delta
        return copy.deepcopy(doc)
    return None

  def delete_one(self, query: dict[str, Any], *, session: Any = None) -> None:
    del session
    for doc_id, doc in list(self._docs.items()):
      if _matches(doc, query):
        del self._docs[doc_id]
        return

  def delete_many(self, query: dict[str, Any], *, session: Any = None) -> None:
    del session
    for doc_id, doc in list(self._docs.items()):
      if _matches(doc, query):
        del self._docs[doc_id]

  def count_documents(
      self, query: dict[str, Any], *, session: Any = None
  ) -> int:
    del session
    return sum(1 for doc in self._docs.values() if _matches(doc, query))


class FakeDatabase:

  def __init__(self) -> None:
    self._collections: dict[str, FakeCollection] = {}

  def __getitem__(self, name: str) -> FakeCollection:
    if name not in self._collections:
      self._collections[name] = FakeCollection()
    return self._collections[name]


class FakeMongoClient:
  """In-memory MongoClient without session or transaction support."""

  def __init__(self) -> None:
    self._databases: dict[str, FakeDatabase] = {}
    self.closed = False

  def __getitem__(self, name: str) -> FakeDatabase:
    if name not in self._databases:
      self._databases[name] = FakeDatabase()
    return self._databases[name]

  def start_session(self) -> Any:
    raise NotImplementedError("Sessions are not supported by FakeMongoClient")

  def close(self) -> None:
    self.closed = True
