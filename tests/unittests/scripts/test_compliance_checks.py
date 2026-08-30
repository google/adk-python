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

import pathlib

from scripts import compliance_checks

# A filename that is not in the exclusion list, so check_mtls runs the real
# check instead of short-circuiting on the exclusion.
_UNEXCLUDED_NAME = 'unexcluded.py'

_REPO_ROOT = pathlib.Path(compliance_checks.__file__).resolve().parents[1]


def test_check_mtls_ignores_oauth_scope() -> None:
  content = 'scope = "https://www.googleapis.com/auth/cloud-platform"\n'
  assert compliance_checks.check_mtls(content, 'test_file.py') is True


def test_check_mtls_detects_missing_mtls() -> None:
  content = 'endpoint = "https://storage.googleapis.com"\n'
  assert compliance_checks.check_mtls(content, 'test_file.py') is False


def test_check_mtls_passes_with_mtls() -> None:
  content = (
      'endpoint = "https://storage.googleapis.com"\n'
      'mtls_endpoint = "https://storage.mtls.googleapis.com"\n'
  )
  assert compliance_checks.check_mtls(content, 'test_file.py') is True


def test_mtls_exclusions_are_all_still_needed() -> None:
  assert _UNEXCLUDED_NAME not in compliance_checks._EXCLUDED_FROM_MTLS
  redundant: list[str] = []
  for path in sorted(compliance_checks._EXCLUDED_FROM_MTLS):
    source = _REPO_ROOT / path
    if not source.is_file():
      continue
    content = source.read_text(encoding='utf-8')
    if compliance_checks.check_mtls(content, _UNEXCLUDED_NAME):
      redundant.append(path)
  assert not redundant, (
      'These files pass the mTLS check on their own; drop them from'
      f' _EXCLUDED_FROM_MTLS: {redundant}'
  )


# Assembled rather than written out, so that this file does not trip the very
# check it is testing.
_INTERNAL_LINK = 'go' + '/some-design-doc'


def test_check_internal_links_detects_a_shortlink() -> None:
  content = f'# lives in the experimental namespace of {_INTERNAL_LINK}\n'
  assert not compliance_checks.check_internal_links(content)


def test_check_internal_links_allows_a_public_url_with_a_go_path() -> None:
  content = 'url = "https://example.com/go/somewhere"\n'
  assert compliance_checks.check_internal_links(content)


def test_check_internal_links_allows_a_go_file_name() -> None:
  content = 'path = "internal/registry.go/../main.go"\n'
  assert compliance_checks.check_internal_links(content)


def test_no_shipped_source_file_has_an_internal_link() -> None:
  offenders = [
      str(path.relative_to(_REPO_ROOT))
      for path in sorted((_REPO_ROOT / 'src').rglob('*.py'))
      if not compliance_checks.check_internal_links(
          path.read_text(encoding='utf-8')
      )
  ]
  assert not offenders, (
      'These files ship an internal shortlink that no reader outside Google'
      f' can resolve: {offenders}'
  )


def test_route_decorator_order_flags_a_guard_above_the_route() -> None:
  # The shape that left /builder/save unguarded: the route registers the raw
  # handler, so working_in_progress never binds.
  content = (
      '@working_in_progress(block_usage=True)\n'
      '@app.post("/builder/save")\n'
      'async def save(): ...\n'
  )
  assert compliance_checks.check_route_decorator_order(content) == [
      (1, 'working_in_progress')
  ]


def test_route_decorator_order_accepts_a_guard_below_the_route() -> None:
  content = (
      '@app.post("/builder/save")\n'
      '@working_in_progress(block_usage=True)\n'
      'async def save(): ...\n'
  )
  assert compliance_checks.check_route_decorator_order(content) == []


def test_route_decorator_order_flags_every_decorator_above_the_route() -> None:
  content = (
      '@experimental\n'
      '@deprecated("gone soon")\n'
      '@router.websocket("/live")\n'
      'async def live(): ...\n'
  )
  assert compliance_checks.check_route_decorator_order(content) == [
      (1, 'experimental'),
      (2, 'deprecated'),
  ]


def test_route_decorator_order_allows_a_second_route_below_the_first() -> None:
  # Registering one handler under two paths is fine; both decorators run.
  content = (
      '@app.get("/eval-sets")\n'
      '@app.get("/eval_sets")\n'
      'async def list_eval_sets(): ...\n'
  )
  assert compliance_checks.check_route_decorator_order(content) == []


def test_route_decorator_order_ignores_calls_that_take_no_path() -> None:
  # @cache.get('key') is not a route, so nothing here is out of order.
  content = '@retry\n@cache.get("some-key")\ndef load(): ...\n'
  assert compliance_checks.check_route_decorator_order(content) == []


def test_route_decorator_order_ignores_non_routing_methods() -> None:
  # A path-shaped argument is not enough; the method has to register a route.
  content = (
      '@deprecated("gone soon")\n@app.mount("/static")\ndef assets(): ...\n'
  )
  assert compliance_checks.check_route_decorator_order(content) == []


def test_route_decorator_order_ignores_unparsable_content() -> None:
  assert compliance_checks.check_route_decorator_order('def (:\n') == []
