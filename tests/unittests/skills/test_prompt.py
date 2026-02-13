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

"""Unit tests for prompt."""

from google.adk.skills import models
from google.adk.skills import prompt
import pytest


class TestPrompt:

  def test_format_skills_as_xml(self):
    skills = [
        models.Frontmatter(name="skill1", description="desc1"),
        models.Frontmatter(name="skill2", description="desc2"),
    ]
    xml = prompt.format_skills_as_xml(skills)

    assert "<name>\nskill1\n</name>" in xml
    assert "<description>\ndesc1\n</description>" in xml
    assert "<location>" not in xml
    assert "<name>\nskill2\n</name>" in xml
    assert "<description>\ndesc2\n</description>" in xml
    assert xml.startswith("<available_skills>")
    assert xml.endswith("</available_skills>")

  def test_format_skills_as_xml_empty(self):
    xml = prompt.format_skills_as_xml([])
    assert xml == "<available_skills>\n</available_skills>"

  def test_format_skills_as_xml_escaping(self):
    skills = [
        models.Frontmatter(name="skill&name", description="desc<ription>"),
    ]
    xml = prompt.format_skills_as_xml(skills)
    assert "skill&amp;name" in xml
    assert "desc&lt;ription&gt;" in xml
