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

"""Tests for gcp_utils."""

import unittest
from unittest import mock

from google.adk.cli.utils import gcp_utils
import google.auth
import google.auth.exceptions
import requests


class TestGcpUtils(unittest.TestCase):

  @mock.patch("google.auth.default")
  def test_check_adc_success(self, mock_auth_default):
    mock_auth_default.return_value = (mock.Mock(), "test-project")
    self.assertTrue(gcp_utils.check_adc())

  @mock.patch("google.auth.default")
  def test_check_adc_failure(self, mock_auth_default):
    mock_auth_default.side_effect = (
        google.auth.exceptions.DefaultCredentialsError()
    )
    self.assertFalse(gcp_utils.check_adc())

  @mock.patch("google.auth.default")
  def test_get_access_token(self, mock_auth_default):
    mock_creds = mock.Mock()
    mock_creds.token = "test-token"
    mock_creds.valid = True
    mock_auth_default.return_value = (mock_creds, "test-project")
    self.assertEqual(gcp_utils.get_access_token(), "test-token")

  @mock.patch("google.adk.cli.utils.gcp_utils.AuthorizedSession")
  @mock.patch("google.auth.default")
  def test_retrieve_express_project_success(
      self, mock_auth_default, mock_session_cls
  ):
    mock_auth_default.return_value = (mock.Mock(), "test-project-id")

    mock_session = mock.Mock()
    mock_session_cls.return_value = mock_session
    mock_response = mock.Mock()
    mock_response.json.return_value = {
        "expressProject": {
            "projectId": "test-project",
            "defaultApiKey": "test-api-key",
            "region": "us-central1",
        }
    }
    mock_session.get.return_value = mock_response

    result = gcp_utils.retrieve_express_project()
    self.assertEqual(result["project_id"], "test-project")
    self.assertEqual(result["api_key"], "test-api-key")
    self.assertEqual(result["region"], "us-central1")
    mock_session.get.assert_called_once()
    args, kwargs = mock_session.get.call_args
    self.assertEqual(
        args[0],
        "https://us-central1-aiplatform.googleapis.com/v1beta1/vertexExpress:retrieveExpressProject",
    )
    self.assertEqual(kwargs["params"], {"get_default_api_key": True})

  @mock.patch("google.adk.cli.utils.gcp_utils.AuthorizedSession")
  @mock.patch("google.auth.default")
  def test_retrieve_express_project_not_found(
      self, mock_auth_default, mock_session_cls
  ):
    mock_auth_default.return_value = (mock.Mock(), "test-project-id")

    mock_session = mock.Mock()
    mock_session_cls.return_value = mock_session
    mock_response = mock.Mock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        response=mock_response
    )
    mock_session.get.return_value = mock_response

    result = gcp_utils.retrieve_express_project()
    self.assertIsNone(result)

  @mock.patch("google.adk.cli.utils.gcp_utils.AuthorizedSession")
  @mock.patch("google.auth.default")
  def test_check_express_eligibility(self, mock_auth_default, mock_session_cls):
    mock_auth_default.return_value = (mock.Mock(), "test-project-id")

    mock_session = mock.Mock()
    mock_session_cls.return_value = mock_session
    mock_response = mock.Mock()
    mock_response.json.return_value = {"eligibility": "IN_SCOPE"}
    mock_session.get.return_value = mock_response

    self.assertTrue(gcp_utils.check_express_eligibility())

  @mock.patch("google.adk.cli.utils.gcp_utils.AuthorizedSession")
  @mock.patch("google.auth.default")
  def test_sign_up_express(self, mock_auth_default, mock_session_cls):
    mock_auth_default.return_value = (mock.Mock(), "test-project-id")

    mock_session = mock.Mock()
    mock_session_cls.return_value = mock_session
    mock_response = mock.Mock()
    mock_response.json.return_value = {
        "projectId": "new-project",
        "defaultApiKey": "new-api-key",
        "region": "us-central1",
    }
    mock_session.post.return_value = mock_response

    result = gcp_utils.sign_up_express()
    self.assertEqual(result["project_id"], "new-project")
    self.assertEqual(result["api_key"], "new-api-key")
    args, _ = mock_session.post.call_args
    self.assertEqual(
        args[0],
        "https://us-central1-aiplatform.googleapis.com/v1beta1/vertexExpress:signUp",
    )

  @mock.patch(
      "google.adk.cli.utils.gcp_utils.resourcemanager_v3.ProjectsClient"
  )
  def test_list_gcp_projects(self, mock_client_cls):
    mock_client = mock.Mock()
    mock_client_cls.return_value = mock_client

    mock_project1 = mock.Mock()
    mock_project1.project_id = "p1"
    mock_project1.display_name = "Project 1"

    mock_project2 = mock.Mock()
    mock_project2.project_id = "p2"
    mock_project2.display_name = None

    mock_client.search_projects.return_value = [mock_project1, mock_project2]

    projects = gcp_utils.list_gcp_projects()
    self.assertEqual(len(projects), 2)
    self.assertEqual(projects[0], ("p1", "Project 1"))
    self.assertEqual(projects[1], ("p2", "p2"))


if __name__ == "__main__":
  unittest.main()
