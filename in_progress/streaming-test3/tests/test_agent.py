import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import sys
import os
import time # for time.time() in agent if __name__ == '__main__'

# --- Global Mocking for vertexai (BEFORE app.agent is imported) ---
os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'True'
os.environ['GOOGLE_CLOUD_PROJECT'] = 'test-project'
os.environ['GOOGLE_CLOUD_LOCATION'] = 'test-location'

mock_vertexai_module = MagicMock(name="vertexai_mock")
mock_generative_models_module = MagicMock(name="generative_models_mock")
mock_generative_model_class = MagicMock(name="GenerativeModelClassMock")
mock_model_instance = MagicMock(name="GenerativeModelInstanceMock")
mock_chat_instance = MagicMock(name="ChatInstanceMock")

# --- Part Mocking ---
mock_part_class = MagicMock(name="PartMock")
# Configure from_text and from_data to return mocks that have a to_dict method
mock_text_part_instance = MagicMock(name="TextPartInstance")
mock_text_part_instance.to_dict.return_value = {"text": "mocked text"} # Example to_dict content
mock_data_part_instance = MagicMock(name="DataPartInstance")
mock_data_part_instance.to_dict.return_value = {"inline_data": {"mime_type": "image/jpeg", "data": "mocked_data"}}

mock_part_class.from_text.return_value = mock_text_part_instance
mock_part_class.from_data.return_value = mock_data_part_instance
# --- End Part Mocking ---

mock_vertexai_module.init = MagicMock(name="vertexai_init_mock")
mock_vertexai_module.generative_models = mock_generative_models_module
mock_generative_models_module.GenerativeModel = mock_generative_model_class
mock_generative_model_class.return_value = mock_model_instance
mock_generative_models_module.Part = mock_part_class
mock_model_instance.start_chat.return_value = mock_chat_instance

sys.modules['vertexai'] = mock_vertexai_module
sys.modules['vertexai.generative_models'] = mock_generative_models_module
# --- End Global Mocking ---

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.agent import MultimodalAgent

@patch('app.agent.VideoMonitor')
@patch('app.agent.record_audio')
@patch('app.agent.transcribe_audio_bytes')
@patch('app.agent.synthesize_text_to_audio_bytes')
@patch('app.agent.play_audio_bytes')
@patch('app.agent.load_dotenv')
class TestMultimodalAgent(unittest.TestCase):

    def setUp(self):
        mock_vertexai_module.init.reset_mock()
        mock_generative_model_class.reset_mock()
        mock_generative_model_class.return_value = mock_model_instance
        mock_model_instance.start_chat.reset_mock()
        mock_model_instance.start_chat.return_value = mock_chat_instance
        mock_chat_instance.reset_mock()
        mock_chat_instance.send_message.reset_mock()

        mock_part_class.from_data.reset_mock()
        mock_part_class.from_text.reset_mock()
        # Ensure return values are reassigned for each test if they could be altered
        mock_part_class.from_text.return_value = mock_text_part_instance
        mock_part_class.from_data.return_value = mock_data_part_instance
        # Reset to_dict mocks if necessary (though typically defined once)
        mock_text_part_instance.to_dict.reset_mock(return_value=MagicMock()) # Avoid state leakage
        mock_text_part_instance.to_dict.return_value = {"text": "mocked text"}
        mock_data_part_instance.to_dict.reset_mock(return_value=MagicMock())
        mock_data_part_instance.to_dict.return_value = {"inline_data": {"mime_type": "image/jpeg", "data": "mocked_data"}}


    def test_agent_initialization(self, MockLoadDotenv, MockPlayAudio, MockSynth, MockTranscribe, MockRecord, MockVideoMonitor):
        mock_video_monitor_instance = MockVideoMonitor.return_value
        agent = MultimodalAgent(project_id="test-proj", location="test-loc", model_name="gemini-test")
        mock_vertexai_module.init.assert_called_once_with(project="test-proj", location="test-loc")
        mock_generative_model_class.assert_called_once_with("gemini-test")
        MockVideoMonitor.assert_called_once_with(camera_index=0, fps_limit=1)
        self.assertIsNotNone(agent.model)

    def test_start_chat(self, MockLoadDotenv, MockPlayAudio, MockSynth, MockTranscribe, MockRecord, MockVideoMonitor):
        mock_video_monitor_instance = MockVideoMonitor.return_value
        mock_video_monitor_instance.cap = None
        mock_video_monitor_instance.start_capture.return_value = True
        agent = MultimodalAgent(project_id="p", location="l", model_name="m")
        agent.start_chat()
        mock_model_instance.start_chat.assert_called_once_with(response_validation=False)
        self.assertEqual(agent.chat, mock_chat_instance)
        mock_video_monitor_instance.start_capture.assert_called_once()

    def test_stop_chat(self, MockLoadDotenv, MockPlayAudio, MockSynth, MockTranscribe, MockRecord, MockVideoMonitor):
        mock_video_monitor_instance = MockVideoMonitor.return_value
        mock_video_monitor_instance.cap = True
        agent = MultimodalAgent(project_id="p", location="l", model_name="m")
        agent.chat = mock_chat_instance
        agent.stop_chat()
        mock_video_monitor_instance.stop_capture.assert_called_once()
        self.assertIsNone(agent.chat)

    def test_send_text_message_success(self, MockLoadDotenv, MockPlayAudio, MockSynth, MockTranscribe, MockRecord, MockVideoMonitor):
        mock_chat_instance.send_message.return_value = MagicMock(text="Model reply")
        agent = MultimodalAgent(project_id="p", location="l", model_name="m")
        agent.chat = mock_chat_instance
        reply = agent.send_text_message("Hello agent")
        self.assertEqual(reply, "Model reply")
        mock_chat_instance.send_message.assert_called_with("Hello agent")

    def test_send_message_with_image(self, MockLoadDotenv, MockPlayAudio, MockSynth, MockTranscribe, MockRecord, MockVideoMonitor):
        mock_chat_instance.send_message.return_value = MagicMock(text="Understood image")
        agent = MultimodalAgent(project_id="p", location="l", model_name="m")
        agent.chat = mock_chat_instance
        # Pass the actual mock instance for image_parts
        reply = agent.send_message(text_prompt="Describe this", image_parts=[mock_data_part_instance])
        self.assertEqual(reply, "Understood image")
        sent_parts_list = mock_chat_instance.send_message.call_args[0][0]
        self.assertIn(mock_text_part_instance, sent_parts_list)
        self.assertIn(mock_data_part_instance, sent_parts_list)

    def test_check_for_video_changes_and_comment_detected(self, MockLoadDotenv, MockPlayAudio, MockSynth, MockTranscribe, MockRecord, MockVideoMonitor):
        mock_video_monitor_instance = MockVideoMonitor.return_value
        mock_video_monitor_instance.cap = MagicMock()
        mock_video_monitor_instance.cap.isOpened.return_value = True
        mock_video_monitor_instance.process_frame_for_changes.return_value = (True, "Motion detected", b"jpeg_bytes")
        mock_chat_instance.send_message.return_value = MagicMock(text="I see motion!")
        mock_event_queue = MagicMock()
        agent = MultimodalAgent(project_id="p", location="l", model_name="m")
        agent.chat = mock_chat_instance
        agent.check_for_video_changes_and_comment(event_queue=mock_event_queue)
        mock_video_monitor_instance.process_frame_for_changes.assert_called_once()
        mock_part_class.from_data.assert_called_with(b"jpeg_bytes", mime_type="image/jpeg")
        sent_parts_list = mock_chat_instance.send_message.call_args[0][0]
        self.assertIn(mock_text_part_instance, sent_parts_list)
        self.assertIn(mock_data_part_instance, sent_parts_list)
        mock_event_queue.put.assert_called_once_with(
            {"type": "video_change_comment", "comment": "I see motion!"}
        )

    def test_handle_voice_interaction(self, MockLoadDotenv, MockPlayAudio, MockSynth, MockTranscribe, MockRecord, MockVideoMonitor):
        MockRecord.return_value = b"recorded_audio_bytes"
        MockTranscribe.return_value = "user speech text"
        mock_chat_instance.send_message.return_value = MagicMock(text="model speech text")
        MockSynth.return_value = b"synthesized_audio_bytes"
        mock_event_queue = MagicMock()
        agent = MultimodalAgent(project_id="p", location="l", model_name="m")
        agent.chat = mock_chat_instance
        agent.handle_voice_interaction(record_duration_seconds=3, event_queue=mock_event_queue)
        MockRecord.assert_called_once_with(duration_seconds=3)
        MockTranscribe.assert_called_once_with(b"recorded_audio_bytes")
        mock_chat_instance.send_message.assert_called_with("user speech text")
        MockSynth.assert_called_once_with("model speech text")
        MockPlayAudio.assert_called_once_with(b"synthesized_audio_bytes", output_format="mp3")
        self.assertIn(unittest.mock.call({"type": "transcription", "text": "user speech text"}), mock_event_queue.put.call_args_list)
        self.assertIn(unittest.mock.call({"type": "model_response_audio_text", "text": "model speech text"}), mock_event_queue.put.call_args_list)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
