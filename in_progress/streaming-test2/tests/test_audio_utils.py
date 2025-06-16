import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from scipy.io.wavfile import write as write_wav
import io
import sys
import os

# Mock sounddevice is done globally in run_tests.py.
# We get a reference to that global mock here.
# This ensures that app.audio_utils, when it 'import sounddevice as sd', gets the global mock.
mock_sd_module = sys.modules.get('sounddevice', MagicMock(name="FallbackMockSoundDeviceModule"))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.audio_utils import record_audio, transcribe_audio_bytes, synthesize_text_to_audio_bytes, play_audio_bytes
from app.config import AUDIO_SAMPLE_RATE

class TestAudioUtils(unittest.TestCase):

    def setUp(self):
        # Reset the attributes of the global mock_sd_module for each test
        # This is important because mock_sd_module is shared state modified by run_tests.py
        mock_sd_module.rec = MagicMock(name="sd_rec_mock_from_setUp")
        mock_sd_module.wait = MagicMock(name="sd_wait_mock_from_setUp")
        mock_sd_module.play = MagicMock(name="sd_play_mock_from_setUp")
        # Ensure other methods that might be called are also reset if necessary
        mock_sd_module.reset_mock() # Resets call counts etc. on the main mock
                                    # and also replaces attributes like .rec with new MagicMocks.
                                    # So, re-assign them if specific naming is desired or if they need to be shared.
        # Re-establish well-known mocks on the global mock after reset_mock()
        # These will be the mocks that app.audio_utils.sd.* actually calls.
        # The 'global' keyword here was causing a SyntaxError and is not needed
        # as mock_sd_module is already in the module's global scope.
        mock_sd_module.rec = MagicMock(name="sd.rec")
        mock_sd_module.wait = MagicMock(name="sd.wait")
        mock_sd_module.play = MagicMock(name="sd.play")


    def generate_dummy_wav_bytes(self, duration_seconds=1, sample_rate=AUDIO_SAMPLE_RATE, channels=1):
        num_samples = int(duration_seconds * sample_rate)
        frequency = 440
        t = np.linspace(0, duration_seconds, num_samples, False)
        audio_data = (0.5 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
        if channels == 2:
            audio_data = np.column_stack((audio_data, audio_data))
        byte_io = io.BytesIO()
        write_wav(byte_io, sample_rate, audio_data)
        return byte_io.getvalue()

    def test_record_audio_success(self):
        duration = 1
        sample_rate = 16000
        channels = 1
        expected_samples = duration * sample_rate
        # This is a REAL numpy array that scipy.io.wavfile.write expects.
        dummy_numpy_array = np.random.randint(-32768, 32767, size=(expected_samples, channels), dtype=np.int16)

        # Configure the mock that app.audio_utils.sd.rec will resolve to
        mock_sd_module.rec.return_value = dummy_numpy_array

        audio_bytes = record_audio(duration_seconds=duration, sample_rate=sample_rate, channels=channels)

        mock_sd_module.rec.assert_called_once_with(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='int16')
        mock_sd_module.wait.assert_called_once()
        self.assertIsNotNone(audio_bytes)
        self.assertIsInstance(audio_bytes, bytes)

    def test_record_audio_failure(self):
        # Configure the mock that app.audio_utils.sd.rec will resolve to
        mock_sd_module.rec.side_effect = Exception("Recording failed")

        audio_bytes = record_audio(duration_seconds=1)

        self.assertIsNone(audio_bytes)
        mock_sd_module.rec.assert_called_once() # Should still be called once


    @patch('app.audio_utils.speech.SpeechClient')
    def test_transcribe_audio_bytes_success(self, MockSpeechClient):
        mock_client_instance = MockSpeechClient.return_value
        mock_recognize_response = MagicMock()
        mock_alternative = MagicMock()
        mock_alternative.transcript = "hello world"
        mock_result = MagicMock()
        mock_result.alternatives = [mock_alternative]
        mock_recognize_response.results = [mock_result]
        mock_client_instance.recognize.return_value = mock_recognize_response

        dummy_audio = self.generate_dummy_wav_bytes()
        transcript = transcribe_audio_bytes(dummy_audio, sample_rate=AUDIO_SAMPLE_RATE)
        self.assertEqual(transcript, "hello world")

    @patch('app.audio_utils.speech.SpeechClient', side_effect=Exception("GCS API Error"))
    def test_transcribe_audio_bytes_api_failure(self, MockSpeechClient):
        dummy_audio = self.generate_dummy_wav_bytes()
        transcript = transcribe_audio_bytes(dummy_audio)
        self.assertIsNone(transcript)

    def test_transcribe_audio_bytes_no_input(self):
        transcript = transcribe_audio_bytes(None)
        self.assertIsNone(transcript)

    @patch('app.audio_utils.speech.SpeechClient')
    def test_transcribe_audio_bytes_no_results(self, MockSpeechClient):
        mock_client_instance = MockSpeechClient.return_value
        mock_recognize_response = MagicMock()
        mock_recognize_response.results = []
        mock_client_instance.recognize.return_value = mock_recognize_response
        dummy_audio = self.generate_dummy_wav_bytes()
        transcript = transcribe_audio_bytes(dummy_audio, sample_rate=AUDIO_SAMPLE_RATE)
        self.assertEqual(transcript, "")

    @patch('app.audio_utils.texttospeech.TextToSpeechClient')
    def test_synthesize_text_to_audio_bytes_success(self, MockTextToSpeechClient):
        mock_client_instance = MockTextToSpeechClient.return_value
        mock_synthesize_response = MagicMock()
        mock_synthesize_response.audio_content = b"dummy_audio_content"
        mock_client_instance.synthesize_speech.return_value = mock_synthesize_response
        audio_bytes = synthesize_text_to_audio_bytes("hello")
        self.assertEqual(audio_bytes, b"dummy_audio_content")

    @patch('app.audio_utils.texttospeech.TextToSpeechClient', side_effect=Exception("TTS API Error"))
    def test_synthesize_text_to_audio_bytes_api_failure(self, MockTextToSpeechClient):
        audio_bytes = synthesize_text_to_audio_bytes("test text")
        self.assertIsNone(audio_bytes)

    def test_synthesize_text_to_audio_bytes_no_input(self):
        audio_bytes = synthesize_text_to_audio_bytes("")
        self.assertIsNone(audio_bytes)

    @patch('scipy.io.wavfile.read')
    def test_play_audio_bytes_wav(self, mock_scipy_read_wav):
        dummy_wav_data = np.array([1, 2, 3], dtype=np.int16)
        dummy_sample_rate = AUDIO_SAMPLE_RATE
        mock_scipy_read_wav.return_value = (dummy_sample_rate, dummy_wav_data)

        wav_bytes = self.generate_dummy_wav_bytes()
        play_audio_bytes(wav_bytes, output_format="wav", sample_rate=dummy_sample_rate)

        mock_scipy_read_wav.assert_called_once()
        self.assertIsInstance(mock_scipy_read_wav.call_args[0][0], io.BytesIO)
        mock_sd_module.play.assert_called_once_with(dummy_wav_data, samplerate=dummy_sample_rate)
        mock_sd_module.wait.assert_called_once()

    @patch('app.audio_utils.time.sleep')
    def test_play_audio_bytes_mp3_placeholder(self, mock_time_sleep):
        mp3_bytes = b"dummy_mp3_data"
        play_audio_bytes(mp3_bytes, output_format="mp3")
        mock_time_sleep.assert_called()

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
