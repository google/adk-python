import unittest
import sys
import os
from unittest.mock import MagicMock

# --- Global mock for sounddevice ---
# This must be done before any module (app code or test code) tries to import sounddevice
mock_sd_module = MagicMock(name="GlobalMockSoundDeviceModule")
# Make common sounddevice functions available on the mock if they are called directly
mock_sd_module.rec = MagicMock(name="sd_rec_mock")
mock_sd_module.wait = MagicMock(name="sd_wait_mock")
mock_sd_module.play = MagicMock(name="sd_play_mock")
mock_sd_module.query_devices = MagicMock(return_value=[{'name': 'mock_mic', 'max_input_channels': 1, 'default_samplerate': 44100.0}]) # Example
mock_sd_module.RawInputStream = MagicMock()
mock_sd_module.InputStream = MagicMock()
mock_sd_module.OutputStream = MagicMock()
mock_sd_module.Stream = MagicMock()


sys.modules['sounddevice'] = mock_sd_module
# --- End global mock ---

# Add project root to sys.path to allow imports from app
# This should come AFTER the global mock is inserted into sys.modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    # Discover and run tests in the 'tests' directory
    # Assumes this script is in streaming-test/ (which is the project root for these tests)
    print("Attempting to discover and run tests...")
    loader = unittest.TestLoader()
    try:
        suite = loader.discover('tests')
        print(f"Test suite loaded: {suite.countTestCases()} tests found.")
    except Exception as e:
        print(f"Error during test discovery: {e}")
        sys.exit(1)

    runner = unittest.TextTestRunner(verbosity=2) # Increased verbosity
    result = runner.run(suite)

    # Exit with a non-zero status if any tests failed
    if not result.wasSuccessful():
        sys.exit(1)
    sys.exit(0)
