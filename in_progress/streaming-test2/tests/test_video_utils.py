import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import cv2 # OpenCV is a direct dependency here
import time # Import the time module

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.video_utils import VideoMonitor
from app.config import VIDEO_FPS

class TestVideoUtils(unittest.TestCase):

    def create_dummy_frame(self, width=640, height=480, channels=3, color=(0,0,0)):
        frame = np.full((height, width, channels), color, dtype=np.uint8)
        return frame

    @patch('app.video_utils.cv2.VideoCapture')
    def test_videomonitor_start_capture_success(self, MockVideoCapture):
        mock_cap_instance = MockVideoCapture.return_value
        mock_cap_instance.isOpened.return_value = True

        monitor = VideoMonitor(camera_index=0)
        self.assertTrue(monitor.start_capture())
        MockVideoCapture.assert_called_once_with(0)
        self.assertIsNotNone(monitor.cap)
        self.assertTrue(monitor.cap.isOpened())
        monitor.stop_capture() # Clean up

    @patch('app.video_utils.cv2.VideoCapture')
    def test_videomonitor_start_capture_failure(self, MockVideoCapture):
        mock_cap_instance = MockVideoCapture.return_value
        mock_cap_instance.isOpened.return_value = False # Simulate camera not opening

        monitor = VideoMonitor(camera_index=0)
        self.assertFalse(monitor.start_capture())
        self.assertIsNone(monitor.cap)

    def test_videomonitor_get_frame_no_capture(self):
        monitor = VideoMonitor()
        # No call to start_capture
        self.assertIsNone(monitor.get_frame())

    @patch('app.video_utils.cv2.VideoCapture')
    def test_videomonitor_get_frame_success(self, MockVideoCapture):
        mock_cap_instance = MockVideoCapture.return_value
        mock_cap_instance.isOpened.return_value = True
        dummy_frame_data = self.create_dummy_frame()
        mock_cap_instance.read.return_value = (True, dummy_frame_data)

        monitor = VideoMonitor()
        monitor.start_capture() # Sets up monitor.cap
        frame = monitor.get_frame()

        self.assertIsNotNone(frame)
        np.testing.assert_array_equal(frame, dummy_frame_data)
        mock_cap_instance.read.assert_called_once()
        monitor.stop_capture()

    @patch('app.video_utils.cv2.VideoCapture')
    def test_videomonitor_get_frame_read_fail(self, MockVideoCapture):
        mock_cap_instance = MockVideoCapture.return_value
        mock_cap_instance.isOpened.return_value = True
        mock_cap_instance.read.return_value = (False, None) # Simulate read failure

        monitor = VideoMonitor()
        monitor.start_capture()
        frame = monitor.get_frame()

        self.assertIsNone(frame)
        monitor.stop_capture()

    def test_detect_significant_change_no_frame(self):
        monitor = VideoMonitor()
        changed, desc, _ = monitor.detect_significant_change(None)
        self.assertFalse(changed)
        self.assertEqual(desc, "No frame")

    @patch('cv2.imencode', return_value=(True, np.array([1,2,3], dtype=np.uint8))) # Corrected mock target and return value
    def test_detect_significant_change_with_motion(self, mock_imencode):
        monitor = VideoMonitor()
        # monitor.start_capture() # Initializes bg_subtractor - not needed if only testing detect_significant_change directly with a frame
        # Let's ensure bg_subtractor is available for this test method if it's not through start_capture
        monitor.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)


        frame1 = self.create_dummy_frame(color=(10,10,10)) # Initial background
        monitor.bg_subtractor.apply(frame1) # establish baseline

        frame2 = self.create_dummy_frame(color=(10,10,10))
        # Simulate a moving object by drawing a different color rectangle
        cv2.rectangle(frame2, (100,100), (200,200), (200,200,200), -1)

        changed, desc, frame_bytes = monitor.detect_significant_change(frame2, min_contour_area=100)

        self.assertTrue(changed, f"Change should have been detected. Description: {desc}")
        self.assertEqual(desc, "Significant motion detected in the scene.")
        self.assertIsNotNone(frame_bytes)
        mock_imencode.assert_called_once()
        # monitor.stop_capture() # Only if start_capture was called

    def test_detect_significant_change_no_motion(self):
        monitor = VideoMonitor()
        # monitor.start_capture() # Initializes bg_subtractor
        monitor.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)


        frame1 = self.create_dummy_frame(color=(10,10,10))
        monitor.bg_subtractor.apply(frame1) # establish baseline

        # Process same frame again or very similar one
        frame2 = self.create_dummy_frame(color=(10,10,10))
        # Make a tiny change that should be filtered out by morphology or area
        cv2.rectangle(frame2, (0,0), (2,2), (20,20,20), -1)


        changed, desc, _ = monitor.detect_significant_change(frame2, min_contour_area=500)

        self.assertFalse(changed, f"No significant change should be detected. Description: {desc}")
        self.assertEqual(desc, "No significant change detected.")
        # monitor.stop_capture()

    @patch('app.video_utils.VideoMonitor.get_frame')
    @patch('app.video_utils.VideoMonitor.detect_significant_change')
    def test_process_frame_for_changes_fps_limit(self, mock_detect_change, mock_get_frame):
        monitor = VideoMonitor(fps_limit=10) # Limit to 10 FPS
        monitor.last_processed_time = time.time() # Pretend we just processed a frame

        # Call immediately, should be skipped due to FPS limit
        changed, desc, _ = monitor.process_frame_for_changes()

        self.assertFalse(changed)
        self.assertEqual(desc, "FPS limit, frame skipped")
        mock_get_frame.assert_not_called()
        mock_detect_change.assert_not_called()

    @patch('app.video_utils.VideoMonitor.get_frame')
    @patch('app.video_utils.VideoMonitor.detect_significant_change')
    def test_process_frame_for_changes_proceeds(self, mock_detect_change, mock_get_frame):
        monitor = VideoMonitor(fps_limit=1)
        monitor.last_processed_time = time.time() - 2 # Pretend last frame was 2s ago

        dummy_frame = self.create_dummy_frame()
        mock_get_frame.return_value = dummy_frame
        mock_detect_change.return_value = (True, "Test change", b"frame_bytes")

        changed, desc, frame_bytes_out = monitor.process_frame_for_changes()

        self.assertTrue(changed)
        self.assertEqual(desc, "Test change")
        self.assertEqual(frame_bytes_out, b"frame_bytes")
        mock_get_frame.assert_called_once()
        mock_detect_change.assert_called_once_with(dummy_frame)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
