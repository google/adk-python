import cv2
import numpy as np
import time
from app.config import VIDEO_FPS # Placeholder for now

# --- Video Capture ---
class VideoMonitor:
    def __init__(self, camera_index=0, fps_limit=None):
        """
        Initializes the VideoMonitor.

        Args:
            camera_index: The index of the camera to use (e.g., 0 for default).
            fps_limit: Limit processing to this many FPS. None for no limit (process as fast as camera provides).
        """
        self.camera_index = camera_index
        self.cap = None
        self.previous_frame = None
        self.last_processed_time = 0
        self.fps_limit = fps_limit if fps_limit else VIDEO_FPS # Default to VIDEO_FPS from config if not provided
        self.cv2 = cv2 # Store cv2 module reference for imencode in web_interface

        # For person detection (optional, basic example)
        # self.hog = cv2.HOGDescriptor()
        # self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # More robust change detection might involve background subtraction
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)


    def start_capture(self):
        """Starts video capture from the camera."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open video stream from camera index {self.camera_index}.")
            print("Make sure a camera is connected and accessible.")
            print("If using a virtual environment, ensure OpenCV can access camera drivers.")
            self.cap = None # Ensure it's None if failed
            return False
        print(f"Video capture started on camera index {self.camera_index}.")
        self.previous_frame = None # Reset previous frame on start
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False) # Re-initialize
        return True

    def stop_capture(self):
        """Stops video capture and releases the camera."""
        if self.cap:
            self.cap.release()
            self.cap = None
            print("Video capture stopped.")
        self.previous_frame = None

    def get_frame(self):
        """
        Retrieves a single frame from the camera.

        Returns:
            A numpy array representing the frame, or None if capture failed.
        """
        if not self.cap or not self.cap.isOpened():
            # print("Capture not started or failed.")
            return None

        ret, frame = self.cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            return None
        return frame

    def detect_significant_change(self, frame: np.ndarray, threshold_abs_diff=30, min_contour_area=500) -> tuple[bool, str, bytes | None]:
        """
        Detects a significant change in the video frame using background subtraction and motion.

        Args:
            frame: The current video frame.
            threshold_abs_diff: Sensitivity for absolute difference method. (Not used with MOG2 directly)
            min_contour_area: Minimum area of a contour to be considered significant motion.


        Returns:
            A tuple: (has_changed: bool, change_description: str, processed_frame_bytes: bytes | None).
            processed_frame_bytes will be the JPEG encoded frame if a change is detected.
        """
        if frame is None:
            return False, "No frame", None

        # 1. Background Subtraction (more robust than simple frame diff)
        fg_mask = self.bg_subtractor.apply(frame)

        # 2. Noise reduction - Apply morphological opening
        kernel = np.ones((5,5),np.uint8)
        fg_mask_opened = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask_closed = cv2.morphologyEx(fg_mask_opened, cv2.MORPH_CLOSE, kernel)


        # 3. Find contours of moving objects
        contours, _ = cv2.findContours(fg_mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        significant_motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > min_contour_area:
                significant_motion_detected = True
                # (x, y, w, h) = cv2.boundingRect(contour)
                # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Draw on original for viz
                break # Found one significant motion

        if significant_motion_detected:
            change_description = "Significant motion detected in the scene."
            # Encode the current frame to JPEG bytes to send to the model
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes() # This is correct. imencode returns a numpy array.
                return True, change_description, frame_bytes
            else:
                return True, change_description, None # Encoding failed

        return False, "No significant change detected.", None


    def process_frame_for_changes(self) -> tuple[bool, str, bytes | None]:
        """
        Captures a frame and detects significant changes. Limits processing by FPS.

        Returns:
            Tuple: (change_detected, change_description, frame_bytes_if_change)
        """
        current_time = time.time()
        if self.fps_limit and (current_time - self.last_processed_time) < (1.0 / self.fps_limit):
            # Skip processing this frame to maintain FPS limit
            time.sleep(0.01) # Small sleep to prevent busy-looping if camera is faster
            return False, "FPS limit, frame skipped", None

        self.last_processed_time = current_time

        frame = self.get_frame()
        if frame is None:
            return False, "Failed to get frame", None

        return self.detect_significant_change(frame)


    # Optional: Person detection (can be slow)
    # def detect_persons(self, frame: np.ndarray) -> tuple[bool, str, np.ndarray | None]:
    #     if frame is None:
    #         return False, "No frame for person detection", None

    #     (rects, weights) = self.hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

    #     if len(rects) > 0:
    #         # For now, just signal if any person is detected.
    #         # Could be enhanced to count, track entries/exits relative to a previous state.
    #         # for (x, y, w, h) in rects:
    #         #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2) # Draw on original for viz

    #         change_description = f"{len(rects)} person(s) detected in the view."
    #         ret, buffer = cv2.imencode('.jpg', frame) # Send frame with detected people
    #         if ret:
    #             frame_bytes = buffer.tobytes() # Corrected
    #             return True, change_description, frame_bytes
    #         else:
    #             return True, change_description, None
    #     return False, "No persons detected.", None


if __name__ == '__main__':
    print("Testing video_utils.py...")
    # This test requires a camera. If no camera, it will try to proceed but fail gracefully.
    monitor = VideoMonitor(camera_index=0, fps_limit=1) # Limit to 1 FPS for this test

    if not monitor.start_capture():
        print("Failed to start camera capture. Ensure a camera is connected and drivers are working.")
        print("If running in a headless environment, this test might not be fully effective without a virtual camera.")
    else:
        print("Camera capture started. Will monitor for changes for a few seconds.")
        start_time = time.time()
        frames_processed = 0
        try:
            while time.time() - start_time < 10: # Run for 10 seconds
                changed, description, frame_bytes = monitor.process_frame_for_changes()
                if changed:
                    print(f"Change detected: {description}")
                    if frame_bytes:
                        print(f"Frame data size: {len(frame_bytes)} bytes")
                        # with open("detected_change.jpg", "wb") as f: # Option to save frame
                        #     f.write(frame_bytes)
                        # print("Saved detected_change.jpg")
                elif description == "Failed to get frame":
                    print("Stopping test due to frame capture failure.")
                    break
                elif description != "FPS limit, frame skipped": # Don't flood with "no change"
                     print(f"Status: {description} (Frame: {frames_processed})")

                frames_processed +=1
                # A short sleep is important if not FPS limited in process_frame_for_changes,
                # or if camera is very fast. Here it's mainly for responsive exit.
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("Test interrupted by user.")
        finally:
            monitor.stop_capture()
            print("Video utils testing finished.")
