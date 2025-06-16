import cv2 # OpenCV library for computer vision tasks
import numpy as np # For numerical operations, especially with images
import time # For handling FPS limiting

# Attempt to import VIDEO_FPS from app.config
# If app.config or VIDEO_FPS is not found, use a default value.
try:
    from app.config import VIDEO_FPS
except ImportError:
    print("VIDEO_UTILS: Could not import VIDEO_FPS from app.config. Using default value 1.")
    VIDEO_FPS = 1


# --- Video Capture Class ---
class VideoMonitor:
    """
    VideoMonitor encapsulates operations related to capturing and processing
    video frames from a camera. It can start and stop video capture, retrieve
    individual frames, and detect significant changes or motion between frames.
    """
    def __init__(self, camera_index=0, fps_limit=None):
        """
        Initializes the VideoMonitor.

        Args:
            camera_index (int, optional): The index of the camera to use (e.g., 0 for default).
                                        Defaults to 0.
            fps_limit (int or None, optional): Limit processing to this many frames per
                                               second. None for no limit (process as
                                               fast as camera provides).
                                               Defaults to `VIDEO_FPS` from `config.py` if None is provided.
        """
        self.camera_index = camera_index  # Store the camera index
        self.cap = None  # Initialize the capture object to None; opened in `start_capture`
        self.previous_frame = None # Stores the previous frame for change detection
        self.last_processed_time = 0  # Timestamp of the last processed frame, for FPS limiting

        # Set the FPS limit. If `fps_limit` is not provided (None), use the VIDEO_FPS value
        # from the `app.config` file (or its default if import failed). This allows a centralized default FPS configuration.
        self.fps_limit = fps_limit if fps_limit is not None else VIDEO_FPS

        # Store a reference to the `cv2` module, primarily for using `cv2.imencode` without
        # having to import it in other files that might use this class.
        self.cv2 = cv2

        # Optional: Hog Feature Descriptor for person detection. Commented out as it
        # can be slow and is not tthe rimary focus of this module.
        # self.hog = cv2.HOGDescriptor()
        # self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Background subtractor: This is a common OpenCV technique for detecting motion.
        # It builds a model of the static background and then compares new frames to it.
        #   - `history`: Number of frames used to build the background model.
        #   - `varThreshold`: Threshold on the Mahalanobis distance to classify a pixel as foreground.
        #   - `detectShadows`: Whether to detect and mark shadows. Set to False as shadows can be
        #     incorrectly detected as foreground objects.
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    def start_capture(self):
        """
        Starts video capture from the specified camera.
        It initializes the OpenCV `cv2.VideoCapture` object.

        Returns:
            bool: True if capture started successfully, False otherwise.
        """
        # Create a VideoCapture object. The argument is the camera index.
        self.cap = cv2.VideoCapture(self.camera_index)

        # Check if the camera was opened successfully.
        if not self.cap.isOpened():
            print(f"Error: Could not open video stream from camera index {self.camera_index}.")
            print("Make sure a camera is connected and accessible.")
            print("If using a virtual environment, ensure OpenCV can access camera drivers.")
            self.cap = None  # Ensure `self.cap` is set to None if the operation failed.
            return False

        print(f"Video capture started on camera index {self.camera_index}.")
        # Reset the previous frame when starting a new capture session.
        self.previous_frame = None
        # Re-initialize the background subtractor to ensure it has a fresh background
        # history when capture starts.
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        return True

    def stop_capture(self):
        """
        Stops video capture and releases the camera resource.
        """
        if self.cap:
            self.cap.release()  # Release the camera.
            self.cap = None  # Set the capture object to None.
            print("Video capture stopped.")
        # Reset the previous frame as it's no longer relevant.
        self.previous_frame = None

    def get_frame(self):
        """
        Retrieves a single frame from the camera.

        Returns:
            np.ndarray: A numpy array representing the frame, or None if capture failed.
        """
        # Check if the capture object is valid and the camera is open.
        if not self.cap or not self.cap.isOpened():
            # print("Capture not started or failed.") # Can be noisy if called frequently.
            return None

        # Read a frame from the camera.
        #   - `ret`: A boolean indicating if the frame was read successfully.
        #   - `frame`: The captured frame (a np.ndarray array).
        ret, frame = self.cap.read()

        # If the frame was not received successfully (e.g., end of a video file).
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            return None
        return frame

    def detect_significant_change(self, frame: np.ndarray, threshold_abs_diff=30, min_contour_area=500) -> tuple[bool, str, bytes | None]:
        """
        Detects a significant change in the video frame using background subtraction,
        morphological operations for noise reduction, and contour analysis for motion.
        Color conversion to grayscale is not used here as background subtraction
        often works better on color frames directly or has built-in mechanisms.

        Args:
            frame (np.ndarray): The current video frame.
            threshold_abs_diff (int, optional): Sensitivity for absolute difference
                method. (Not used directly with MOG2 but kept for potential alternatives).
            min_contour_area (int, optional): Minimum area of a contour to be considered
                as significant motion. Defaults to 500.

        Returns:
            tuple[bool, str, bytes | None]:
                - has_changed (bool): True if a significant change was detected, False otherwise.
                - change_description (str): A textual description of the change.
                - processed_frame_bytes (bytes | None): The JPEG encoded frame if
                  a change is detected, otherwise None.
        """
        if frame is None:
            return False, "No frame", None

        # 1. Background Subtraction
        # The bg_subtractor.apply() method computes a foreground mask.
        # The mask consists of white pixels for foreground and black for background.
        fgmask = self.bg_subtractor.apply(frame)

        # 2. Noise Reduction
        # Morphological operations are used to reduce noise and improve the foreground mask.
        # - `cv2.MORPH_OPEN`: Removes small white noise from the foreground.
        #   It is an erosion followed by a dilation.
        # kernel: A small matrix used for the morphological operations.
        kernel = np.ones((5,5), np.uint8)
        fgmask_opened = self.cv2.morphologyEx(fgmask, self.cv2.MORPH_OPEN, kernel)
        # - `cv2.MORPH_CLOSE`: Fills small holes in the foreground objects.
        #   It is a dilation followed by an erosion.
        fgmask_closed = cv2.morphologyEx(fgmask_opened, self.cv2.MORPH_CLOSE, kernel)

        # 3. Find Contours of Moved Objects
        # Contours are the boundaries of the white regions (foreground objects) in the mask.
        #   - `cv2.RETR_EXTERNAL`: Retrieves only the extreme outer contours.
        #   - `cv2.CHAIN_APPROX_SIMPLE`: Compresses horizontal, vertical, and diagonal segments
        #     and leaves only their end points. For example, a rectangle's is encoded with 4 points.
        contours, _ = cv2.findContours(fgmask_closed, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE)

        significant_motion_detected = False
        for contour in contours:
            # Calculate the area of each contour.
            if cv2.contourArea(contour) > min_contour_area:
                # If the area is greater than the threshold, consider it significant motion.
                significant_motion_detected = True
                # Optional: Draw bounding box on the original frame for visualization or describe as "Visual motion"
                # description.
                # (x, y, w, h) = cv2.boundingRect(contour)
                # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255,0), 2) # Draw on original for viz
                break # Found one significant motion, so no need to check further contours.

        if significant_motion_detected:
            change_description = "Significant motion detected in the scene."
            # Encode the current frame to JPEG bytes to send to the model.
            #   - `ret`: True if encoding was successful.
            #   - `buffer`: The numpy array containing the encoded bytes.
            ret, buffer = self.cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()  # Convert the numpy array to bytes.
                return True, change_description, frame_bytes
            else:
                print("VideoMonitor: Error encoding frame to JPEG.")
                return True, change_description, None # Encoding failed.

        # If no significant motion was detected.
        return False, "No significant change detected.", None


    def process_frame_for_changes(self) -> tuple[bool, str, bytes | None]:
        """
        Captures a frame from the camera, detects significant changes, and limits
        the processing rate according to the `self.fps_limit`.

        Returns:
            tuple(bool, str, bytes | None):
                - change_detected (bool): True if a change was detected, False otherwise.
                - change_description (str): A description of the change or a status message.
                - frame_bytes_if_change (bytes | None): The JPEG encoded frame bytes
                  if a change was detected, otherwise None.
        """
        current_time = time.time()

        # FPS limiting logic
        if self.fps_limit and (current_time - self.last_processed_time) < (1.0 / self.fps_limit):
            # If the time since the last processed frame is less than the desired frame
            # interval (calculated from fps_limit), then skip processing this frame.
            time.sleep(0.01)  # Add a small sleep to prevent a busy-loop if the camera is much
                             # faster than the fps_limit, or if the camera capture itself is non-blocking.
            return False, "FPS limit, frame skipped", None

        self.last_processed_time = current_time  # Update the last processed time

        frame = self.get_frame()  # Get a frame from the camera
        if frame is None:
            return False, "Failed to get frame", None

        # Detect changes in the frame.
        return self.detect_significant_change(frame)


    # Optional: Person detection method.
    # This is commented out because it can be slow and computationally intensive.
    # It uses a Pre-trained HOG (Histogram of Oriented Gradients) detector.
    # def detect_persons(self,frame: np.ndarray) -> tuple[bool, str, np.ndarray | None]:
    #     if frame is None:
    #         return False, "No frame for person detection", None

    #     # To speed up detection, resize the frame to decrease the number of pixels to be processed
    #     frame_resized = cv2.resize(frame, (640, 480))
    #     # (rects, weights) = self.hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

    #     if len(rects) > 0:
    #         # For now, just signal if any person is detected.
    #         # Could be enhanced to count, track entries/exits relative to a previous state.
    #         # for (x, y, w, h) in rects:
    #         #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2) # Draw on original for viz

    #         change_description = f"{len(rects)} person(s) detected in the view."
    #         ret, buffer = self.cv2.imencode('.jpg', frame) # Send frame with detected people
    #         if ret:
    #             frame_bytes = buffer.tobytes() # Corrected
    #             return True, change_description, frame_bytes
    #         else:
    #             return True, change_description, None
    #     return False, "No persons detected.", None

# This block allows the script to be run directly for testing.
def main_test():
    print("Testing video_utils.py...")
    # This test requires a camera. If no camera, it will try to proceed but fail gracefully.
    # The VideoMonitor class is defined in the same file, so it's in scope here.
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
                    print(f'Change detected: {description}')
                    if frame_bytes:
                        print(f"Frame data size: {len(frame_bytes)} bytes")
                        # with open("detected_change.jpg", "wb") as f: # Option to save frame
                        #     f.write(frame_bytes)
                        # print("Saved detected_change.jpg")
                elif description == "Failed to get frame":
                    print("Stopping test due to frame capture failure.")
                    break
                elif description != "FPS limit, frame skipped": # Don't print "FPS limit" too often, but other status messages
                    print(f'Status: {description} (Frame: {frames_processed})')

                frames_processed +=1
                # A short sleep is important if not FPS limited in process_frame_for_changes,
                # if camera is very fast. Here it's mainly for responsive exit.
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("Test interrupted by user.")
        finally:
            monitor.stop_capture()
            print("Video utils testing finished.")

# The `if __name__ == '__main__':` block ensures that `main_test()` is called only when the script is executed directly,
# not when it's imported as a module. This is a standard Python practice.
# By the time the `if __name__ == '__main__':` block is executed,
# the class definition has already been processed and the class name is in the module global scope.
# The original error (NameErro) often occurs in more complex scenarios like circular imports
# or if the runnable block tries to access the class in a way that depends on import-at time side-effects
# that aren't complete when the `if __name__ == '__main__':` check happens.
# Wrapping it in a function is a good practice for cleanliness and to avoid potential scoping issues,
# especially with tools like Uvicorn's reloader that might import modules in specific ways.
if __name__ == '__main__':
    main_test()
