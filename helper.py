from ultralytics import YOLO
import time
import streamlit as st # type: ignore
import cv2
from pytube import YouTube # type: ignore
import settings
import torch
from ultralytics import YOLO
import app




def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    return model_



def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='دیاریکردنی تاسە',
                   channels="BGR",
                   use_column_width=True
                   )


def play_youtube_video(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_youtube = st.sidebar.text_input("لینکی ڤیدیۆی یوتووب")

    # is_display_tracker, tracker = display_tracker_options()

    if st.sidebar.button('دیاریکردنی تاسە'):
        try:
            yt = YouTube(source_youtube)
            stream = yt.streams.filter(file_extension="mp4", res=720).first()
            vid_cap = cv2.VideoCapture(stream.url)

            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error(":کێشە لە بارکردنی ڤیدیۆ هەیە" + str(e))


def play_rtsp_stream(conf, model):
    """
    Plays an rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption('Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error loading RTSP stream: " + str(e))




# def play_webcam(conf, model):
#     """
#     Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

#     Parameters:
#         conf: Confidence of YOLOv8 model.
#         model: An instance of the `YOLOv8` class containing the YOLOv8 model.

#     Returns:
#         None

#     Raises:
#         None
#     """
#     # Replace with the appropriate webcam index for your system
#     source_webcam = 0
#     # is_display_tracker, tracker = display_tracker_options()
#     if st.sidebar.button('Detect Objects'):
#         try:
#             vid_cap = cv2.VideoCapture(source_webcam)
#             st_frame = st.empty()  # Create an empty container

#             while (vid_cap.isOpened()):
#                 success, image = vid_cap.read()
#                 if success:
#                     _display_detected_frames(conf,
#                                             model,
#                                             st_frame,  # Update the container with the new frame
#                                             image,
#                                             )
#                 else:
#                     vid_cap.release()
#                     break
#         except Exception as e:
#             st.sidebar.error("کێشە لە کردنەوەی وێبکام: " + str(e))


#####################

# def play_webcam(conf, model):
#     """
#     Plays a webcam stream, detects objects in real-time using the YOLOv5 model,
#     and displays the results in a Streamlit app.

#     Parameters:
#         conf (float): Confidence threshold for object detection.
#         model (YOLOv5): An instance of the `Detector` class containing the YOLOv5 model.

#     Returns:
#         None
#     """

#     # Replace with the appropriate webcam index for your system
#     source_webcam = 0  # Assuming webcam 0 is your default

#     while True:
#         try:
#             # Capture video from webcam
#             vid_cap = cv2.VideoCapture(source_webcam)

#             # Create an empty Streamlit container for the frame
#             st_frame = st.empty()

#             if not vid_cap.isOpened():
#                 print("Error opening webcam. Please check your webcam connection.")
#                 break

#             while (vid_cap.isOpened()):
#                 success, image = vid_cap.read()

#                 if not success:
#                     print("Error reading frame from webcam.")
#                     break

#                 # Perform object detection using your YOLOv5 model
#                 results = model(image)  # Assuming YOLOv5 model returns results

#                 # Extract detected objects and filter by confidence threshold
#                 detected_objects = [obj for obj in results.pandas().xyxy[0] if obj['confidence'] >= conf]

#                 # Display the frame with detected objects (replace with your visualization logic)
#                 display_detected_frames(st_frame, image, detected_objects)

#                 # Add a brief pause to prevent overwhelming processing
#                 cv2.waitKey(1)

#             vid_cap.release()

#         except Exception as e:
#             st.sidebar.error("Error: " + str(e))
#             break





# def display_detected_frames(st_frame, image, detected_objects):
#     """
#     This function displays the image frame with detected objects overlaid.

#     Parameters:
#         st_frame (streamlit.delta_extractor.DeltaExtractor): The Streamlit container for the frame.
#         image (numpy.ndarray): The image frame to be displayed.
#         detected_objects (list): A list of detected objects, where each object is a dictionary
#               containing information like 'name', 'confidence', and 'bbox' (bounding box coordinates).
#     """

#     # Draw bounding boxes and labels on the image
#     for obj in detected_objects:
#         class_name = obj.get('name', 'Unknown')  # Handle potential missing 'name' key
#         confidence = obj.get('confidence', 0.0)  # Handle potential missing 'confidence' key
#         bbox = obj.get('bbox', [0, 0, 0, 0])  # Handle potential missing 'bbox' key

#         # Adjust bounding box coordinates (if necessary) and draw on the image
#         # (Assuming YOLOv5 bbox format: [xmin, ymin, xmax, ymax])
#         x_min, y_min, x_max, y_max = bbox
#         cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
#         cv2.putText(image, f"{class_name}: {confidence:.2f}", (x_min, y_min - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#     # Display the image in Streamlit
#     st_frame.image(image, channels="BGR")  # Assuming BGR format for OpenCV

# # Load your YOLOv5 model (replace with your model path)
# model_path = "models/yolov5s.pt"  # Replace with your model file path
# # model = Detector(model_path)  # Create the YOLOv5 model instance
# # model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
# # Set confidence threshold (adjust as needed)
# confidence_threshold = 0.5

# # Call the play_webcam function
# # play_webcam(confidence_threshold, model)