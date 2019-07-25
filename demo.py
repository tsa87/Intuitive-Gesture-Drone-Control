from helper.config import hand_config as config
from helper import detector_utils as utils
from helper import CentroidTracker
import numpy as np
import argparse
import datetime
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v","--video", dest='video_path')
args = ap.parse_args()

# load the pre-trained model
detection_graph, sess = utils.initalize_model()

# initalize the tracker
left_right_tracker = CentroidTracker()

# decide between webcam or video file
if args.video_path is None:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(args.video_path)

# initalize starting time for fps calculations
start_time = datetime.datetime.now()
num_frames = 0

while (cap.isOpened()):
    grabbed, frame = cap.read()
    # end the video if there are no more frames.
    if not grabbed:
        break

    width, height = (cap.get(3), cap.get(4))

    boxes, scores = utils.detect_objects(frame, detection_graph, sess)

    rects = utils.return_final_predictions(
        config.NUM_HANDS_TO_DETECT,
        config.SCORE_THRESH,
        scores,
        boxes,
        width,
        height)

    utils.draw_bounding_boxes(frame, rects)
    utils.track_objects(frame, rects, left_right_tracker)
    utils.display_movement(frame, left_right_tracker)

    cv2.imshow("video",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
