import os

# Change this to your absolute path to handtracking project folder
BASE_PATH = "/home/name/custom_handtracker"

PATH_TO_MODEL = os.path.sep.join(
    [BASE_PATH,"frozen_weights/frozen_inference_graph.pb"])
PATH_TO_LABELS = os.path.sep.join(
    [BASE_PATH,"frozen_weights/hand_label_map.pbtxt"])

# Project specific Parameters [Do not need to change]
NUM_CLASSES = 1
NUM_HANDS_TO_DETECT = 2

# Detection score threshold
SCORE_THRESH = 0.25

# Object identification parameters
# Refer to Class CentroidTracker in hand_tracker.py for explanations
# Unit: (frames, pixels, pixels)
FORGET_THRESH = 20
MAX_MOVEMENT = 400
MIN_DISTANCE = 300

# Movement parameters
# Unit: (frames, % of image, % of image)
# Compare dx and dy this number of frames before
PREV_FRAMES_COMPARE = 5
# Threshold to trigger a movement
CHANGEX_THRESH = 50
CHANGEY_THRESH = 50
